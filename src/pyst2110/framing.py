"""What a chunk says about the frames it carries and the loss between them.

Two things span acquisitions, so two things hold state (§spec:interface-shape):
a sequence tracker carrying the last number it saw, so a gap straddling two
chunks is counted once, and a frame tracker carrying the last marker bit, so a
frame ending on a chunk boundary does not end twice. Where a payload belongs
needs no state and lives in :mod:`pyst2110.payload`.

Seeded from an earlier per-packet implementation, with the Python loop it
used to widen its span replaced by cumulative array operations
— a 4K60 flow approaches 250,000 packets a second (§req:priorities).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyst2110 import _chunk

__all__ = ["FrameTracker", "SequenceTracker", "frame_boundaries", "frame_starts"]

_SEQUENCE_SPACE = 1 << 16

# ST 2110-20 section 6.1.5: an interlaced frame is sent as two fields, so an
# interlaced flow marks twice per frame.
_FIELDS_PER_FRAME = 2

# RFC 3550 Appendix A.1's constants, for the sixteen-bit RTP field alone. A
# forward step below MAX_DROPOUT is an ordinary gap; a step landing within
# MAX_MISORDER behind is reordering or a duplicate; anything else is too large
# to read as either.
#
# The boundary between the last two is one value from A.1's: a step of exactly
# MAX_MISORDER back reads here as reordering, where A.1 reads it as a jump.
# A.1 validates each packet before admitting it; this widens a cumulative
# span, where admitting one more late packet is the cheaper mistake.
#
# Not "half the sequence space": at ST 2110-20 rates a flow runs at some 3e5
# packets a second, so a tenth-of-a-second outage exceeds 32768 packets and a
# half-space rule reads it as one step backwards and zero loss — the metric
# goes quiet exactly when it matters.
_MAX_DROPOUT = 3000
_MAX_MISORDER = 100

# With the RFC 4175 extended sequence number in hand the space is 2^32, where
# the same ambiguity is four hours off rather than a fifth of a second. Half
# that space is the dropout bound there, so every outage worth measuring is a
# plain gap and only a sender that restarted its numbering reads as a resync.
_EXTENDED_SPACE = 1 << 32
_EXTENDED_MAX_DROPOUT = _EXTENDED_SPACE // 2


def frame_boundaries(
    marker: NDArray[np.bool_], previous: bool = False
) -> NDArray[np.bool_]:
    """Which packets end a frame, from the marker bit's rising edge.

    ST 2110-20 section 6.1.2 sets the marker on the last packet of a frame —
    of a *field*, when interlaced — so a rising edge ends one and the packet
    after it begins the next. The edge rather than the bit is what keeps a
    duplicated final packet from ending a frame twice (§spec:rtp).

    ``previous`` is the marker bit of the packet before this chunk, which is
    what carries that across a chunk boundary.
    """
    flags = _chunk.per_packet(marker, "marker", dtype=np.bool_)
    prior = np.concatenate(([previous], flags[:-1]))
    return flags & ~prior


def frame_starts(
    marker: NDArray[np.bool_],
    *,
    interlaced: bool = False,
    field: NDArray[np.bool_] | None = None,
    previous: bool = False,
) -> NDArray[np.bool_]:
    """Which packets begin a frame, from the RTP marker column.

    The packet after a marker's rising edge begins the next marked unit
    (§spec:rtp) — a frame progressive, a field interlaced. ``previous`` is
    the marker bit of the packet before this chunk; without it the first
    packet cannot be known to start anything and is not claimed to.

    Interlaced, a frame is two units, as :class:`FrameTracker` counts them.
    ``field`` — the F bit of
    :func:`pyst2110.payload.parse_payload_headers` — says which unit starts
    a frame: the one whose first packet reads first-field. Without it the
    units are paired in arrival order from the first unit start, which is
    wrong when the capture opens on a second field; pass the F bit where the
    stream carries one.
    """
    ends = frame_boundaries(marker, previous=previous)
    if ends.size == 0:
        return np.zeros(0, dtype=np.bool_)
    starts: NDArray[np.bool_] = np.concatenate(([previous], ends[:-1]))
    if not interlaced:
        return starts
    if field is not None:
        first = _chunk.per_packet(
            field, "field flag", count=int(ends.size), plural="field flags"
        )
        paired: NDArray[np.bool_] = starts & ~first.astype(np.bool_)
        return paired
    keep = np.flatnonzero(starts)[::_FIELDS_PER_FRAME]
    paired = np.zeros(ends.shape, dtype=np.bool_)
    paired[keep] = True
    return paired


class FrameTracker:
    """Which frame each packet belongs to, continuing across chunks.

    The unit is the marker's: ST 2110-20 section 6.1.2 sets it on the last
    packet of a progressive frame, and on the last packet of each *field* of
    an interlaced one. So an interlaced flow counts two units per frame, while
    :func:`pyst2110.geometry.packets_per_frame` counts a whole frame's packets
    — one unit here is half of that.
    """

    def __init__(self) -> None:
        #: Marked units whose last packet has been seen: frames progressive,
        #: fields interlaced. The unit in progress is numbered ``frames``, so
        #: an index equal to it is not yet complete.
        self.frames = 0
        self._marked = False

    def observe(self, marker: NDArray[np.bool_]) -> NDArray[np.int64]:
        """Account for one chunk's marker column, in arrival order.

        Returns a packet-aligned index: how many marked units ended before
        each packet, counted from the first packet this tracker ever saw.
        """
        flags = _chunk.per_packet(marker, "marker", dtype=np.bool_)
        edges = frame_boundaries(flags, previous=self._marked)
        if flags.size == 0:
            return np.zeros(0, dtype=np.int64)

        # How many ended before each packet: the exclusive scan of the edges.
        index = self.frames + (np.cumsum(edges) - edges)
        self.frames += int(np.count_nonzero(edges))
        self._marked = bool(flags[-1])
        return index.astype(np.int64)


class SequenceTracker:
    """RTP sequence continuity across chunks, wraparound included.

    Loss is counted the way RFC 3550 Appendix A.3 defines it: the span between
    the lowest and highest numbers seen, less the number received. That form
    is invariant to reordering, so ``lost`` goes down as late packets arrive
    and is negative where duplicates arrived, exactly as the RFC describes.

    The arithmetic runs over the whole thirty-two-bit number wherever
    :meth:`observe` is given the RFC 4175 extended sequence, and over the
    sixteen-bit RTP field alone otherwise. A jump too large for the space in
    use to read as either a gap or a reordering closes the span and opens a
    new one, counted in ``resyncs`` and left out of ``lost`` (§spec:rtp).
    """

    def __init__(self) -> None:
        self.received = 0
        self.duplicated = 0
        self.reordered = 0
        self.discontinuities = 0
        self.resyncs = 0
        self._closed = 0
        self._lowest = 0
        self._highest = 0
        self._extended: int | None = None
        self._space = _SEQUENCE_SPACE
        self._dropout = _MAX_DROPOUT

    def observe(
        self,
        sequence_numbers: NDArray[np.integer[Any]],
        extended: NDArray[np.integer[Any]] | None = None,
    ) -> None:
        """Account for one chunk's sequence numbers, in arrival order.

        ``extended`` is the same packets' extended sequence numbers — the
        high sixteen bits, which
        :func:`pyst2110.payload.parse_payload_headers` returns. With it the
        gap arithmetic runs over the whole thirty-two-bit number; without it
        the sixteen-bit field is all there is, and a gap beyond MAX_DROPOUT
        becomes a resync rather than a loss.

        One tracker counts in one space: supply ``extended`` for every chunk
        or for none.
        """
        numbers = _chunk.per_packet(sequence_numbers, "number")
        if extended is not None:
            high = _chunk.per_packet(
                extended,
                "extended sequence number",
                count=int(numbers.size),
                plural="numbers",
            )
            numbers = (high << 16) | numbers
        if numbers.size == 0:
            return
        if extended is None:
            space, dropout = _SEQUENCE_SPACE, _MAX_DROPOUT
        else:
            space, dropout = _EXTENDED_SPACE, _EXTENDED_MAX_DROPOUT
        if self._extended is None:
            self._space, self._dropout = space, dropout
        elif space != self._space:
            raise ValueError(
                "the extended sequence number was supplied for some chunks and "
                "not others; one tracker counts in one sequence space"
            )
        self.received += int(numbers.size)

        if self._extended is None:
            # Nothing to step from, so the first packet only opens the span.
            previous = int(numbers[0])
            self._open(previous)
            landed = numbers[1:]
        else:
            # The running extended number, which is the last number seen plus
            # some multiple of the space — and a step is taken modulo it.
            previous = self._extended
            landed = numbers
        if landed.size == 0:
            return

        # Each packet's forward distance from the one before it. The previous
        # chunk's last number is the predecessor of this chunk's first, which
        # is what classifies a straddling step once and only there.
        befores = np.concatenate(([previous], landed[:-1]))
        steps = ((landed - befores) % self._space).astype(np.int64)

        self.discontinuities += int(np.count_nonzero(steps != 1))
        self.duplicated += int(np.count_nonzero(steps == 0))
        self.reordered += int(np.count_nonzero(steps >= self._space - _MAX_MISORDER))
        self._widen(landed, steps)

    def _widen(
        self,
        landed: NDArray[np.int64],
        steps: NDArray[np.int64],
    ) -> None:
        """Grow the observed span to cover this chunk, unwrapping as it goes.

        The span is kept in extended (unwrapped) numbers so the arithmetic is
        ordinary: a space that wraps makes every comparison on the raw value a
        special case.

        Vectorized between resyncs rather than per packet. A resync breaks the
        running total by construction, so the walk is over resyncs — normally
        none — and never over packets.
        """
        backward = steps >= self._space - _MAX_MISORDER
        resync = (steps >= self._dropout) & ~backward
        signed = np.where(backward, steps - self._space, steps)

        start = 0
        base = self._extended
        for cut in [*np.flatnonzero(resync).tolist(), landed.size]:
            if cut > start:
                extended = base + np.cumsum(signed[start:cut])
                self._lowest = min(self._lowest, int(extended.min()))
                self._highest = max(self._highest, int(extended.max()))
                base = int(extended[-1])
            if cut < landed.size:
                self.resyncs += 1
                self._reopen(int(landed[cut]))
                base = self._extended
                start = cut + 1
        self._extended = base

    def _open(self, number: int) -> None:
        self._lowest = self._highest = self._extended = number

    def _reopen(self, number: int) -> None:
        """Bank the span so far and start a new one at ``number``.

        Reached only from :meth:`_widen`, which runs after a span is open, so
        there is always one to bank.
        """
        self._closed += self._highest - self._lowest + 1
        self._open(number)

    @property
    def expected(self) -> int:
        """Packets the sender appears to have sent, by RFC 3550's definition."""
        if self._extended is None:
            return 0
        return self._closed + self._highest - self._lowest + 1

    @property
    def lost(self) -> int:
        """Expected less received. Negative is possible, and means duplicates."""
        return self.expected - self.received

    def summary(self) -> str:
        return (
            f"packets {self.received}, lost {self.lost}, "
            f"reordered {self.reordered}, duplicated {self.duplicated}, "
            f"discontinuities {self.discontinuities}, resyncs {self.resyncs}"
        )
