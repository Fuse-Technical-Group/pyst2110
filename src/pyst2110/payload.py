"""The RFC 4175 payload header, parsed across a chunk of packets.

Two octets of extended sequence number, then one to three line segments, laid
out in RFC 4175 section 4.2 and constrained by SMPTE ST 2110-20 section 6.1.4
(§spec:payload-header). The parse holds no state — a packet says where its own
payload belongs — which is what separates it from the trackers in
:mod:`pyst2110.framing`, its only sibling on this side of the RTP header.

A segment resolves to a descriptor: which line, where in it, and how many
octets. That is the whole of what this library produces about a payload
(§spec:scope-boundary).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyst2110 import _chunk, _layout

__all__ = ["PayloadHeaders", "parse_payload_headers"]

# SMPTE ST 2110-20 §6.1.4: "one, two, or three Sample Row Data (SRD) Headers".
# RFC 4175 itself sets no limit, so the bound is a parameter — but a walk over
# segments has to be bounded by something to stay vectorized across packets.
_ST2110_SEGMENTS = 3

# The conforming header's size and its word positions are `_layout`'s, stated
# once there beside the octet offsets they are halved from — a parse reading a
# field and a builder writing it have to agree, and two restatements agree
# only until one is edited (§spec:conforming-fast-path).
_CONFORMING_SIZE = _layout.CONFORMING_HEADER_SIZE
_EXTENDED_WORD = _layout.EXTENDED_SEQUENCE_WORD
_LENGTH_WORD = _layout.SRD_LENGTH_WORD
_ROW_WORD = _layout.SRD_ROW_WORD
_OFFSET_WORD = _layout.SRD_OFFSET_WORD
# In the field's own width, so masking one does not widen it back again.
_VALUE_MASK = np.uint16(_layout.VALUE_MASK)


@dataclass(frozen=True)
class PayloadHeaders:
    """One chunk's RFC 4175 payload headers, in two alignments.

    The descriptor fields are **segment-aligned**, packet-major, with
    :attr:`packet` mapping each one back to the row it came from; a packet
    carries as many as its continuation bits declared, so no packet-aligned
    array can hold them. Every other field is **packet-aligned**, as the rest
    of this library is (§spec:interface-shape).
    """

    # Packet-aligned.
    #: The high sixteen bits of the thirty-two-bit sequence number, which
    #: :meth:`pyst2110.framing.SequenceTracker.observe` counts loss over
    #: (§spec:rtp).
    extended_sequence: NDArray[np.int64]
    #: Descriptors this packet contributed. Zero where :attr:`overflowed`,
    #: whose descriptors are withheld.
    segments: NDArray[np.int64]
    #: Where the packet's sample row data begins, past the whole payload
    #: header. Subtracting it from :attr:`source` gives a segment's offset
    #: within that sample data. It counts every SRD header parsed, including
    #: any read out of a ``payloads`` buffer, so an :attr:`overflowed`
    #: packet's is short by six octets for every header the bound cut off.
    #:
    #: Under a header-data split the payload buffer begins at the cut rather
    #: than at the end of the payload header, and the two coincide only for a
    #: one-segment packet. A consumer indexing that buffer subtracts the cut
    #: from :attr:`source`; ``data_offset - cut`` is the header the cut left
    #: at the head of the payload row (§spec:split-segments).
    data_offset: NDArray[np.int64]

    # Segment-aligned.
    #: Row of ``packets`` this descriptor came from.
    packet: NDArray[np.int64]
    #: SRD Length: octets of sample row data, a multiple of the pgroup.
    #: Sixteen bits on the wire, reported at the width a consumer scales it
    #: at (§spec:conforming-fast-path).
    length: NDArray[np.int64]
    #: SRD Row Number, counting from the top of the image (of the field, when
    #: interlaced). Fifteen bits as the wire carried them, so a packet may
    #: name a row no flow has; this parse holds no format to refuse it with.
    #: :func:`pyst2110.geometry.fits_raster` is what bounds it against one.
    line: NDArray[np.int64]
    #: The F bit: which field of an interlaced image, or which segment of a
    #: PsF one. Always clear for progressive scan.
    field: NDArray[np.bool_]
    #: SRD Offset: the sample position the data starts at within its row.
    #: Counts pixels, not octets — :func:`pyst2110.geometry.byte_offset`
    #: turns it into one using the format's pgroup. Unbounded here for the
    #: same reason as :attr:`line`, and bounded by the same mask: the scaling
    #: carries a position past the row's width straight through.
    offset_samples: NDArray[np.int64]
    #: Where this descriptor's data begins within its packet row. Derived
    #: from the declared lengths ahead of it, which are unchecked, so it can
    #: point past the packet — and ``size - source`` is then negative, which
    #: an unsigned length in a C or CUDA gather turns enormous. Clamp it
    #: against the buffer actually read.
    source: NDArray[np.int64]

    #: Packet-aligned: which packets declared a sample row this parse could
    #: not read — the continuation bit still set at ``max_segments``, or an
    #: SRD header past the end of every buffer offered. Such a packet's
    #: descriptors are missing *and* the survivors are invalid — each
    #: unparsed SRD header leaves every :attr:`source` of that packet six
    #: octets early — so none of them is emitted and :attr:`segments` reads
    #: zero. The flag is what says the packet arrived and was dropped, which
    #: a receiver counting loss needs and a chunk-wide count cannot give: it
    #: names one packet in thousands as poison without saying which.
    #:
    #: Two senders reach it. A generic RFC 4175 one declares more SRDs than
    #: the bound allows; ST 2110-20 caps a packet at the three the default
    #: bound holds. And a header-data-split receiver that withholds
    #: ``payloads`` cannot follow a packet whose later headers the cut left
    #: in the payload buffer (§spec:split-segments).
    overflowed: NDArray[np.bool_]

    @property
    def overflowed_count(self) -> int:
        """Overflowed packets in this chunk, for a metric that counts them."""
        return int(np.count_nonzero(self.overflowed))


def parse_payload_headers(
    packets: NDArray[np.uint8],
    payload_offset: NDArray[np.integer[Any]],
    sizes: NDArray[np.integer[Any]] | None = None,
    max_segments: int = _ST2110_SEGMENTS,
    *,
    payloads: NDArray[np.uint8] | None = None,
) -> PayloadHeaders:
    """Parse the RFC 4175 payload header of every packet in a chunk.

    ``payload_offset`` is where each packet's payload begins, which is what
    :func:`pyst2110.rtp.parse_rtp` returns — the CSRC count and the extension
    bit both move it, so it differs per packet and cannot be hoisted.

    ``max_segments`` bounds the walk over a packet's segments and defaults to
    the three ST 2110-20 permits; a packet declaring more is flagged in
    :attr:`PayloadHeaders.overflowed` and contributes no descriptors. Below
    one it raises.

    ``payloads`` is what a header-data-split receiver holds beside ``packets``
    — one row a packet, holding the octets past the cut. The cut is a fixed
    offset and a payload header is not, so a two- or three-segment packet
    leaves its later SRD headers at the head of its payload row; given the
    buffer, the walk continues across the seam and reports every segment the
    packet declared. Withheld, such a packet is flagged instead, which is the
    answer a device payload ring needs: its payload is on an accelerator this
    parse cannot address (§spec:split-segments).

    Lengths are reported as the header declared them and are not checked
    against the packet, so bounding a gather is the consumer's, against the
    buffer it actually reads. Rows and offsets are likewise as declared:
    :func:`pyst2110.geometry.fits_raster` bounds those against a flow, which
    needs a format this parse does not take (§spec:payload-header).

    Which of the two parses runs is read from the chunk and never promised by
    a caller (§spec:conforming-fast-path); the arrays returned are the same
    either way.
    """
    _chunk.validate(packets, _layout.EXTENDED_SEQUENCE_SIZE + _layout.SRD_SIZE)
    if max_segments < 1:
        raise ValueError(
            f"a payload header carries at least one SRD, so max_segments must "
            f"be at least one, not {max_segments}"
        )
    count = packets.shape[0]
    if payloads is not None:
        # A row narrower than one SRD header cannot hold the header the cut
        # displaced, which is the whole of what this buffer is read for.
        _chunk.validate(payloads, _layout.SRD_SIZE)
        if payloads.shape[0] != count:
            raise ValueError(
                f"expected one payload row per packet: {count} packets, "
                f"{payloads.shape[0]} payload rows"
            )
    starts = _chunk.per_packet(
        payload_offset, "payload offset", count=count, plural="offsets"
    )
    bounds = _chunk.limits(packets, sizes)
    conforming = _conforming(packets, starts, bounds)
    if conforming is not None:
        return conforming
    return _general(packets, starts, bounds, max_segments, payloads)


def _conforming(
    packets: NDArray[np.uint8],
    starts: NDArray[np.int64],
    bounds: NDArray[np.int64],
    /,
) -> PayloadHeaders | None:
    """The chunk read as columns, or ``None`` where it is not that shape.

    Three vector tests decide it, beside the check that the buffer can be read
    as words at all, and every one of them reads the chunk rather than
    anything a caller said about it. The flags octet equalling
    :data:`pyst2110._layout.VERSION_2` is what says no packet carried a CSRC
    list or an extension; ``payload_offset`` agreeing with that is what says
    the offsets describe *this* chunk, since they are a caller's array and
    :func:`pyst2110.rtp.parse_rtp` is only their usual source; and no first
    SRD setting its continuation bit is what says every packet carries exactly
    one segment, which is what makes the descriptors packet-aligned and the
    walk a single column slice a field (§spec:conforming-fast-path).

    *Why read the octet here too*, when the caller has usually just read it:
    the path a chunk takes is a property of the chunk. Selecting on the
    offsets alone would let a caller computing its own choose the path, which
    is exactly what §spec:conforming-fast-path says nobody can do — and it
    costs one comparison over a column already in cache.

    ``max_segments`` does not appear: a chunk where nothing continues yields
    one descriptor a packet at any bound of one or more, and one is already
    the floor. Neither does a payload buffer: nothing continues, so nothing
    crosses a header-data split and there is nothing to read out of it
    (§spec:split-segments).
    """
    count = packets.shape[0]
    if count == 0 or bounds.min() < _CONFORMING_SIZE:
        return None
    words = _chunk.u16_view(packets)
    if (
        words is None
        or (packets[:, 0] != _layout.VERSION_2).any()
        or (starts != _layout.FIXED_HEADER_SIZE).any()
    ):
        return None
    # Each flag tops its word, so a comparison against the flag reads the bit
    # in one pass where a mask and a test against zero take two, and allocate
    # twice.
    offsets = words[:, _OFFSET_WORD]
    # A mask and `.any()` rather than `offsets.max()`: the column is a
    # big-endian sixteen-bit slice, and numpy's reduction over one is the
    # unbuffered loop, measured at twice the cost of the comparison it saves.
    if (offsets >= _layout.FLAG_MASK).any():
        return None

    lines = words[:, _ROW_WORD]
    return PayloadHeaders(
        extended_sequence=words[:, _EXTENDED_WORD].astype(np.int64),
        segments=np.ones(count, dtype=np.int64),
        data_offset=np.full(count, _CONFORMING_SIZE, dtype=np.int64),
        packet=np.arange(count, dtype=np.int64),
        length=words[:, _LENGTH_WORD].astype(np.int64),
        line=(lines & _VALUE_MASK).astype(np.int64),
        field=(lines >= _layout.FLAG_MASK),
        offset_samples=(offsets & _VALUE_MASK).astype(np.int64),
        # One segment a packet, so its data begins where the header ends.
        source=np.full(count, _CONFORMING_SIZE, dtype=np.int64),
        overflowed=np.zeros(count, dtype=np.bool_),
    )


def _general(
    packets: NDArray[np.uint8],
    starts: NDArray[np.int64],
    bounds: NDArray[np.int64],
    max_segments: int,
    payloads: NDArray[np.uint8] | None,
    /,
) -> PayloadHeaders:
    """The reference parse: a bounded walk over each packet's segments.

    Unchanged by the fast path beside it, and the authority where the two
    disagree (§spec:conforming-fast-path).

    Where a payload buffer is offered, the walk runs a second time over the
    packets that ran out of header buffer mid-segment — and over no others.
    Reading the first segment's continuation bit is what finds them, and that
    bit is in the header buffer, so the selection costs a mask rather than a
    stitched copy of the chunk (§spec:split-segments).
    """
    walk = _walk(packets, starts, bounds, max_segments)
    crossing = walk.crossing
    if payloads is not None and crossing is not None and crossing.any():
        _continue_into_payloads(
            walk,
            packets,
            payloads,
            starts,
            bounds,
            np.flatnonzero(crossing),
            max_segments,
        )
    return _assemble(walk, starts, max_segments)


@dataclass
class _Walk:
    """One pass of the segment walk, before its descriptors are raveled.

    Segment-aligned as ``(packets, max_segments)`` matrices rather than the
    ragged arrays a caller gets, because a second pass overwrites whole rows
    of them and a ravel cannot be rewritten in place.
    """

    sequence: NDArray[np.int64]
    present: NDArray[np.bool_]
    length: NDArray[np.int64]
    line: NDArray[np.int64]
    field: NDArray[np.bool_]
    offset: NDArray[np.int64]
    #: Declared a sample row this pass could not read: the bound cut it, or
    #: the buffer ended under it.
    overflowed: NDArray[np.bool_]
    #: The buffer ended under it — the subset a payload buffer can rescue.
    #: ``None`` where no packet reached a second segment at all, which is the
    #: ordinary chunk and the reason offering the buffer costs nothing.
    crossing: NDArray[np.bool_] | None


def _walk(
    packets: NDArray[np.uint8],
    starts: NDArray[np.int64],
    bounds: NDArray[np.int64],
    max_segments: int,
    /,
) -> _Walk:
    """Walk every packet's segments over one buffer, vectorized at each one."""
    count = packets.shape[0]
    # One row index for all nine reads below rather than one apiece: the
    # gather gets the same column vector every time, and allocating it per
    # field is pure overhead on a path that runs a thousand chunks a second.
    rows_index = np.arange(count, dtype=np.int64)
    sequence, readable = _chunk.read_u16(packets, starts, bounds, rows_index)
    shape = (count, max_segments)
    length = np.zeros(shape, dtype=np.int64)
    line = np.zeros(shape, dtype=np.int64)
    field = np.zeros(shape, dtype=np.bool_)
    offset = np.zeros(shape, dtype=np.int64)
    present = np.zeros(shape, dtype=np.bool_)
    crossing: NDArray[np.bool_] | None = None

    # A segment exists only if the one before it set the continuation bit and
    # its own six octets are inside the packet.
    active = readable
    base = starts + _layout.EXTENDED_SEQUENCE_SIZE
    for segment in range(max_segments):
        raw_length, _ = _chunk.read_u16(
            packets, base + _layout.SRD_LENGTH, bounds, rows_index
        )
        raw_row, _ = _chunk.read_u16(
            packets, base + _layout.SRD_ROW, bounds, rows_index
        )
        raw_offset, fits = _chunk.read_u16(
            packets, base + _layout.SRD_OFFSET, bounds, rows_index
        )
        # An SRD header is one six-octet unit, so its last field landing
        # inside the packet is the whole of it landing inside.
        read = active & fits
        if segment:
            # Past the first segment `active` is the previous segment's
            # continuation bit, so a read that does not fit is a sample row
            # the packet declared and this buffer does not hold — which is
            # what a header-data split leaves behind. Allocated here rather
            # than before the loop: a chunk whose packets each carry one
            # segment never reaches a second, and pays nothing for the
            # possibility.
            missed: NDArray[np.bool_] = active & ~fits
            crossing = missed if crossing is None else (crossing | missed)

        present[:, segment] = read
        length[:, segment] = np.where(read, raw_length, 0)
        line[:, segment] = np.where(read, raw_row & _layout.VALUE_MASK, 0)
        field[:, segment] = read & ((raw_row & _layout.FLAG_MASK) != 0)
        offset[:, segment] = np.where(read, raw_offset & _layout.VALUE_MASK, 0)

        active = read & ((raw_offset & _layout.FLAG_MASK) != 0)
        base = base + _layout.SRD_SIZE
        # Nothing continues, so every later segment would read `active` false
        # and write the zeros its column already holds. Leaving the loop is
        # therefore the same parse, not a shortened one — and it is the
        # ordinary case: a 2110-20 packet at the standard datagram size
        # carries one SRD, so the walk otherwise reads two segments that are
        # not there, for two thirds of its gathers. This is a bound on work,
        # never on how far the walk may reach: a packet that does continue
        # still gets `max_segments`, and one still continuing at the bound is
        # `overflowed` below exactly as before.
        if not active.any():
            break

    # A packet with SRD headers nobody parsed has every source it did parse
    # six octets early per missing header, whether the bound stopped the walk
    # or the buffer did. Its descriptors are wrong rather than merely
    # incomplete, so `_assemble` emits none of them and the flag says why.
    overflowed = active if crossing is None else (active | crossing)
    return _Walk(sequence, present, length, line, field, offset, overflowed, crossing)


def _continue_into_payloads(
    walk: _Walk,
    packets: NDArray[np.uint8],
    payloads: NDArray[np.uint8],
    starts: NDArray[np.int64],
    bounds: NDArray[np.int64],
    rows: NDArray[np.int64],
    max_segments: int,
    /,
) -> None:
    """Re-walk the packets that crossed the seam, over both their buffers.

    The two buffers are one packet, so the walk is the same walk — what
    changes is where it reads. Each selected packet gets a window onto its own
    octets from ``payload_offset`` on, drawn from the header row up to the cut
    and from the payload row after it, and the window is only as wide as
    ``max_segments`` SRD headers: the sample data behind them is never read.
    Twenty octets a packet at the default bound, over the crossing packets
    alone, is what keeps this proportional to how often a sender straddles
    (§spec:split-segments).

    ``walk`` is overwritten in place for those rows, so what follows cannot
    tell a stitched packet from a packet that arrived whole.
    """
    span = _layout.EXTENDED_SEQUENCE_SIZE + _layout.SRD_SIZE * max_segments
    window = _window(packets, payloads, starts, bounds, rows, span)
    # Both buffers end at one place in the packet, so what the pair holds is a
    # prefix of the window and one length bounds it.
    reach = np.clip(bounds[rows] + payloads.shape[1] - starts[rows], 0, span)
    again = _walk(window, np.zeros(rows.size, dtype=np.int64), reach, max_segments)

    walk.present[rows] = again.present
    walk.length[rows] = again.length
    walk.line[rows] = again.line
    walk.field[rows] = again.field
    walk.offset[rows] = again.offset
    # A packet still declaring a sample row neither buffer holds stays
    # flagged: the bound is the caller's, and so is the payload row's width.
    walk.overflowed[rows] = again.overflowed


def _window(
    packets: NDArray[np.uint8],
    payloads: NDArray[np.uint8],
    starts: NDArray[np.int64],
    bounds: NDArray[np.int64],
    rows: NDArray[np.int64],
    span: int,
    /,
) -> NDArray[np.uint8]:
    """One row a crossing packet: its own octets from ``payload_offset`` on.

    Both halves come from where the packet's octets actually are — the header
    buffer up to the cut, the payload buffer after it.

    A receiver cuts at a fixed offset and an RTP header with no CSRC list and
    no extension is a fixed size, so both bounds are usually one number for
    the whole chunk, and each half of the window is then a column slice of
    gathered rows — a copy a row at a time. Where either varies, so does every
    octet's home, and the window is gathered element by element instead: six
    times the cost, over the packets that cross and no others
    (§spec:conforming-fast-path).
    """
    begin = starts[rows]
    keep = np.clip(bounds[rows] - begin, 0, span)
    if begin[0] >= 0 and _one_number(begin) and _one_number(keep):
        head = int(keep[0])
        take = min(span - head, payloads.shape[1])
        window = np.zeros((rows.size, span), dtype=np.uint8)
        window[:, :head] = packets[rows, int(begin[0]) : int(begin[0]) + head]
        window[:, head : head + take] = payloads[rows, :take]
        return window

    at = rows[:, None]
    # Offsets within the packet, not within either buffer holding it.
    logical = begin[:, None] + np.arange(span, dtype=np.int64)
    seam = bounds[rows][:, None]
    tail = logical - seam
    width = payloads.shape[1]
    gathered: NDArray[np.uint8] = np.where(
        (logical >= 0) & (logical < seam),
        packets[at, np.clip(logical, 0, packets.shape[1] - 1)],
        np.where(
            (tail >= 0) & (tail < width),
            payloads[at, np.clip(tail, 0, width - 1)],
            0,
        ),
    )
    return gathered


def _one_number(values: NDArray[np.int64]) -> bool:
    """Whether every packet agrees on this bound, so a slice can read it."""
    return bool((values == values[0]).all())


def _assemble(
    walk: _Walk, starts: NDArray[np.int64], max_segments: int, /
) -> PayloadHeaders:
    """Ravel a walk's segment matrices into the ragged arrays a caller gets."""
    count = starts.shape[0]
    shape = (count, max_segments)
    parsed = walk.present.sum(axis=1, dtype=np.int64)
    data_offset = starts + _layout.EXTENDED_SEQUENCE_SIZE + _layout.SRD_SIZE * parsed
    # Every segment's data follows the whole header, one after another, so a
    # segment starts where the lengths before it end.
    preceding = np.cumsum(walk.length, axis=1, dtype=np.int64) - walk.length
    source = data_offset[:, None] + preceding

    present = walk.present
    present &= ~walk.overflowed[:, None]
    counted = present.sum(axis=1, dtype=np.int64)

    # A packet's segments are contiguous from the first — the continuation bit
    # cannot resume — so raveling packet-major keeps them in order.
    kept = present.ravel()
    rows = np.broadcast_to(np.arange(count, dtype=np.int64)[:, None], shape).ravel()
    return PayloadHeaders(
        extended_sequence=walk.sequence,
        segments=counted,
        data_offset=data_offset,
        packet=rows[kept],
        length=walk.length.ravel()[kept],
        line=walk.line.ravel()[kept],
        field=walk.field.ravel()[kept],
        offset_samples=walk.offset.ravel()[kept],
        source=source.ravel()[kept],
        overflowed=walk.overflowed,
    )
