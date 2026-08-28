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
    #: header. Subtracting it from :attr:`source` gives an offset into the
    #: payload alone, which is what a header-data-split receiver holds
    #: separately. It counts the SRD headers parsed, so an :attr:`overflowed`
    #: packet's is short by six octets for every header the bound cut off.
    data_offset: NDArray[np.int64]

    # Segment-aligned.
    #: Row of ``packets`` this descriptor came from.
    packet: NDArray[np.int64]
    #: SRD Length: octets of sample row data, a multiple of the pgroup.
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

    #: Packet-aligned: which packets still had the continuation bit set at
    #: ``max_segments``, having declared more sample rows than were parsed.
    #: Such a packet's descriptors are missing *and* the survivors are
    #: invalid — each unparsed SRD header leaves every :attr:`source` of that
    #: packet six octets early — so none of them is emitted and
    #: :attr:`segments` reads zero. The flag is what says the packet arrived
    #: and was dropped, which a receiver counting loss needs and a chunk-wide
    #: count cannot give: it names one packet in thousands as poison without
    #: saying which. Only a generic RFC 4175 sender reaches this; ST 2110-20
    #: caps a packet at the three SRDs the default bound allows.
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
) -> PayloadHeaders:
    """Parse the RFC 4175 payload header of every packet in a chunk.

    ``payload_offset`` is where each packet's payload begins, which is what
    :func:`pyst2110.rtp.parse_rtp` returns — the CSRC count and the extension
    bit both move it, so it differs per packet and cannot be hoisted.

    ``max_segments`` bounds the walk over a packet's segments and defaults to
    the three ST 2110-20 permits; a packet declaring more is flagged in
    :attr:`PayloadHeaders.overflowed` and contributes no descriptors. Below
    one it raises.

    Lengths are reported as the header declared them and are not checked
    against the packet, so bounding a gather is the consumer's, against the
    buffer it actually reads. Rows and offsets are likewise as declared:
    :func:`pyst2110.geometry.fits_raster` bounds those against a flow, which
    needs a format this parse does not take (§spec:payload-header).
    """
    _chunk.validate(packets, _layout.EXTENDED_SEQUENCE_SIZE + _layout.SRD_SIZE)
    if max_segments < 1:
        raise ValueError(
            f"a payload header carries at least one SRD, so max_segments must "
            f"be at least one, not {max_segments}"
        )
    count = packets.shape[0]
    starts = _chunk.per_packet(
        payload_offset, "payload offset", count=count, plural="offsets"
    )
    bounds = _chunk.limits(packets, sizes)

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

    parsed = present.sum(axis=1).astype(np.int64)
    data_offset = starts + _layout.EXTENDED_SEQUENCE_SIZE + _layout.SRD_SIZE * parsed
    # Every segment's data follows the whole header, one after another, so a
    # segment starts where the lengths before it end.
    preceding = np.cumsum(length, axis=1) - length
    source = data_offset[:, None] + preceding

    # A packet whose continuation was still set at the bound has SRD headers
    # nobody parsed, which puts every source it did parse six octets early per
    # missing header. Its descriptors are wrong rather than merely incomplete,
    # so the packet contributes none and the flag says why.
    overflowed = active
    present &= ~overflowed[:, None]
    counted = present.sum(axis=1).astype(np.int64)

    # A packet's segments are contiguous from the first — the continuation bit
    # cannot resume — so raveling packet-major keeps them in order.
    kept = present.ravel()
    rows = np.broadcast_to(np.arange(count, dtype=np.int64)[:, None], shape).ravel()
    return PayloadHeaders(
        extended_sequence=sequence,
        segments=counted,
        data_offset=data_offset,
        packet=rows[kept],
        length=length.ravel()[kept],
        line=line.ravel()[kept],
        field=field.ravel()[kept],
        offset_samples=offset.ravel()[kept],
        source=source.ravel()[kept],
        overflowed=overflowed,
    )
