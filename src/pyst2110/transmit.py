"""Building a frame's headers once, and stamping them per frame.

The RFC 3550 fixed header and the RFC 4175 payload header of every packet in
a frame, laid out as one row per packet — the shape the parses take
(§spec:interface-shape). The block is built for a frame *shape* and stamped
per frame, and no Python loop runs over packets in either
(§spec:transmit-headers).

No pixels: this says which octets of a frame buffer each packet carries and
never carries them (§spec:scope-boundary).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pyst2110 import _layout
from pyst2110.geometry import (
    packets_per_frame,
    packets_per_line,
    pgroup,
    raster_offset,
    rows_per_field,
    sample_offset,
)
from pyst2110.sdp import RTP_CLOCK_RATE, SdpVideo, validate_payload_type

__all__ = ["PACKET_HEADER_SIZE", "UDP_HEADER_SIZE", "FrameHeaders", "max_payload_size"]

#: Octets of header a packet carries: the twelve of RFC 3550's fixed header,
#: two of extended sequence number, and one six-octet SRD header.
PACKET_HEADER_SIZE = (
    _layout.FIXED_HEADER_SIZE + _layout.EXTENDED_SEQUENCE_SIZE + _layout.SRD_SIZE
)

#: The UDP header, which SMPTE ST 2110-10 section 6.3 counts inside its size
#: limit: "The UDP Size is reflected in the UDP header, and includes the length
#: of the UDP header (8 octets) and also the RTP headers and data."
UDP_HEADER_SIZE = 8

# Packet-relative positions, derived rather than restated: the extended
# sequence number sits directly after the fixed header and the one SRD header
# after that, and the SRD's own fields are offset from _SRD_BASE.
_EXTENDED_SEQUENCE = _layout.FIXED_HEADER_SIZE
_SRD_BASE = _EXTENDED_SEQUENCE + _layout.EXTENDED_SEQUENCE_SIZE

# ST 2110-20 section 7.2 bounds width and height at 32767, which is what the
# SRD Row Number and Offset fields hold under their own flag.
_MAX_RASTER = _layout.VALUE_MASK
_MAX_SRD_LENGTH = _layout.U16_MODULUS - 1
# The derivation in stamp() runs in numpy's int64, which overflows rather than
# wrapping. An index past what it holds is refused by name instead.
_INT64_MAX = (1 << 63) - 1
_FIELDS_PER_FRAME = 2
# ST 2110-20 section 6.1.4: the extended sequence number carries "the 16 high
# order bits of the extended 32-bit sequence number".
_HIGH_HALF = 16
_LOW_HALF_MASK = _layout.U16_MODULUS - 1
_OCTET_MASK = 0xFF
_OCTET_BITS = 8


def max_payload_size(video: SdpVideo) -> int:
    """The largest sample data a packet may carry under this flow's UDP limit.

    ``MAXUDP`` bounds the whole datagram, so this subtracts the UDP, RTP and
    payload headers SMPTE ST 2110-10 section 6.3 counts inside it. The result
    is the ``limit`` :func:`pyst2110.geometry.choose_payload_size` takes,
    which is a payload size (§spec:transmit-headers).
    """
    room = video.max_udp - UDP_HEADER_SIZE - PACKET_HEADER_SIZE
    if room <= 0:
        raise ValueError(
            f"a UDP size limit of {video.max_udp} octets leaves no room for "
            f"the {UDP_HEADER_SIZE + PACKET_HEADER_SIZE} octets of header a "
            f"packet carries"
        )
    return room


class FrameHeaders:
    """One frame's header block, built for a shape and stamped per frame.

    Everything the wire needs except the two fields that move: the payload
    type, SSRC and marker bit of §spec:rtp, and the row numbers, offsets and
    lengths that walk the raster (§spec:payload-header). :meth:`stamp` writes
    the sequence numbers and the media timestamp for a given frame.

    One SRD header a packet. ``payload_size`` has to tile a line exactly —
    :func:`pyst2110.geometry.choose_payload_size` is what picks one — so a
    packet never straddles a row and the Line Continuation bit stays clear.
    That is ST 2110-20's General Packing Mode, which is what
    :func:`pyst2110.sdp.format_sdp` declares.

    **The block is reused.** :meth:`stamp` writes into the array it returned
    last time and hands back the same object, so a caller holding two frames
    at once copies the first (§spec:transmit-headers).
    """

    def __init__(
        self,
        video: SdpVideo,
        payload_size: int,
        *,
        payload_type: int = _layout.DYNAMIC_PAYLOAD_TYPE,
        ssrc: int = 0,
        initial_sequence: int = 0,
    ) -> None:
        """Lay out the headers for one frame of this format.

        Raises ``ValueError`` where the format and payload size cannot go on
        the wire: a sampling with no pgroup, a payload size that is not whole
        pgroups or does not tile a line or overruns the flow's UDP limit or
        the SRD Length field, a raster with no rows or past what the row and
        offset fields hold, or a payload type, SSRC or initial sequence number
        outside its own field.
        """
        per_line, rows = _validate(
            video, payload_size, payload_type, ssrc, initial_sequence
        )
        self.video = video
        self.payload_size = payload_size
        self.packet_count = packets_per_frame(video, payload_size)
        self._initial_sequence = initial_sequence
        self._index = np.arange(self.packet_count, dtype=np.int64)
        # The largest index whose sequence arithmetic stays inside int64.
        self._last_frame = (_INT64_MAX - initial_sequence) // self.packet_count - 1

        # Which field each packet belongs to, and which row within it. An
        # interlaced frame sends its fields in time order, first field first,
        # and numbers rows from the top of each (ST 2110-20 section 6.1.5).
        split = rows * per_line if video.interlaced else self.packet_count
        field = (self._index >= split).astype(np.int64)
        line = np.where(field == 1, self._index - split, self._index) // per_line
        self._field = field

        # Where each packet starts within its row, as the sample position the
        # SRD Offset carries. RFC 4175 section 4.2 counts that offset in
        # pixels — "increments by one for each pixel" — where the length
        # beside it counts octets (§spec:geometry).
        offset_pixels = sample_offset(video, self._index % per_line * payload_size)

        #: Where in the frame buffer each packet's payload starts, in octets.
        #: The transmit-side counterpart of a receive descriptor: a consumer
        #: reads :attr:`payload_size` octets from here. An interlaced frame's
        #: second field sits below the like-numbered rows of the first, so
        #: field 1 row *r* is frame row 2r+1 (ST 2110-20 section 6.1.5).
        frame_row = line * _FIELDS_PER_FRAME + field if video.interlaced else line
        self.frame_offset_octets: NDArray[np.int64] = raster_offset(
            video, frame_row, offset_pixels
        )

        # Everything a stamp derives is allocated here beside the block, so
        # the hot path writes through these and allocates nothing.
        self._sequence = np.empty(self.packet_count, dtype=np.int64)
        self._stamps = np.empty(self.packet_count, dtype=np.int64)
        self._half = np.empty(self.packet_count, dtype=np.int64)
        self._octets = np.empty(self.packet_count, dtype=np.int64)

        # The marker ends a frame, or a field when interlaced (section 6.1.2).
        marker = np.zeros(self.packet_count, dtype=np.bool_)
        marker[split - 1] = True
        marker[-1] = True

        block = np.zeros((self.packet_count, PACKET_HEADER_SIZE), dtype=np.uint8)
        block[:, 0] = _layout.VERSION_2
        block[:, 1] = np.where(marker, _layout.MARKER_MASK | payload_type, payload_type)
        _put_u32(
            block,
            _layout.SSRC,
            np.full(self.packet_count, ssrc, dtype=np.int64),
            self._half,
            self._octets,
        )
        _put_u16(
            block,
            _SRD_BASE + _layout.SRD_LENGTH,
            np.full(self.packet_count, payload_size, np.int64),
            self._octets,
        )
        _put_u16(
            block,
            _SRD_BASE + _layout.SRD_ROW,
            (line | field * _layout.FLAG_MASK).astype(np.int64),
            self._octets,
        )
        _put_u16(block, _SRD_BASE + _layout.SRD_OFFSET, offset_pixels, self._octets)
        self._block = block

    def stamp(self, frame_index: int) -> NDArray[np.uint8]:
        """Write this frame's sequence numbers and timestamp into the block.

        Both derive from ``frame_index`` rather than from a running counter
        (§spec:transmit-headers). The RTP sequence number is the low sixteen
        bits of a thirty-two-bit counter and the extended sequence number the
        high sixteen (§spec:rtp).

        Returns the block, restamped in place. Raises ``ValueError`` on an
        index outside what the derivation holds.
        """
        if not 0 <= frame_index <= self._last_frame:
            raise ValueError(
                f"{frame_index} is not a frame index this block can stamp, "
                f"which runs to {self._last_frame}"
            )
        first = self._initial_sequence + frame_index * self.packet_count
        np.add(self._index, first, out=self._sequence)
        np.bitwise_and(self._sequence, _layout.U32_MODULUS - 1, out=self._sequence)
        np.bitwise_and(self._sequence, _LOW_HALF_MASK, out=self._half)
        _put_u16(self._block, _layout.SEQUENCE, self._half, self._octets)
        np.right_shift(self._sequence, _HIGH_HALF, out=self._half)
        _put_u16(self._block, _EXTENDED_SEQUENCE, self._half, self._octets)
        self._write_timestamps(frame_index)
        return self._block

    def timestamp(self, frame_index: int, field_index: int = 0) -> int:
        """The RTP timestamp this frame's field carries.

        The 90 kHz clock of ST 2110-20 section 6.1.3 sampled at the frame's
        own rate, in exact integer arithmetic (§spec:transmit-headers). An
        interlaced frame carries one per field, section 6.1.3 requiring a
        value across a field rather than across a frame.
        """
        rate = self.video.frame_rate
        ticks = (
            (_FIELDS_PER_FRAME * frame_index + field_index)
            * RTP_CLOCK_RATE
            * rate.denominator
        ) // (_FIELDS_PER_FRAME * rate.numerator)
        return ticks % _layout.U32_MODULUS

    def _write_timestamps(self, frame_index: int) -> None:
        """One timestamp per packet: constant over a progressive frame, and
        over each field of an interlaced one."""
        by_field = np.array(
            [self.timestamp(frame_index, field) for field in range(_FIELDS_PER_FRAME)],
            dtype=np.int64,
        )
        np.take(by_field, self._field, out=self._stamps)
        _put_u32(self._block, _layout.TIMESTAMP, self._stamps, self._half, self._octets)


def _validate(
    video: SdpVideo,
    payload_size: int,
    payload_type: int,
    ssrc: int,
    initial_sequence: int,
) -> tuple[int, int]:
    """Refuse a frame that cannot go on the wire, and return its geometry.

    :meth:`FrameHeaders.__init__` names what each refusal is. Returns the
    packets a line takes and the rows in the temporally first field, which
    the checks compute on the way and the layout then needs.
    """
    group_bytes, _ = pgroup(video)
    if payload_size % group_bytes:
        raise ValueError(
            f"a payload of {payload_size} octets is not a whole number of "
            f"{group_bytes}-octet pgroups, which RFC 4175 requires an SRD "
            f"length to be; see choose_payload_size()"
        )
    if payload_size > _MAX_SRD_LENGTH:
        raise ValueError(
            f"a payload of {payload_size} octets is past the "
            f"{_MAX_SRD_LENGTH} the SRD Length field holds"
        )
    if video.width > _MAX_RASTER:
        raise ValueError(
            f"a width of {video.width} needs a sample offset past "
            f"{_MAX_RASTER}, which is all the SRD Offset field holds "
            f"below the Line Continuation bit"
        )
    per_line = packets_per_line(video, payload_size)
    # ST 2110-10 section 6.3: "Senders shall not generate IP Datagrams
    # containing UDP packet sizes larger than this limit."
    if payload_size > max_payload_size(video):
        raise ValueError(
            f"a payload of {payload_size} octets makes a UDP size of "
            f"{payload_size + PACKET_HEADER_SIZE + UDP_HEADER_SIZE}, past "
            f"the {video.max_udp} this flow declares; size it with "
            f"choose_payload_size(video, max_payload_size(video))"
        )
    validate_payload_type(payload_type)
    if not 0 <= ssrc < _layout.U32_MODULUS:
        raise ValueError(f"{ssrc} is not a 32-bit SSRC")
    if not 0 <= initial_sequence < _layout.U32_MODULUS:
        raise ValueError(f"{initial_sequence} is not a 32-bit sequence number")
    if video.height <= 0:
        raise ValueError(f"a height of {video.height} is not an image")
    rows = rows_per_field(video)
    if rows > _MAX_RASTER:
        raise ValueError(
            f"a height of {video.height} needs a row number past "
            f"{_MAX_RASTER}, which is all the SRD Row Number field holds"
        )
    return per_line, rows


def _put_u16(
    block: NDArray[np.uint8],
    offset: int,
    values: NDArray[np.int64],
    octets: NDArray[np.int64],
) -> None:
    """Write a big-endian 16-bit field into every packet's header.

    RFC 3550 section 5.1: "All integer fields are carried in network byte
    order". Written across the whole frame at once, one column pair at a
    time, through the caller's ``octets`` buffer so nothing is allocated.
    """
    np.right_shift(values, _OCTET_BITS, out=octets)
    np.bitwise_and(octets, _OCTET_MASK, out=octets)
    block[:, offset] = octets
    np.bitwise_and(values, _OCTET_MASK, out=octets)
    block[:, offset + 1] = octets


def _put_u32(
    block: NDArray[np.uint8],
    offset: int,
    values: NDArray[np.int64],
    half: NDArray[np.int64],
    octets: NDArray[np.int64],
) -> None:
    """Write a big-endian 32-bit field into every packet's header."""
    np.right_shift(values, _HIGH_HALF, out=half)
    _put_u16(block, offset, half, octets)
    np.bitwise_and(values, _LOW_HALF_MASK, out=half)
    _put_u16(block, offset + 2, half, octets)
