"""Building a frame's headers once, and stamping them per frame.

The RFC 3550 fixed header and the RFC 4175 payload header of every packet in
a frame, laid out as one row per packet — the same shape the parses take
(§spec:interface-shape). The layout repeats frame after frame and only two
fields move, so the block is built once for a frame *shape* and stamped
(§spec:transmit-headers).

Stamping is the hot path: a 4K flow is hundreds of thousands of packets a
second, so both moving fields are written across the whole frame at once and
no Python loop runs over packets here.

No pixels. This says which octets of a frame buffer each packet carries and
never carries them, which is the same boundary the receive path draws from
the other side (§spec:scope-boundary).
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


def max_payload_size(video: SdpVideo) -> int:
    """The largest sample data a packet may carry under this flow's UDP limit.

    ``MAXUDP`` — and the Standard UDP Size Limit it defaults to — bounds the
    whole UDP datagram, not the sample data inside it: SMPTE ST 2110-10
    section 6.3 counts the UDP header, the RTP header and the payload header
    within it. Subtracting them is what turns a declared limit into the
    ``limit`` :func:`pyst2110.geometry.choose_payload_size` takes, which is a
    payload size.

    Passing ``video.max_udp`` to that function instead overstates the room by
    28 octets, and a raster whose line divides in that gap yields packets over
    the limit — which :class:`FrameHeaders` refuses.
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
    last time and hands back the same object, because allocating a frame of
    headers per frame is the one cost this design exists to avoid. A caller
    holding two frames at once copies the first.
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

        Raises ``ValueError`` where the format and payload size cannot be put
        on the wire: a sampling with no pgroup, a payload size that does not
        tile a line or is not a whole number of pgroups or overruns the flow's
        UDP limit, a raster with no rows or more than the row number field
        holds, or a payload type or SSRC outside its field.
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

        self.video = video
        self.payload_size = payload_size
        self.packets = packets_per_frame(video, payload_size)
        self._initial_sequence = initial_sequence
        self._index = np.arange(self.packets, dtype=np.int64)
        # The largest index whose sequence arithmetic stays inside int64.
        self._last_frame = (_INT64_MAX - initial_sequence) // self.packets - 1

        # Which field each packet belongs to, and which row within it. An
        # interlaced frame sends its fields in time order, first field first,
        # and numbers rows from the top of each (ST 2110-20 section 6.1.5).
        split = rows * per_line if video.interlaced else self.packets
        field = (self._index >= split).astype(np.int64)
        line = np.where(field == 1, self._index - split, self._index) // per_line
        self._field = field

        # Where each packet starts within its row: as octets, and as the sample
        # position the SRD Offset carries. RFC 4175 section 4.2 counts that
        # offset in pixels — "increments by one for each pixel" — where the
        # length beside it counts octets (§spec:geometry).
        offset_pixels = sample_offset(video, self._index % per_line * payload_size)

        #: Where in the frame buffer each packet's payload starts, in octets.
        #: The transmit-side counterpart of a receive descriptor: a consumer
        #: reads :attr:`payload_size` octets from here. An interlaced frame's
        #: second field sits below the like-numbered rows of the first, so
        #: field 1 row *r* is frame row 2r+1 (ST 2110-20 section 6.1.5).
        frame_row = line * _FIELDS_PER_FRAME + field if video.interlaced else line
        self.frame_offset: NDArray[np.int64] = raster_offset(
            video, frame_row, offset_pixels
        )

        # The marker ends a frame, or a field when interlaced (section 6.1.2).
        marker = np.zeros(self.packets, dtype=np.bool_)
        marker[split - 1] = True
        marker[-1] = True

        block = np.zeros((self.packets, PACKET_HEADER_SIZE), dtype=np.uint8)
        block[:, 0] = _layout.VERSION_2
        block[:, 1] = np.where(marker, _layout.MARKER_MASK | payload_type, payload_type)
        _put_u32(block, _layout.SSRC, np.full(self.packets, ssrc, dtype=np.int64))
        _put_u16(
            block,
            _SRD_BASE + _layout.SRD_LENGTH,
            np.full(self.packets, payload_size, np.int64),
        )
        _put_u16(
            block,
            _SRD_BASE + _layout.SRD_ROW,
            (line | field * _layout.FLAG_MASK).astype(np.int64),
        )
        _put_u16(block, _SRD_BASE + _layout.SRD_OFFSET, offset_pixels)
        self._block = block

    def stamp(self, frame_index: int) -> NDArray[np.uint8]:
        """Write this frame's sequence numbers and timestamp into the block.

        Both fields are derived from ``frame_index`` rather than carried
        forward, so a transmitter's hundred-thousandth frame is as exact as
        its first and a dropped frame does not shift what follows it.

        The RTP sequence number is the low sixteen bits of a thirty-two-bit
        counter and the extended sequence number the high sixteen, which is
        what keeps a flow fast enough to wrap the RTP field inside one frame
        readable (§spec:rtp).

        Returns the block — the same array every time, restamped in place.
        """
        if not 0 <= frame_index <= self._last_frame:
            raise ValueError(
                f"{frame_index} is not a frame index this block can stamp, "
                f"which runs to {self._last_frame}"
            )
        sequence = (
            self._initial_sequence + frame_index * self.packets + self._index
        ) % _layout.U32_MODULUS
        _put_u16(self._block, _layout.SEQUENCE, sequence & 0xFFFF)
        _put_u16(self._block, _EXTENDED_SEQUENCE, sequence >> 16)
        _put_u32(self._block, _layout.TIMESTAMP, self._timestamps(frame_index))
        return self._block

    def timestamp(self, frame_index: int, field_index: int = 0) -> int:
        """The RTP timestamp this frame's field carries.

        The 90 kHz clock of ST 2110-20 section 6.1.3 sampled at the frame's
        own rate, in exact integer arithmetic: a rate of 60000/1001 advances
        1501.5 ticks a frame, and computing from the index truncates each
        frame's own value rather than accumulating the half tick as an error.

        An interlaced frame carries a timestamp per field, section 6.1.3
        requiring one value across a field rather than across the frame.
        """
        rate = self.video.frame_rate
        ticks = (
            (_FIELDS_PER_FRAME * frame_index + field_index)
            * RTP_CLOCK_RATE
            * rate.denominator
        ) // (_FIELDS_PER_FRAME * rate.numerator)
        return ticks % _layout.U32_MODULUS

    def _timestamps(self, frame_index: int) -> NDArray[np.int64]:
        """One timestamp per packet: constant over a progressive frame, and
        over each field of an interlaced one."""
        by_field = np.array(
            [self.timestamp(frame_index, field) for field in range(_FIELDS_PER_FRAME)],
            dtype=np.int64,
        )
        return by_field[self._field]


def _put_u16(block: NDArray[np.uint8], offset: int, values: NDArray[np.int64]) -> None:
    """Write a big-endian 16-bit field into every packet's header.

    RFC 3550 section 5.1: "All integer fields are carried in network byte
    order". Written across the whole frame at once, one column pair at a time.
    """
    block[:, offset] = (values >> 8) & 0xFF
    block[:, offset + 1] = values & 0xFF


def _put_u32(block: NDArray[np.uint8], offset: int, values: NDArray[np.int64]) -> None:
    """Write a big-endian 32-bit field into every packet's header."""
    _put_u16(block, offset, (values >> 16) & 0xFFFF)
    _put_u16(block, offset + 2, values & 0xFFFF)
