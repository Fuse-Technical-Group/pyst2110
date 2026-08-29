"""The RFC 3550 fixed header, parsed across a chunk of packets (§spec:rtp).

Twelve octets of version, flags, marker, payload type, sequence number,
timestamp and SSRC, followed by whatever the CSRC count and the extension bit
say follows. Every field comes back as one array over the whole chunk, because
a per-packet Python object at 250,000 packets a second is not affordable
(§spec:interface-shape).

SMPTE ST 2110-20 section 6.1.2 constrains the header rather than changing it:
the marker bit is the last packet of a frame (of a field, when interlaced),
and the sequence number is the low sixteen bits of a thirty-two-bit counter
whose high half rides in the RFC 4175 payload header.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyst2110 import _chunk, _layout
from pyst2110._layout import FIXED_HEADER_SIZE

__all__ = ["FIXED_HEADER_SIZE", "RtpHeaders", "parse_rtp"]

_CSRC_SIZE = 4
# RFC 3550 §5.3.1: two profile-defined octets and a length, then that many
# 32-bit words. The length counts the words alone, so the four octets of the
# extension header itself are always there when X is set — including for the
# zero-length extension the RFC permits.
_EXTENSION_HEADER_SIZE = 4
_EXTENSION_WORD_SIZE = 4

# Where the fixed header's fields sit in a big-endian 16-bit view of the
# chunk. The two wide ones take a word each side, high half first.
_SEQUENCE_WORD = _layout.SEQUENCE // 2
_TIMESTAMP_WORD = _layout.TIMESTAMP // 2
_SSRC_WORD = _layout.SSRC // 2
_HALF_WORD_SHIFT = 16
# In the field's own width, so masking one does not widen it back again.
_PAYLOAD_TYPE_MASK = np.uint8(_layout.PAYLOAD_TYPE_MASK)


@dataclass(frozen=True)
class RtpHeaders:
    """One chunk's RTP fixed headers, one array per field.

    Every array is packet-aligned: row ``i`` of the input describes element
    ``i`` of each field. Fields are reported as the wire carried them and are
    not corrected — a version other than two, or a payload type the SDP did
    not declare, is the receiver's to act on with context this parse lacks.

    A packet whose reported size is under :data:`FIXED_HEADER_SIZE` carries no
    fixed header, so all of these read zero rather than the bytes a reused
    buffer still holds there (§spec:interface-shape). A version of zero is
    what says so, no sender emitting one.
    """

    version: NDArray[np.uint8]
    padding: NDArray[np.bool_]
    extension: NDArray[np.bool_]
    csrc_count: NDArray[np.uint8]
    marker: NDArray[np.bool_]
    payload_type: NDArray[np.uint8]
    sequence: NDArray[np.uint16]
    timestamp: NDArray[np.uint32]
    ssrc: NDArray[np.uint32]
    #: Where each packet's payload begins, past the CSRC list and extension.
    #: Never past the packet's own length, so a truncated packet reports an
    #: offset equal to its size and yields no payload rather than an index
    #: error downstream.
    payload_offset: NDArray[np.int64]


def parse_rtp(
    packets: NDArray[np.uint8], sizes: NDArray[np.integer[Any]] | None = None
) -> RtpHeaders:
    """Parse the fixed header of every packet in a chunk.

    ``packets`` is a ``(packets, stride)`` uint8 view — a chunk's header
    sub-block, or its whole packets where the stream has no header-data split.
    ``sizes`` gives each packet's true length; without it the stride is
    assumed, which is right for a capture and wrong for a ring whose entries
    are wider than its packets (§spec:interface-shape).

    Which of the two parses runs is read from the chunk and never promised by
    a caller (§spec:conforming-fast-path); the arrays returned are the same
    either way.
    """
    _chunk.validate(packets, FIXED_HEADER_SIZE)
    bounds = _chunk.limits(packets, sizes)
    conforming = _conforming(packets, bounds)
    return conforming if conforming is not None else _general(packets, bounds)


def _conforming(
    packets: NDArray[np.uint8], bounds: NDArray[np.int64]
) -> RtpHeaders | None:
    """The chunk read as columns, or ``None`` where it is not that shape.

    One comparison over the flags octet decides it. A conforming ST 2110-20
    sender emits exactly :data:`pyst2110._layout.VERSION_2` there — version
    two, no padding, no extension, no CSRC list — so the single pass settles
    four fields as well as the payload offset, and each of them is a constant
    the length of the chunk rather than an array read off the wire. Anything
    else, padding included, is the general path's (§spec:conforming-fast-path).

    The payload offset follows from the same octet: with no CSRC list and no
    extension every payload begins at the fixed header's end, which is the
    per-packet gather this path exists to avoid.
    """
    count = packets.shape[0]
    if count == 0 or bounds.min() < FIXED_HEADER_SIZE:
        return None
    words = _chunk.u16_view(packets)
    if words is None or (packets[:, 0] != _layout.VERSION_2).any():
        return None

    # The marker tops its octet and the field flags top their words, so a
    # comparison against the flag reads the bit in one pass where a mask and
    # a test against zero take two, and allocate twice.
    types = packets[:, 1]
    return RtpHeaders(
        version=np.full(count, _layout.RTP_VERSION, dtype=np.uint8),
        padding=np.zeros(count, dtype=np.bool_),
        extension=np.zeros(count, dtype=np.bool_),
        csrc_count=np.zeros(count, dtype=np.uint8),
        marker=(types >= _layout.MARKER_MASK),
        payload_type=(types & _PAYLOAD_TYPE_MASK),
        sequence=words[:, _SEQUENCE_WORD].astype(np.uint16),
        timestamp=_joined_u32(words, _TIMESTAMP_WORD),
        ssrc=_joined_u32(words, _SSRC_WORD),
        payload_offset=np.full(count, FIXED_HEADER_SIZE, dtype=np.int64),
    )


def _joined_u32(words: NDArray[np.uint16], high: int) -> NDArray[np.uint32]:
    """A 32-bit field as its two 16-bit columns, joined in place.

    Copying the four octets out to read them as one word touches twice the
    memory a column slice does and costs about twice as long, the columns
    being what a strided copy has to walk anyway. Widening the high half and
    folding the low one into it allocates once for the result and no more.
    """
    value: NDArray[np.uint32] = words[:, high].astype(np.uint32)
    value <<= _HALF_WORD_SHIFT
    value |= words[:, high + 1]
    return value


def _general(packets: NDArray[np.uint8], bounds: NDArray[np.int64]) -> RtpHeaders:
    """The reference parse: every field read wherever this packet put it.

    Unchanged by the fast path beside it, and the authority where the two
    disagree (§spec:conforming-fast-path).
    """
    # The fixed header is one twelve-octet unit, so a packet reporting less
    # carries none of it and every field below reads zero instead of whatever
    # the buffer held there before.
    whole = bounds >= FIXED_HEADER_SIZE
    flags = np.where(whole, packets[:, 0], 0)
    types = np.where(whole, packets[:, 1], 0)
    csrc_count = (flags & _layout.CSRC_COUNT_MASK).astype(np.uint8)
    extension = (flags & _layout.EXTENSION_MASK) != 0

    return RtpHeaders(
        version=(flags >> _layout.VERSION_SHIFT).astype(np.uint8),
        padding=((flags & _layout.PADDING_MASK) != 0),
        extension=extension,
        csrc_count=csrc_count,
        marker=((types & _layout.MARKER_MASK) != 0),
        payload_type=(types & _layout.PAYLOAD_TYPE_MASK).astype(np.uint8),
        sequence=_fixed_u16(packets, _layout.SEQUENCE, whole),
        timestamp=_fixed_u32(packets, _layout.TIMESTAMP, whole),
        ssrc=_fixed_u32(packets, _layout.SSRC, whole),
        payload_offset=_payload_offset(packets, bounds, csrc_count, extension),
    )


def _payload_offset(
    packets: NDArray[np.uint8],
    bounds: NDArray[np.int64],
    csrc_count: NDArray[np.uint8],
    extension: NDArray[np.bool_],
) -> NDArray[np.int64]:
    """Where the payload starts, once the CSRC list and extension are counted.

    The extension's length field sits after the CSRC list, not after the fixed
    header, so it can only be read once the count is known. ST 2110-20 section
    6.1.2 provides for an extension — where X is set, an RFC 8285 header
    extension follows the SSRC — so a receiver that assumed the bit clear
    would read its first octets as an RFC 4175 payload header and report a
    line that does not exist (§spec:rtp).
    """
    csrc_end = FIXED_HEADER_SIZE + _CSRC_SIZE * csrc_count.astype(np.int64)
    # A length that does not fit inside the packet reads as zero, which is the
    # zero-word extension: four octets of header and no words.
    words, _ = _chunk.read_u16(packets, csrc_end + 2, bounds)
    extension_size = np.where(
        extension, _EXTENSION_HEADER_SIZE + _EXTENSION_WORD_SIZE * words, 0
    )
    return np.minimum(csrc_end + extension_size, bounds).astype(np.int64)


def _fixed_u16(
    packets: NDArray[np.uint8], offset: int, whole: NDArray[np.bool_]
) -> NDArray[np.uint16]:
    """The big-endian 16-bit field at a fixed offset, as its own dtype.

    A column slice is not contiguous along the last axis, so the copy is what
    lets the octets be read as one wide big-endian value instead of shifted
    together a byte at a time. ``whole`` says which packets are long enough to
    have the field at all; the rest read zero.
    """
    columns = np.ascontiguousarray(packets[:, offset : offset + 2])
    return np.where(whole, columns.view(">u2")[:, 0], 0).astype(np.uint16)


def _fixed_u32(
    packets: NDArray[np.uint8], offset: int, whole: NDArray[np.bool_]
) -> NDArray[np.uint32]:
    """The big-endian 32-bit field at a fixed offset, as its own dtype."""
    columns = np.ascontiguousarray(packets[:, offset : offset + 4])
    return np.where(whole, columns.view(">u4")[:, 0], 0).astype(np.uint32)
