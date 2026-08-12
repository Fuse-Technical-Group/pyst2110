"""Building and stamping a frame's headers (SPEC §spec:transmit-headers).

The expected octets are written by hand from the field diagrams in RFC 3550
section 5.1 and SMPTE ST 2110-20 section 6.1.4, and compared byte for byte.
They are the evidence; the round trip through this library's own parse is an
additional property and lives in ``test_transmit_slice.py`` (§spec:testing).

The raster is deliberately tiny — four pixels by two rows — so that every
octet of every packet can be worked out on paper.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from pyst2110.geometry import choose_payload_size
from pyst2110.sdp import SdpVideo
from pyst2110.transmit import (
    PACKET_HEADER_SIZE,
    UDP_HEADER_SIZE,
    FrameHeaders,
    max_payload_size,
)

_SSRC = 0x1234ABCD
_PAYLOAD_TYPE = 96


def u16(block: np.ndarray, offset: int) -> np.ndarray:
    """The big-endian 16-bit field at this octet, over every packet.

    RFC 3550 section 5.1 puts every integer field in network byte order, so
    one reader serves them all. The offsets stay literal at each call site,
    which is where the field diagrams are being checked (§spec:testing).
    """
    return block[:, offset].astype(np.int64) << 8 | block[:, offset + 1]


def u32(block: np.ndarray, offset: int) -> np.ndarray:
    """The big-endian 32-bit field at this octet, over every packet."""
    return u16(block, offset) << 16 | u16(block, offset + 2)


def sequence32(block: np.ndarray) -> np.ndarray:
    """The whole counter: the extended half at octet 12 above the RTP field."""
    return u16(block, 12) << 16 | u16(block, 2)


def video(**overrides: Any) -> SdpVideo:
    """Four pixels by two rows of YCbCr-4:2:2 10-bit: a 10-octet line."""
    fields: dict[str, Any] = {
        "width": 4,
        "height": 2,
        "frame_rate": Fraction(25),
        "depth": 10,
        "sampling": "YCbCr-4:2:2",
    }
    return SdpVideo(**(fields | overrides))


def headers() -> FrameHeaders:
    """The reference frame: two packets a line, two lines, wrapping sequence."""
    return FrameHeaders(
        video(),
        payload_size=5,
        payload_type=_PAYLOAD_TYPE,
        ssrc=_SSRC,
        initial_sequence=65534,
    )


# One pgroup per packet: two packets a line, two lines, four packets a frame.
# Sequence numbers start at 65534 so the RTP field wraps inside the frame and
# the extended sequence number has to carry the difference.
_FRAME = [
    # V=2 P=0 X=0 CC=0 | M=0 PT=96 | seq 65534 | ts 0 | SSRC | ext 0
    # | SRD length 5 | F=0 row 0 | C=0 offset 0
    "80 60 FF FE 00000000 1234ABCD 0000 0005 0000 0000",
    "80 60 FF FF 00000000 1234ABCD 0000 0005 0000 0002",
    "80 60 00 00 00000000 1234ABCD 0001 0005 0001 0000",
    # The marker bit: ST 2110-20 section 6.1.2 sets it on the last packet
    # carrying video essence data for a frame, so 0x80 | 96 = 0xE0.
    "80 E0 00 01 00000000 1234ABCD 0001 0005 0001 0002",
]


def octets(text: str) -> list[int]:
    return list(bytes.fromhex(text.replace(" ", "")))


def test_a_frames_headers_are_the_octets_the_diagrams_specify():
    block = headers().stamp(0)
    # Arrays in, arrays out — the shape the rest of the library speaks.
    assert (block.dtype, block.shape) == (np.uint8, (4, PACKET_HEADER_SIZE))
    assert [row.tolist() for row in block] == [octets(one) for one in _FRAME]


def test_the_header_is_twelve_of_rtp_two_of_sequence_and_one_srd():
    assert PACKET_HEADER_SIZE == 12 + 2 + 6


def test_the_marker_falls_on_the_last_packet_and_nowhere_else():
    block = headers().stamp(0)
    marked = (block[:, 1] & 0x80) != 0
    assert marked.tolist() == [False, False, False, True]


def test_the_row_numbers_and_offsets_walk_the_raster_in_order():
    """ST 2110-20 section 6.1.4: rows "start at 0 at the top of the transmitted
    image" and only increase; section 6.1.5: offsets only increase within a row.
    """
    block = headers().stamp(0)
    assert u16(block, 16).tolist() == [0, 0, 1, 1]
    assert u16(block, 18).tolist() == [0, 2, 0, 2]


def test_the_srd_length_is_the_payload_size_in_octets():
    """RFC 4175 section 4.2: "Number of octets of data included from this scan
    line", a multiple of the pgroup — not a count of pgroups or of pixels."""
    block = headers().stamp(0)
    assert u16(block, 14).tolist() == [5, 5, 5, 5]


def test_no_packet_sets_the_continuation_bit():
    """A payload size that tiles a line exactly never straddles a row, so one
    SRD header describes each packet and the C bit stays clear."""
    block = headers().stamp(0)
    assert ((block[:, 18] & 0x80) != 0).tolist() == [False] * 4


def test_the_extended_sequence_carries_the_high_half_of_the_counter():
    """ST 2110-20 section 6.1.4: "the 16 high order bits of the extended 32-bit
    sequence number", which is what makes the wrap inside this frame readable.
    """
    block = headers().stamp(0)
    assert u16(block, 2).tolist() == [65534, 65535, 0, 1]
    assert u16(block, 12).tolist() == [0, 0, 1, 1]


def test_sequence_numbers_run_on_across_frames_without_a_gap():
    block = headers()
    first = sequence32(block.stamp(0)).tolist()
    second = sequence32(block.stamp(1)).tolist()
    assert first + second == list(range(65534, 65534 + 8))


def test_the_thirty_two_bit_counter_wraps_rather_than_overflowing():
    block = FrameHeaders(
        video(), payload_size=5, ssrc=_SSRC, initial_sequence=(1 << 32) - 2
    ).stamp(0)
    assert u16(block, 2).tolist() == [65534, 65535, 0, 1]
    assert u16(block, 12).tolist() == [65535, 65535, 0, 0]


# --- The media timestamp --------------------------------------------------


def test_every_packet_of_a_progressive_frame_shares_one_timestamp():
    """ST 2110-20 section 6.1.3: "All RTP packets which are part of the same
    progressive frame shall contain the same RTP Timestamp value"."""
    assert len(set(u32(headers().stamp(7), 4).tolist())) == 1


@pytest.mark.parametrize(
    ("rate", "index", "expected"),
    [
        # 90 kHz over the frame rate: exact at the integer rates.
        (Fraction(25), 1, 3600),
        (Fraction(24), 1, 3750),
        (Fraction(50), 3, 5400),
        # 1501.5 ticks a frame, so the odd frames land on the half tick and
        # truncate — and the even ones stay exact, which is the property a
        # frame index buys over accumulating a per-frame increment.
        (Fraction(60000, 1001), 1, 1501),
        (Fraction(60000, 1001), 2, 3003),
        (Fraction(60000, 1001), 3, 4504),
        (Fraction(60000, 1001), 4, 6006),
    ],
)
def test_the_timestamp_is_the_ninety_kilohertz_clock_at_the_frame_rate(
    rate: Fraction, index: int, expected: int
):
    block = FrameHeaders(video(frame_rate=rate), payload_size=5, ssrc=_SSRC).stamp(
        index
    )
    assert u32(block, 4)[0] == expected


def test_a_timestamp_far_into_a_run_is_exact_rather_than_drifted():
    """Derived from the frame index, so an hour in is as exact as frame one."""
    rate = Fraction(60000, 1001)
    block = FrameHeaders(video(frame_rate=rate), payload_size=5, ssrc=_SSRC)
    index = 60 * 60 * 60  # an hour of frames
    expected = index * 90_000 * rate.denominator // rate.numerator
    assert u32(block.stamp(index), 4)[0] == expected % (1 << 32)


def test_the_timestamp_wraps_within_its_thirty_two_bits():
    rate = Fraction(25)
    block = FrameHeaders(video(frame_rate=rate), payload_size=5, ssrc=_SSRC)
    index = (1 << 32) // 3600 + 1  # past the 32-bit clock's roll-over
    assert u32(block.stamp(index), 4)[0] == (index * 3600) % (1 << 32)


# --- Interlace ------------------------------------------------------------


# Four pixels by four rows, interlaced: two rows a field, two packets a row.
# ST 2110-20 section 6.1.2 marks the last packet of each field rather than of
# the frame; section 6.1.5 restarts row numbers within a field and sets the F
# bit on the second; section 6.1.3 gives each field its own timestamp, and 25
# frames a second is 50 fields, so the second field is 1800 ticks (0x0708) on.
_INTERLACED_FRAME = [
    "80 60 00 00 00000000 1234ABCD 0000 0005 0000 0000",
    "80 60 00 01 00000000 1234ABCD 0000 0005 0000 0002",
    "80 60 00 02 00000000 1234ABCD 0000 0005 0001 0000",
    "80 E0 00 03 00000000 1234ABCD 0000 0005 0001 0002",
    "80 60 00 04 00000708 1234ABCD 0000 0005 8000 0000",
    "80 60 00 05 00000708 1234ABCD 0000 0005 8000 0002",
    "80 60 00 06 00000708 1234ABCD 0000 0005 8001 0000",
    "80 E0 00 07 00000708 1234ABCD 0000 0005 8001 0002",
]


def test_an_interlaced_frames_headers_are_the_octets_the_diagrams_specify():
    """The interlace vector, written from the field diagrams. The round trip
    in ``test_transmit_slice.py`` is the additional property (§spec:testing).
    """
    block = FrameHeaders(
        video(height=4, interlaced=True),
        payload_size=5,
        payload_type=_PAYLOAD_TYPE,
        ssrc=_SSRC,
    ).stamp(0)
    assert [row.tolist() for row in block] == [octets(one) for one in _INTERLACED_FRAME]


# --- The target format ----------------------------------------------------

# 2160p24 YCbCr-4:2:2 10-bit: a 9600-octet line, 1200 octets a packet, eight
# packets a line, 17,280 a frame. Four packets of frame 4 worked out from the
# diagrams — the first, the last of row 0, the first of row 1, and the last of
# the frame. 90 kHz over 24 frames a second is 3750 ticks a frame, so frame 4
# stamps 15000 (0x3A98). The counter starts at 4 * 17280 = 69120, past the RTP
# field's wrap, so the extended sequence number reads 1 across the frame. The
# offsets are sample positions: 7 * 1200 octets is 1680 pgroups is 3360 pixels
# (0x0D20), and the last row of 2160 is 2159 (0x086F).
_2160P24_FRAME_4 = {
    0: "80 60 0E 00 00003A98 1234ABCD 0001 04B0 0000 0000",
    7: "80 60 0E 07 00003A98 1234ABCD 0001 04B0 0000 0D20",
    8: "80 60 0E 08 00003A98 1234ABCD 0001 04B0 0001 0000",
    17_279: "80 E0 51 7F 00003A98 1234ABCD 0001 04B0 086F 0D20",
}


def test_the_target_formats_packets_are_the_octets_the_diagrams_specify():
    """2160p24 is what the consuming runtime asks for first, and the round
    trip is otherwise the only thing that checks it (§spec:testing)."""
    frame = FrameHeaders(
        video(width=3840, height=2160, frame_rate=Fraction(24)),
        payload_size=1200,
        payload_type=_PAYLOAD_TYPE,
        ssrc=_SSRC,
    )
    assert frame.packet_count == 17_280
    block = frame.stamp(4)
    for index, expected in _2160P24_FRAME_4.items():
        assert block[index].tolist() == octets(expected), f"packet {index}"


def test_an_odd_height_gives_the_first_field_the_extra_line():
    """Section 6.1.5: "if the height is odd, the temporally first field shall
    contain one more line than the temporally second field"."""
    block = FrameHeaders(
        video(height=5, interlaced=True), payload_size=5, ssrc=_SSRC
    ).stamp(0)
    fields = ((u16(block, 16) & 0x8000) != 0).tolist()
    assert fields.count(False) == 6, "three lines of two packets in the first field"
    assert fields.count(True) == 4, "two lines in the second"


# --- Where the payload comes from -----------------------------------------


def test_the_frame_offsets_walk_the_raster_a_packet_at_a_time():
    """The transmit-side descriptor: which octets of the frame buffer this
    packet carries. Never the pixels themselves (§spec:scope-boundary)."""
    block = headers()
    assert block.frame_offset_octets.tolist() == [0, 5, 10, 15]
    assert block.payload_size == 5


def test_the_frame_offsets_of_an_interlaced_frame_interleave_the_fields():
    """Section 6.1.5: the second field's rows sit below the like-numbered rows
    of the first, so field 1 row r is frame row 2r+1."""
    block = FrameHeaders(video(height=4, interlaced=True), payload_size=5, ssrc=_SSRC)
    # Line 10 octets: field 0 rows 0 and 2, field 1 rows 1 and 3.
    assert block.frame_offset_octets.tolist() == [0, 5, 20, 25, 10, 15, 30, 35]


# --- What a caller has to know --------------------------------------------


def test_stamping_reuses_the_one_block_rather_than_allocating_per_frame():
    """Documented and depended on: the block is built once and restamped, so a
    caller queueing two frames at once copies the first."""
    block = headers()
    first = block.stamp(0)
    second = block.stamp(1)
    assert first is second


def test_the_payload_limit_leaves_room_for_the_headers_and_the_udp_header():
    """SMPTE ST 2110-10 section 6.3: "The Standard UDP Size Limit shall be 1460
    octets. The UDP Size ... includes the length of the UDP header (8 octets)
    and also the RTP headers and data." So 28 octets of the limit are spoken
    for before any sample data — the limit is not a payload size."""
    assert max_payload_size(video()) == 1460 - UDP_HEADER_SIZE - PACKET_HEADER_SIZE
    assert (
        max_payload_size(video(max_udp=8960))
        == 8960 - UDP_HEADER_SIZE - PACKET_HEADER_SIZE
    )


def test_a_payload_that_would_overrun_the_declared_udp_limit_is_refused():
    """Section 6.3: "Senders shall not generate IP Datagrams containing UDP
    packet sizes larger than this limit". A 1440-octet payload tiles a
    576-pixel line and puts 1468 octets on the wire, which is eight too many.
    """
    wide = video(width=576)
    assert choose_payload_size(wide, wide.max_udp) == 1440, "the limit misread"
    with pytest.raises(ValueError, match="UDP size"):
        FrameHeaders(wide, payload_size=1440, ssrc=_SSRC)
    # Sized against the payload limit instead, it fits with nothing to spare.
    payload_size = choose_payload_size(wide, max_payload_size(wide))
    frame = FrameHeaders(wide, payload_size, ssrc=_SSRC)
    assert payload_size + PACKET_HEADER_SIZE + UDP_HEADER_SIZE <= wide.max_udp
    assert frame.packet_count == 2 * wide.height


def test_a_payload_size_that_does_not_tile_a_line_is_refused():
    """Eight pixels is a 20-octet line, which 15 octets leaves a remainder of."""
    with pytest.raises(ValueError, match="does not tile"):
        FrameHeaders(video(width=8), payload_size=15, ssrc=_SSRC)


def test_a_payload_size_that_is_not_a_whole_number_of_pgroups_is_refused():
    """RFC 4175 section 4.2: an SRD length "MUST be a multiple of the pgroup".
    A divisor of the line need not be one — 2 divides a 10-octet line and is
    not a multiple of the 5-octet pgroup."""
    with pytest.raises(ValueError, match="pgroup"):
        FrameHeaders(video(), payload_size=2, ssrc=_SSRC)


@pytest.mark.parametrize("payload_type", [-1, 128])
def test_a_payload_type_outside_the_seven_bit_field_is_refused(payload_type: int):
    with pytest.raises(ValueError, match="payload type"):
        FrameHeaders(video(), payload_size=5, payload_type=payload_type, ssrc=_SSRC)


@pytest.mark.parametrize("ssrc", [-1, 1 << 32])
def test_an_ssrc_outside_its_thirty_two_bits_is_refused(ssrc: int):
    with pytest.raises(ValueError, match="SSRC"):
        FrameHeaders(video(), payload_size=5, ssrc=ssrc)


def test_a_negative_frame_index_is_refused():
    with pytest.raises(ValueError, match="frame index"):
        headers().stamp(-1)


@pytest.mark.parametrize("height", [0, -1])
def test_a_raster_with_no_rows_is_refused(height: int):
    """A frame of no packets has no last packet to mark, so this has to be
    refused where it is named rather than indexed into an empty block."""
    with pytest.raises(ValueError, match="height"):
        FrameHeaders(video(height=height), payload_size=5, ssrc=_SSRC)


def test_a_raster_taller_than_the_row_number_field_is_refused():
    """The SRD Row Number is fifteen bits, so a taller image would truncate
    into the F bit and announce the wrong field."""
    with pytest.raises(ValueError, match="32767"):
        FrameHeaders(video(height=40000), payload_size=5, ssrc=_SSRC)


def test_a_raster_wider_than_the_offset_field_is_refused():
    """The SRD Offset is fifteen bits under the C bit, and ST 2110-20 section
    7.2 bounds the width at 32767 for it. Unbounded, a 40000-pixel line sets
    the Line Continuation bit on 28 of its 160 packets and the receive parse
    reads 188 descriptors out of them — the extra ones out of sample data."""
    with pytest.raises(ValueError, match="width"):
        FrameHeaders(video(width=40000, height=2), payload_size=1250, ssrc=_SSRC)


def test_a_payload_larger_than_the_srd_length_field_is_refused():
    """RFC 4175's SRD Length is sixteen bits. A 32766-pixel line under a
    200,000-octet limit tiles in one 81,915-octet packet, which the field
    would carry as 16,379."""
    wide = video(width=32766, height=2, max_udp=200_000)
    payload_size = choose_payload_size(wide, max_payload_size(wide))
    assert payload_size == 81_915, "the reproduction no longer reproduces"
    with pytest.raises(ValueError, match="SRD Length"):
        FrameHeaders(wide, payload_size, ssrc=_SSRC)


@pytest.mark.parametrize("initial_sequence", [-1, 1 << 32])
def test_an_initial_sequence_outside_the_counter_is_refused(initial_sequence: int):
    """The counter the RTP and extended sequence numbers split is thirty-two
    bits, so a start outside it names a number no packet can carry."""
    with pytest.raises(ValueError, match="sequence number"):
        FrameHeaders(
            video(), payload_size=5, ssrc=_SSRC, initial_sequence=initial_sequence
        )


def test_a_frame_index_past_what_the_arithmetic_holds_is_refused():
    """The contract promises ValueError. Unbounded, the index reaches numpy as
    a multiplier and raises OverflowError, which no caller of it expects."""
    with pytest.raises(ValueError, match="frame index"):
        headers().stamp(1 << 62)


def test_a_sampling_this_library_refuses_is_refused_here_too():
    """4:2:0 pgroups span two rows, so none of the arithmetic holds."""
    with pytest.raises(ValueError, match="pgroup"):
        FrameHeaders(video(sampling="YCbCr-4:2:0"), payload_size=5, ssrc=_SSRC)
