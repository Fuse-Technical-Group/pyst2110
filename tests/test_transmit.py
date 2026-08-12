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

from pyst2110.sdp import SdpVideo
from pyst2110.transmit import PACKET_HEADER_SIZE, FrameHeaders

_SSRC = 0x1234ABCD
_PAYLOAD_TYPE = 96


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
    assert block.shape == (4, PACKET_HEADER_SIZE)
    assert [row.tolist() for row in block] == [octets(one) for one in _FRAME]


def test_the_block_is_one_row_per_packet_of_uint8():
    """Arrays in, arrays out — the shape the rest of the library speaks."""
    block = headers().stamp(0)
    assert block.dtype == np.uint8
    assert block.ndim == 2


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
    rows = block[:, 16].astype(int) << 8 | block[:, 17]
    offsets = block[:, 18].astype(int) << 8 | block[:, 19]
    assert rows.tolist() == [0, 0, 1, 1]
    assert offsets.tolist() == [0, 2, 0, 2]


def test_the_srd_length_is_the_payload_size_in_octets():
    """RFC 4175 section 4.2: "Number of octets of data included from this scan
    line", a multiple of the pgroup — not a count of pgroups or of pixels."""
    block = headers().stamp(0)
    lengths = block[:, 14].astype(int) << 8 | block[:, 15]
    assert lengths.tolist() == [5, 5, 5, 5]


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
    low = block[:, 2].astype(int) << 8 | block[:, 3]
    high = block[:, 12].astype(int) << 8 | block[:, 13]
    assert low.tolist() == [65534, 65535, 0, 1]
    assert high.tolist() == [0, 0, 1, 1]


def counter(row: np.ndarray) -> int:
    """The whole 32-bit sequence: the extended half above the RTP field."""
    return int.from_bytes(bytes(row[12:14].tolist() + row[2:4].tolist()), "big")


def test_sequence_numbers_run_on_across_frames_without_a_gap():
    block = headers()
    first = [counter(row) for row in block.stamp(0)]
    second = [counter(row) for row in block.stamp(1)]
    assert first + second == list(range(65534, 65534 + 8))


def test_the_thirty_two_bit_counter_wraps_rather_than_overflowing():
    block = FrameHeaders(
        video(), payload_size=5, ssrc=_SSRC, initial_sequence=(1 << 32) - 2
    ).stamp(0)
    low = (block[:, 2].astype(int) << 8 | block[:, 3]).tolist()
    high = (block[:, 12].astype(int) << 8 | block[:, 13]).tolist()
    assert low == [65534, 65535, 0, 1]
    assert high == [65535, 65535, 0, 0]


# --- The media timestamp --------------------------------------------------


def test_every_packet_of_a_progressive_frame_shares_one_timestamp():
    """ST 2110-20 section 6.1.3: "All RTP packets which are part of the same
    progressive frame shall contain the same RTP Timestamp value"."""
    block = headers().stamp(7)
    stamps = {tuple(row[4:8].tolist()) for row in block}
    assert len(stamps) == 1


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
    assert int.from_bytes(bytes(block[0, 4:8].tolist()), "big") == expected


def test_a_timestamp_far_into_a_run_is_exact_rather_than_drifted():
    """Derived from the frame index, so an hour in is as exact as frame one."""
    rate = Fraction(60000, 1001)
    block = FrameHeaders(video(frame_rate=rate), payload_size=5, ssrc=_SSRC)
    index = 60 * 60 * 60  # an hour of frames
    expected = index * 90_000 * rate.denominator // rate.numerator
    stamped = block.stamp(index)
    assert int.from_bytes(bytes(stamped[0, 4:8].tolist()), "big") == expected % (
        1 << 32
    )


def test_the_timestamp_wraps_within_its_thirty_two_bits():
    rate = Fraction(25)
    block = FrameHeaders(video(frame_rate=rate), payload_size=5, ssrc=_SSRC)
    index = (1 << 32) // 3600 + 1  # past the 32-bit clock's roll-over
    stamped = block.stamp(index)
    assert int.from_bytes(bytes(stamped[0, 4:8].tolist()), "big") == (index * 3600) % (
        1 << 32
    )


# --- Interlace ------------------------------------------------------------


def test_an_interlaced_frame_marks_and_stamps_each_field_separately():
    """ST 2110-20 section 6.1.2 sets the marker on the last packet of a field,
    and section 6.1.3 gives each field its own timestamp."""
    block = FrameHeaders(
        video(height=4, interlaced=True), payload_size=5, ssrc=_SSRC
    ).stamp(0)
    assert block.shape[0] == 8
    marked = ((block[:, 1] & 0x80) != 0).tolist()
    assert marked == [False, False, False, True, False, False, False, True]

    fields = ((block[:, 16] & 0x80) != 0).tolist()
    assert fields == [False] * 4 + [True] * 4
    rows = (block[:, 16].astype(int) & 0x7F) << 8 | block[:, 17]
    assert rows.tolist() == [0, 0, 1, 1, 0, 0, 1, 1], "rows restart in each field"

    stamps = [int.from_bytes(bytes(row[4:8].tolist()), "big") for row in block]
    # 25 frames a second is 50 fields, so the second field is 1800 ticks on.
    assert stamps == [0] * 4 + [1800] * 4


def test_an_odd_height_gives_the_first_field_the_extra_line():
    """Section 6.1.5: "if the height is odd, the temporally first field shall
    contain one more line than the temporally second field"."""
    block = FrameHeaders(
        video(height=5, interlaced=True), payload_size=5, ssrc=_SSRC
    ).stamp(0)
    fields = ((block[:, 16] & 0x80) != 0).tolist()
    assert fields.count(False) == 6, "three lines of two packets in the first field"
    assert fields.count(True) == 4, "two lines in the second"


# --- Where the payload comes from -----------------------------------------


def test_the_frame_offsets_walk_the_raster_a_packet_at_a_time():
    """The transmit-side descriptor: which octets of the frame buffer this
    packet carries. Never the pixels themselves (§spec:scope-boundary)."""
    block = headers()
    assert block.frame_offset.tolist() == [0, 5, 10, 15]
    assert block.payload_size == 5


def test_the_frame_offsets_of_an_interlaced_frame_interleave_the_fields():
    """Section 6.1.5: the second field's rows sit below the like-numbered rows
    of the first, so field 1 row r is frame row 2r+1."""
    block = FrameHeaders(video(height=4, interlaced=True), payload_size=5, ssrc=_SSRC)
    # Line 10 octets: field 0 rows 0 and 2, field 1 rows 1 and 3.
    assert block.frame_offset.tolist() == [0, 5, 20, 25, 10, 15, 30, 35]


def test_the_offsets_and_headers_agree_on_how_many_packets_there_are():
    block = headers()
    assert block.packets == 4
    assert block.frame_offset.size == block.packets
    assert block.stamp(0).shape[0] == block.packets


# --- What a caller has to know --------------------------------------------


def test_stamping_reuses_the_one_block_rather_than_allocating_per_frame():
    """Documented and depended on: the block is built once and restamped, so a
    caller queueing two frames at once copies the first."""
    block = headers()
    first = block.stamp(0)
    second = block.stamp(1)
    assert first is second


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


def test_a_raster_taller_than_the_row_number_field_is_refused():
    """The SRD Row Number is fifteen bits, so a taller image would truncate
    into the F bit and announce the wrong field."""
    with pytest.raises(ValueError, match="32767"):
        FrameHeaders(video(height=40000), payload_size=5, ssrc=_SSRC)


def test_a_sampling_this_library_refuses_is_refused_here_too():
    """4:2:0 pgroups span two rows, so none of the arithmetic holds."""
    with pytest.raises(ValueError, match="pgroup"):
        FrameHeaders(video(sampling="YCbCr-4:2:0"), payload_size=5, ssrc=_SSRC)
