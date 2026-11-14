"""Pgroup and packet arithmetic for a video format (SPEC §spec:geometry).

The pgroup sizes are read off SMPTE ST 2110-20 Table 1 (4:4:4), Table 2
(4:2:2) and Table 3 (4:2:0), and the packet counts are checked against
formats whose answers can be worked out by hand.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np
import pytest

import pyst2110
from pyst2110.geometry import (
    byte_offset,
    choose_payload_size,
    fits_raster,
    frame_bytes,
    line_bytes,
    packets_per_frame,
    packets_per_line,
    pgroup,
    pgroup_rows,
    raster_offset,
    rows_per_field,
    sample_offset,
)
from pyst2110.sdp import SdpVideo


def video(**overrides: Any) -> SdpVideo:
    fields: dict[str, Any] = {
        "width": 1920,
        "height": 1080,
        "frame_rate": Fraction(50),
        "depth": 10,
        "sampling": "YCbCr-4:2:2",
    }
    return SdpVideo(**(fields | overrides))


@pytest.mark.parametrize(
    ("sampling", "depth", "expected"),
    [
        # ST 2110-20 Table 2: 4:2:2 covers two pixels at every depth.
        ("YCbCr-4:2:2", 8, (4, 2)),
        ("YCbCr-4:2:2", 10, (5, 2)),
        ("YCbCr-4:2:2", 12, (6, 2)),
        ("YCbCr-4:2:2", 16, (8, 2)),
        ("ICtCp-4:2:2", 10, (5, 2)),
        ("CLYCbCr-4:2:2", 12, (6, 2)),
        # ST 2110-20 Table 1: 4:4:4 needs four pixels to fill whole octets at
        # 10 bits, and two at 12.
        ("YCbCr-4:4:4", 8, (3, 1)),
        ("YCbCr-4:4:4", 10, (15, 4)),
        ("RGB", 10, (15, 4)),
        ("RGB", 12, (9, 2)),
        ("RGB", 8, (3, 1)),
        ("XYZ", 12, (9, 2)),
    ],
)
def test_a_pgroup_is_the_formats_own(
    sampling: str, depth: int, expected: tuple[int, int]
):
    assert pgroup(video(sampling=sampling, depth=depth)) == expected


@pytest.mark.parametrize(
    ("sampling", "depth"),
    [
        ("YCbCr-4:2:2", 10),
        ("YCbCr-4:4:4", 10),
        ("RGB", 12),
        ("YCbCr-4:2:2", 16),
    ],
)
def test_a_pgroup_holds_a_whole_number_of_octets_and_pixels(sampling: str, depth: int):
    """The invariant the table exists to satisfy: depth * pixels is whole
    octets. Checking it catches a transcription slip the table cannot."""
    group_bytes, group_pixels = pgroup(video(sampling=sampling, depth=depth))
    components = 3 if "4:4:4" in sampling or sampling in ("RGB", "XYZ") else 2
    assert group_pixels * depth * components == group_bytes * 8


def test_an_unsupported_format_names_what_is_supported():
    with pytest.raises(ValueError, match="no pgroup") as raised:
        pgroup(video(sampling="YCbCr-4:1:1", depth=10))
    assert "YCbCr-4:2:0" in str(raised.value)


# --- 4:2:0 ------------------------------------------------------------------
#
# ST 2110-20 section 6.2.5 and Table 3. A 4:2:0 pgroup holds samples from two
# consecutive luminance rows (Figure 5: Y'00-Y'01-Y'10-Y'11-CB'00-CR'00), so
# Table 3's "pgroup coverage (pixels)" counts both rows: 4, 8 and 4 pixels
# are 2, 4 and 2 columns of two rows. The SRD Offset is a column of the
# full-bandwidth pixel matrix (section 6.1.4), so it advances by the columns.


@pytest.mark.parametrize("sampling", ["YCbCr-4:2:0", "CLYCbCr-4:2:0", "ICtCp-4:2:0"])
@pytest.mark.parametrize(
    ("depth", "octets", "coverage"),
    [(8, 6, 4), (10, 15, 8), (12, 9, 4)],
)
def test_a_420_pgroup_is_table_3s_coverage_over_two_rows(
    sampling: str, depth: int, octets: int, coverage: int
):
    current = video(sampling=sampling, depth=depth)
    group_bytes, group_pixels = pgroup(current)
    assert pgroup_rows(current) == 2
    assert group_bytes == octets
    assert group_pixels * pgroup_rows(current) == coverage
    # Four luminance samples and two colour difference samples for every two
    # columns of two rows: three samples a column, at the stated depth.
    assert group_pixels * 3 * depth == group_bytes * 8


@pytest.mark.parametrize(
    "sampling", ["YCbCr-4:4:4", "RGB", "XYZ", "YCbCr-4:2:2", "ICtCp-4:2:2"]
)
def test_every_other_pgroup_spans_one_row(sampling: str):
    assert pgroup_rows(video(sampling=sampling, depth=12)) == 1


def test_a_420_depth_table_3_does_not_list_is_refused():
    """Table 3 stops at 12 bits; RFC 4175's 16-bit 4:2:0 pgroup is not one
    ST 2110-20 carries."""
    with pytest.raises(ValueError, match=r"\[8, 10, 12\]"):
        pgroup(video(sampling="YCbCr-4:2:0", depth=16))


def test_interlaced_420_is_refused():
    """Section 6.2.5: "This sampling system does not apply to PsF or
    interlaced video essence"."""
    with pytest.raises(ValueError, match=r"6\.2\.5"):
        pgroup(video(sampling="YCbCr-4:2:0", interlaced=True))


def test_the_row_pair_geometry_is_exported_from_the_package():
    assert pyst2110.pgroup_rows is pgroup_rows
    assert pyst2110.frame_bytes is frame_bytes
    assert {"pgroup_rows", "frame_bytes"} <= set(pyst2110.__all__)


def test_a_420_line_is_the_row_pair_one_srd_row_number_names():
    """1920 columns at four a 15-octet pgroup: 480 pgroups, 7200 octets, and
    two luminance rows of samples in them."""
    current = video(sampling="YCbCr-4:2:0")
    assert line_bytes(current) == 7200
    payload = choose_payload_size(current, 1400)
    # 93 pgroups fit 1400 octets; 80 is the largest count dividing 480.
    assert payload == 1200
    assert packets_per_line(current, payload) == 6
    assert packets_per_frame(current, payload) == 6 * 540


def test_a_frame_is_its_row_pairs_for_420_and_its_rows_otherwise():
    """Ten bits of 1.5 samples a pixel over 1920x1080 is 3,888,000 octets."""
    assert frame_bytes(video(sampling="YCbCr-4:2:0")) == 3_888_000
    assert frame_bytes(video()) == 1080 * 4800
    assert frame_bytes(video(interlaced=True)) == 1080 * 4800


def test_a_420_height_of_an_odd_row_count_is_refused():
    """A pgroup holds two rows and is never fragmented (section 6.2.2), so a
    raster ending in half a pair has a row no pgroup can carry."""
    odd = video(sampling="YCbCr-4:2:0", height=1081)
    with pytest.raises(ValueError, match="1081"):
        packets_per_frame(odd, 1200)
    with pytest.raises(ValueError, match="1081"):
        frame_bytes(odd)


def test_a_420_row_number_is_the_first_row_of_its_pair():
    """Section 6.1.5: "Only every other Sample Row shall be signaled", so an
    odd row names no pair and does not fit, however small."""
    current = video(sampling="YCbCr-4:2:0")
    lines = np.array([0, 1, 2, 1078, 1079, 1080], dtype=np.int64)
    offsets = np.zeros(6, dtype=np.int64)
    assert fits_raster(current, lines, offsets).tolist() == [
        True,
        False,
        True,
        True,
        False,
        False,
    ]


def test_a_420_raster_offset_counts_row_pairs():
    """Row 2 is the second pair, one 7200-octet line in; column 4 is one
    15-octet pgroup along it."""
    current = video(sampling="YCbCr-4:2:0")
    lines = np.array([0, 0, 2, 2, 1078], dtype=np.int64)
    offsets = np.array([0, 4, 0, 4, 1916], dtype=np.int64)
    assert raster_offset(current, lines, offsets).tolist() == [
        0,
        15,
        7200,
        7215,
        539 * 7200 + 479 * 15,
    ]
    assert byte_offset(current, offsets).tolist() == [0, 15, 0, 15, 479 * 15]
    assert sample_offset(current, byte_offset(current, offsets)).tolist() == (
        offsets.tolist()
    )


def test_a_line_is_a_whole_number_of_pgroups():
    """1920 pixels at two per pgroup, five octets each."""
    assert line_bytes(video()) == 4800
    assert line_bytes(video(width=1280)) == 3200
    assert line_bytes(video(width=1920, depth=8)) == 3840


def test_a_width_that_is_not_whole_pgroups_is_refused():
    with pytest.raises(ValueError, match="whole number"):
        line_bytes(video(width=1921))


@pytest.mark.parametrize("width", [0, -8])
def test_a_width_of_zero_is_refused_by_name(width: int):
    """An SDP declaring width=0 would otherwise reach the payload-size search
    with no pgroups to divide, and fail as something unrelated."""
    with pytest.raises(ValueError, match="not an image"):
        line_bytes(video(width=width))
    with pytest.raises(ValueError, match="not an image"):
        choose_payload_size(video(width=width), 1400)


def test_the_payload_size_tiles_a_line_exactly():
    """Uniform sizes are what let a stream declare its layout once."""
    current = video()
    payload = choose_payload_size(current, 1400)
    assert payload == 1200
    assert line_bytes(current) % payload == 0
    assert packets_per_line(current, payload) == 4
    assert packets_per_frame(current, payload) == 4320


@pytest.mark.parametrize(
    ("width", "limit"),
    [
        (1920, 1400),
        (1280, 1400),
        (3840, 1400),
        (720, 900),
        (1920, 500),
        # A limit smaller than two pgroups leaves only one per packet, which
        # is where a search over byte counts rather than pgroup counts breaks.
        (1920, 7),
        (1920, 5),
    ],
)
def test_every_chosen_payload_is_whole_pgroups_and_tiles_its_line(
    width: int, limit: int
):
    """RFC 4175 and ST 2110-20 both require the SRD length be a multiple of
    the pgroup octet length, so the search runs over pgroups per packet."""
    current = video(width=width)
    payload = choose_payload_size(current, limit)
    assert payload <= limit
    assert payload % pgroup(current)[0] == 0
    assert line_bytes(current) % payload == 0


def test_a_payload_of_one_pgroup_is_always_available():
    """Every line is a whole number of pgroups, so one per packet always
    tiles it — there is no limit at or above a pgroup that has no answer."""
    assert choose_payload_size(video(), 7) == 5
    assert choose_payload_size(video(), 5) == 5


def test_a_payload_smaller_than_a_pgroup_is_refused():
    with pytest.raises(ValueError, match="cannot hold one"):
        choose_payload_size(video(), 4)


def test_the_largest_payload_under_the_limit_is_chosen():
    """Fewer, larger packets is the point; a smaller answer would still tile."""
    assert choose_payload_size(video(), 4800) == 4800
    assert choose_payload_size(video(), 4799) == 2400


def test_a_payload_that_does_not_tile_is_refused():
    with pytest.raises(ValueError, match="does not tile"):
        packets_per_line(video(), 1000)


def test_a_frame_is_its_lines_packets_once_per_row():
    current = video(width=1920, height=1080)
    assert packets_per_line(current, 1200) == 4
    assert packets_per_frame(current, 1200) == 4320


def test_an_interlaced_frame_has_the_same_packets_as_a_progressive_one():
    """The two fields between them carry every row once, so the frame total
    does not change — only which field each row is announced in."""
    progressive = video(height=1080)
    interlaced = video(height=1080, interlaced=True)
    assert packets_per_frame(interlaced, 1200) == packets_per_frame(progressive, 1200)


def test_a_sample_offset_becomes_a_byte_offset_within_its_line():
    """ST 2110-20 calls the SRD Offset a Full-Bandwidth Sample Position, so
    it counts pixels and the pgroup is what turns it into octets."""
    offsets = np.array([0, 2, 480, 960, 1918], dtype=np.int64)
    # Two pixels to a five-octet pgroup: 480 pixels is 240 pgroups is 1200 B.
    assert byte_offset(video(), offsets).tolist() == [0, 5, 1200, 2400, 4795]


def test_byte_offsets_follow_the_formats_own_pgroup():
    offsets = np.array([0, 4, 8], dtype=np.int64)
    # 4:4:4 at 10 bits is four pixels to fifteen octets.
    assert byte_offset(video(sampling="RGB", depth=10), offsets).tolist() == [0, 15, 30]


def test_a_byte_offset_is_returned_packet_aligned():
    offsets = np.array([0, 480, 960, 1440], dtype=np.int64)
    result = byte_offset(video(), offsets)
    assert result.shape == offsets.shape
    assert result.dtype == np.int64


def test_an_empty_offset_array_stays_empty():
    assert byte_offset(video(), np.array([], dtype=np.int64)).shape == (0,)


def test_a_descriptor_inside_the_raster_fits():
    lines = np.array([0, 42, 1079], dtype=np.int64)
    offsets = np.array([0, 480, 1918], dtype=np.int64)
    assert fits_raster(video(), lines, offsets).tolist() == [True, True, True]


def test_a_row_past_the_image_does_not_fit():
    """The SRD Row Number is fifteen bits, so a packet can name row 32767 of
    a 1080-row image and nothing in the parse can refuse it."""
    lines = np.array([1079, 1080, 32767], dtype=np.int64)
    offsets = np.zeros(3, dtype=np.int64)
    assert fits_raster(video(), lines, offsets).tolist() == [True, False, False]


def test_a_sample_position_past_the_row_does_not_fit():
    lines = np.zeros(3, dtype=np.int64)
    offsets = np.array([1919, 1920, 32767], dtype=np.int64)
    assert fits_raster(video(), lines, offsets).tolist() == [True, False, False]


def test_an_interlaced_row_is_bounded_by_the_field_not_the_frame():
    """An interlaced flow numbers rows within a field, and the F bit says
    which — so the first field's row count is the bound (ST 2110-20 §6.1.5)."""
    lines = np.array([539, 540, 1079], dtype=np.int64)
    offsets = np.zeros(3, dtype=np.int64)
    interlaced = video(interlaced=True)
    assert fits_raster(interlaced, lines, offsets).tolist() == [True, False, False]
    assert fits_raster(video(), lines, offsets).tolist() == [True, True, True]


@pytest.mark.parametrize(
    ("height", "interlaced", "expected"),
    [(1080, False, 1080), (1080, True, 540), (5, False, 5), (5, True, 3)],
)
def test_a_fields_rows_are_the_temporally_first_fields(
    height: int, interlaced: bool, expected: int
):
    """Section 6.1.5: "if the height is odd, the temporally first field shall
    contain one more line than the temporally second field"."""
    assert rows_per_field(video(height=height, interlaced=interlaced)) == expected


def test_an_odd_interlaced_height_admits_the_row_the_extra_line_needs():
    """Halving the height rounds the extra line away and rejects the row a
    sender builds for it — a descriptor this library made, refused by this
    library's own bound."""
    odd = video(height=5, interlaced=True)
    lines = np.array([0, 1, 2, 3], dtype=np.int64)
    offsets = np.zeros(4, dtype=np.int64)
    assert fits_raster(odd, lines, offsets).tolist() == [True, True, True, False]


def test_the_fit_mask_keeps_the_descriptors_own_shape():
    lines = np.array([0, 1, 2], dtype=np.int64)
    offsets = np.array([0, 2, 4], dtype=np.int64)
    mask = fits_raster(video(), lines, offsets)
    assert mask.shape == lines.shape
    assert mask.dtype == np.bool_
    assert fits_raster(video(), lines[:0], offsets[:0]).shape == (0,)


def test_the_hostile_descriptor_scales_but_does_not_fit():
    """One 20-octet datagram naming row 32767 at sample 32767 scales to an
    offset 157 MB into a 5 MB frame. The scaling is what it is; the mask is
    what keeps it away from a gather."""
    current = video()
    line = np.array([32767], dtype=np.int64)
    offset = np.array([32767], dtype=np.int64)
    assert byte_offset(current, offset).tolist() == [81915]
    start = raster_offset(current, line, offset)
    assert start.tolist() == [32767 * 4800 + 81915]
    assert int(start[0]) > current.height * line_bytes(current)
    assert fits_raster(current, line, offset).tolist() == [False]


def test_a_byte_offset_becomes_the_sample_position_that_names_it():
    """The inverse of byte_offset, and the direction a sender needs: it knows
    which octets a packet carries and has to say which pixel they start at."""
    octets = np.array([0, 5, 1200, 2400, 4795], dtype=np.int64)
    assert sample_offset(video(), octets).tolist() == [0, 2, 480, 960, 1918]


def test_a_sample_position_and_a_byte_offset_invert_each_other():
    samples = np.array([0, 2, 480, 960, 1918], dtype=np.int64)
    current = video()
    assert sample_offset(current, byte_offset(current, samples)).tolist() == (
        samples.tolist()
    )


def test_a_raster_offset_is_the_row_and_the_sample_position_together():
    """Where a descriptor's data belongs in a frame buffer — the library's
    central output, written here rather than at each of its callers."""
    current = video()
    lines = np.array([0, 0, 1, 1079], dtype=np.int64)
    offsets = np.array([0, 960, 0, 1918], dtype=np.int64)
    assert raster_offset(current, lines, offsets).tolist() == [
        0,
        2400,
        4800,
        1079 * 4800 + 4795,
    ]


def test_the_last_packet_of_a_line_ends_exactly_at_the_line():
    """No overlap and no hole: the tiling the roadmap's verify step checks."""
    current = video()
    payload = choose_payload_size(current, 1400)
    per_line = packets_per_line(current, payload)
    group_bytes, group_pixels = pgroup(current)
    starts = np.arange(per_line, dtype=np.int64) * payload // group_bytes * group_pixels
    assert byte_offset(current, starts).tolist() == [0, 1200, 2400, 3600]
    assert byte_offset(current, starts)[-1] + payload == line_bytes(current)
