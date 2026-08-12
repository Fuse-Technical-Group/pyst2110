"""Pgroup and packet arithmetic for a video format (SPEC §spec:geometry).

The pgroup sizes are read off SMPTE ST 2110-20 Table 1 (4:4:4) and Table 2
(4:2:2), and the packet counts are checked against formats whose answers can
be worked out by hand.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from pyst2110.geometry import (
    byte_offset,
    choose_payload_size,
    fits_raster,
    line_bytes,
    packets_per_frame,
    packets_per_line,
    pgroup,
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
    with pytest.raises(ValueError, match="no pgroup"):
        pgroup(video(sampling="YCbCr-4:2:0", depth=10))


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
    which — so half the frame's height is the bound (ST 2110-20 §6.1.5)."""
    lines = np.array([539, 540, 1079], dtype=np.int64)
    offsets = np.zeros(3, dtype=np.int64)
    interlaced = video(interlaced=True)
    assert fits_raster(interlaced, lines, offsets).tolist() == [True, False, False]
    assert fits_raster(video(), lines, offsets).tolist() == [True, True, True]


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
    start = int(line[0]) * line_bytes(current) + int(byte_offset(current, offset)[0])
    assert start > current.height * line_bytes(current)
    assert fits_raster(current, line, offset).tolist() == [False]


def test_the_last_packet_of_a_line_ends_exactly_at_the_line():
    """No overlap and no hole: the tiling the roadmap's verify step checks."""
    current = video()
    payload = choose_payload_size(current, 1400)
    per_line = packets_per_line(current, payload)
    group_bytes, group_pixels = pgroup(current)
    starts = np.arange(per_line, dtype=np.int64) * payload // group_bytes * group_pixels
    assert byte_offset(current, starts).tolist() == [0, 1200, 2400, 3600]
    assert byte_offset(current, starts)[-1] + payload == line_bytes(current)
