"""Every carried format at rasters no broadcast standard names (SPEC §spec:geometry).

Each sampling and depth ST 2110-20 Tables 1 to 3 list, crossed with cinema,
tiled, portrait, odd and tiny rasters. A case checks the packet count against
the frame's size worked out from samples a pixel, the round trip from built
headers through the receive parse back to the placement, and the round trip
of the offer. The expected sizes come from the sample structure rather than
from the pgroup tables, so a transcription slip in a table fails here.

These are round trips and arithmetic identities, the additional property of
§spec:testing; the hand-written vectors in ``test_geometry.py`` and
``test_transmit.py`` tie the library to the documents.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from pyst2110 import (
    SdpFlow,
    SdpVideo,
    choose_payload_size,
    fits_raster,
    format_sdp,
    frame_bytes,
    line_bytes,
    packets_per_frame,
    packets_per_line,
    parse_payload_headers,
    parse_rtp,
    parse_video_format,
    pgroup,
    pgroup_rows,
    raster_offset,
)
from pyst2110.transmit import PACKET_HEADER_SIZE, FrameHeaders, max_payload_size

_FLOW = SdpFlow("239.100.0.1", 20000, "192.168.100.2")
_OCTET_BITS = 8

# Samples a pixel carries, in halves: three components at 4:4:4, a luminance
# sample and half of two colour difference samples at 4:2:2, and a luminance
# sample and a quarter of two at 4:2:0.
_HALF_SAMPLES_PER_PIXEL = {"4:4:4": 6, "4:2:2": 4, "4:2:0": 3}

# ST 2110-20 Tables 1, 2 and 3: every sampling and the depths listed for it.
_FORMATS = [
    *(
        (sampling, depth, "4:4:4")
        for sampling in ("YCbCr-4:4:4", "CLYCbCr-4:4:4", "ICtCp-4:4:4", "RGB")
        for depth in (8, 10, 12, 16)
    ),
    *(("XYZ", depth, "4:4:4") for depth in (12, 16)),
    *(
        (sampling, depth, "4:2:2")
        for sampling in ("YCbCr-4:2:2", "CLYCbCr-4:2:2", "ICtCp-4:2:2")
        for depth in (8, 10, 12, 16)
    ),
    *(
        (sampling, depth, "4:2:0")
        for sampling in ("YCbCr-4:2:0", "CLYCbCr-4:2:0", "ICtCp-4:2:0")
        for depth in (8, 10, 12)
    ),
]

# Every width here is a whole number of the widest pgroup, four columns, so
# each raster fits every format; 1081 rows fits every format but 4:2:0.
_RASTERS = [
    (2048, 858),  # DCI 2K scope
    (4096, 2160),  # DCI 4K
    (5120, 2880),
    (7680, 2160),  # two 4K tiles side by side
    (3840, 1080),
    (1080, 1920),  # portrait
    (1920, 1081),  # odd height
    (1000, 562),  # a line no packet size near the limit divides
    (64, 48),
]


# Each format with each raster it carries: 4:2:0 needs whole row pairs, so
# 1081 rows is refused there, which a case below checks.
_CARRIED = [
    pytest.param(entry, raster, id=f"{entry[0]}-{entry[1]}-{raster[0]}x{raster[1]}")
    for entry in _FORMATS
    for raster in _RASTERS
    if not (entry[2] == "4:2:0" and raster[1] % 2)
]


def _video(sampling: str, depth: int, width: int, height: int) -> SdpVideo:
    return SdpVideo(
        width=width,
        height=height,
        frame_rate=Fraction(60000, 1001),
        depth=depth,
        sampling=sampling,
        colorimetry="BT2020",
    )


def _format_id(entry: tuple[str, int, str]) -> str:
    return f"{entry[0]}-{entry[1]}"


def _raster_id(raster: tuple[int, int]) -> str:
    return f"{raster[0]}x{raster[1]}"


def _check_round_trip(video: SdpVideo, family: str) -> None:
    """Size, build, parse and place one frame, and compare at every step."""
    expected_bits = (
        video.width * video.height * _HALF_SAMPLES_PER_PIXEL[family] * video.depth
    )
    assert expected_bits % (2 * _OCTET_BITS) == 0
    assert frame_bytes(video) == expected_bits // (2 * _OCTET_BITS)
    assert line_bytes(video) * video.height // pgroup_rows(video) == frame_bytes(video)

    payload_size = choose_payload_size(video, max_payload_size(video))
    assert payload_size % pgroup(video)[0] == 0
    assert line_bytes(video) % payload_size == 0
    count = packets_per_frame(video, payload_size)
    assert count * payload_size == frame_bytes(video)

    headers = FrameHeaders(video, payload_size)
    assert headers.packet_count == count
    block = headers.stamp(0)
    sizes = np.full(count, PACKET_HEADER_SIZE, dtype=np.int64)
    rtp = parse_rtp(block, sizes=sizes)
    payload = parse_payload_headers(block, rtp.payload_offset, sizes=sizes)

    assert not payload.overflowed.any()
    assert (payload.length == payload_size).all()
    assert (payload.line % pgroup_rows(video) == 0).all()
    assert fits_raster(video, payload.line, payload.offset_samples).all()
    starts = raster_offset(video, payload.line, payload.offset_samples)
    # Packets in order tile the frame buffer end to end: no hole, no overlap.
    tiled = np.arange(count, dtype=np.int64) * payload_size
    assert np.array_equal(starts, tiled)
    assert np.array_equal(headers.frame_offset_octets, tiled)
    assert int(rtp.marker.sum()) == 1
    assert bool(rtp.marker[-1])


@pytest.mark.parametrize(("entry", "raster"), _CARRIED)
def test_a_raster_round_trips_from_headers_to_placement(
    entry: tuple[str, int, str], raster: tuple[int, int]
):
    sampling, depth, family = entry
    _check_round_trip(_video(sampling, depth, *raster), family)


@pytest.mark.parametrize("raster", _RASTERS, ids=_raster_id)
@pytest.mark.parametrize("entry", _FORMATS, ids=_format_id)
def test_a_raster_round_trips_through_its_offer(
    entry: tuple[str, int, str], raster: tuple[int, int]
):
    sampling, depth, _ = entry
    described = _video(sampling, depth, *raster)
    assert parse_video_format(format_sdp(_FLOW, described)) == described


@pytest.mark.parametrize("entry", _FORMATS, ids=_format_id)
def test_one_pgroup_across_is_the_smallest_raster(entry: tuple[str, int, str]):
    """A width of one pgroup, four pixels at 4:4:4 10-bit, and two row pairs
    of height: one packet a line."""
    sampling, depth, family = entry
    probe = _video(sampling, depth, 64, 48)
    columns, rows = pgroup(probe)[1], pgroup_rows(probe)
    video = _video(sampling, depth, columns, 2 * rows)
    _check_round_trip(video, family)
    assert packets_per_line(video, pgroup(video)[0]) == 1


@pytest.mark.parametrize("entry", _FORMATS, ids=_format_id)
def test_a_line_no_packet_at_the_limit_divides_takes_smaller_equal_packets(
    entry: tuple[str, int, str],
):
    """At 1000 columns no format's line is a whole number of the largest
    payload the standard UDP limit admits, so that payload is refused and
    the chosen one is smaller and tiles."""
    sampling, depth, _ = entry
    video = _video(sampling, depth, 1000, 562)
    limit = max_payload_size(video)
    usable = limit - limit % pgroup(video)[0]
    assert line_bytes(video) % usable
    with pytest.raises(ValueError, match="does not tile"):
        packets_per_line(video, usable)
    assert choose_payload_size(video, limit) < usable


@pytest.mark.parametrize(
    "entry",
    # A one-column pgroup divides every width.
    [one for one in _FORMATS if pgroup(_video(one[0], one[1], 64, 48))[1] > 1],
    ids=_format_id,
)
def test_a_width_the_pgroup_does_not_divide_is_refused(entry: tuple[str, int, str]):
    sampling, depth, _ = entry
    columns = pgroup(_video(sampling, depth, 64, 48))[1]
    video = _video(sampling, depth, 1920 + columns // 2, 1080)
    with pytest.raises(ValueError, match=f"{columns}-pixel pgroups"):
        line_bytes(video)
    with pytest.raises(ValueError, match="whole number"):
        FrameHeaders(video, pgroup(video)[0])


@pytest.mark.parametrize(
    "entry", [one for one in _FORMATS if one[2] == "4:2:0"], ids=_format_id
)
@pytest.mark.parametrize("height", [1081, 1, 47])
def test_a_420_height_the_row_pair_does_not_divide_is_refused(
    entry: tuple[str, int, str], height: int
):
    sampling, depth, _ = entry
    video = _video(sampling, depth, 1920, height)
    with pytest.raises(ValueError, match=f"height of {height} is not a whole"):
        frame_bytes(video)
    with pytest.raises(ValueError, match=f"height of {height} is not a whole"):
        FrameHeaders(video, pgroup(video)[0])
