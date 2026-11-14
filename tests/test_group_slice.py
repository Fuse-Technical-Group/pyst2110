"""One essence over four flows, from the top-level package (§spec:multi-stream).

The roadmap's verify path for a group, end to end through the public
surface: a sender writes the group offer for a 7680x4320 essence, a receiver
reads it back, sizes each constituent as the single-flow geometry sizes any
raster, and places every constituent pixel in the essence through the
division mapping.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from pyst2110 import (
    MULTI_2SI,
    MULTI_SD,
    SdpFlow,
    SdpVideo,
    format_group_sdp,
    line_bytes,
    parse_group_sdp,
    parse_sdp,
    reassemble,
    split,
    sub_image_size,
    to_constituent,
    to_essence,
)

_SENDER = "192.0.2.1"
_FLOWS = tuple(SdpFlow(f"239.10.0.{n}", 20000, _SENDER) for n in range(1, 5))
_SUB_IMAGE = SdpVideo(
    width=3840,
    height=2160,
    frame_rate=Fraction(60000, 1001),
    depth=10,
    sampling="YCbCr-4:2:2",
    colorimetry="BT709",
)


@pytest.mark.parametrize("division", [MULTI_2SI, MULTI_SD])
def test_a_receiver_provisioned_from_a_group_places_every_constituent_pixel(
    division: str,
):
    group = parse_group_sdp(format_group_sdp(_FLOWS, _SUB_IMAGE, division=division))
    essence = {"width": group.essence_width, "height": group.essence_height}
    assert essence == {"width": 7680, "height": 4320}
    assert sub_image_size(division=division, count=4, **essence) == (3840, 2160)
    # Each constituent is a 2160p raster to the single-flow geometry.
    assert line_bytes(group.video) == 3840 * 5 // 2

    # The first and last row of every constituent, placed in the essence and
    # mapped back: each pixel lands where the forward mapping says it came.
    xs = np.arange(group.video.width)
    for number in range(1, len(group.flows) + 1):
        for row in (0, group.video.height - 1):
            ex, ey = to_essence(number, xs, row, division=division, count=4, **essence)
            back, sub_x, sub_y = to_constituent(
                ex, ey, division=division, count=4, **essence
            )
            assert (back == number).all()
            np.testing.assert_array_equal(sub_x, xs)
            assert (sub_y == row).all()


def test_a_group_is_not_read_as_one_flow():
    with pytest.raises(ValueError, match="parse_group_sdp"):
        parse_sdp(format_group_sdp(_FLOWS, _SUB_IMAGE))


def test_the_reference_split_reassembles_an_essence_frame():
    rng = np.random.default_rng(23)
    frame = rng.integers(0, 1024, (432, 768, 3), dtype=np.uint16)
    parts = split(frame, division=MULTI_2SI, count=16)
    assert all(part.shape == (108, 192, 3) for part in parts)
    np.testing.assert_array_equal(reassemble(parts, division=MULTI_2SI), frame)
