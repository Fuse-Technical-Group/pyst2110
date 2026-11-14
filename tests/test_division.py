"""Where an essence pixel sits in an RP 2110-23 group (SPEC §spec:multi-stream).

The fixed vectors are read off SMPTE RP 2110-23:2019 by hand: Figure 1's
coloured grid for four-way two-sample interleave, Figure 2's numbered source
grid for sixteen-way, Figure 3's quadrant boundaries for four-way square
division and Figure 5's numbering for sixteen-way. None is computed from the
implementation's own formulas. The properties — every essence pixel lands in
exactly one constituent pixel, and reassembly undoes the split — run over
every pixel of rasters small enough to enumerate (§spec:testing).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from pyst2110.division import (
    MULTI_2SI,
    MULTI_SD,
    reassemble,
    split,
    sub_image_size,
    to_constituent,
    to_essence,
)

_8K = {"width": 7680, "height": 4320}
_UHD = {"width": 3840, "height": 2160}

# RP 2110-23 Figure 1: the top-left eight pixels of the first six rows of a
# 2160-line image, by the constituent each is coloured as (1 yellow, 2 blue,
# 3 green, 4 red).
_FIGURE_1 = [
    [1, 1, 2, 2, 1, 1, 2, 2],
    [3, 3, 4, 4, 3, 3, 4, 4],
    [1, 1, 2, 2, 1, 1, 2, 2],
    [3, 3, 4, 4, 3, 3, 4, 4],
    [1, 1, 2, 2, 1, 1, 2, 2],
    [3, 3, 4, 4, 3, 3, 4, 4],
]

# RP 2110-23 Figure 2: the 4320-line source image's first four rows and
# sixteen pixels, by the sub-image number printed in each sample.
_FIGURE_2 = [
    [1, 1, 5, 5, 2, 2, 6, 6, 1, 1, 5, 5, 2, 2, 6, 6],
    [9, 9, 13, 13, 10, 10, 14, 14, 9, 9, 13, 13, 10, 10, 14, 14],
    [3, 3, 7, 7, 4, 4, 8, 8, 3, 3, 7, 7, 4, 4, 8, 8],
    [11, 11, 15, 15, 12, 12, 16, 16, 11, 11, 15, 15, 12, 12, 16, 16],
]

# RP 2110-23 Figure 5: the sixteen 1080-line sub-images of a 4320-line image
# in square division, as tiles of the essence.
_FIGURE_5 = [
    [1, 2, 5, 6],
    [3, 4, 7, 8],
    [9, 10, 13, 14],
    [11, 12, 15, 16],
]

_MODES = [
    (MULTI_2SI, 4),
    (MULTI_2SI, 16),
    (MULTI_SD, 4),
    (MULTI_SD, 16),
]


def _point(x: int, y: int, **kwargs) -> tuple[int, int, int]:
    constituent, sub_x, sub_y = to_constituent(x, y, **kwargs)
    return int(constituent), int(sub_x), int(sub_y)


def test_four_way_interleave_colours_the_samples_as_figure_1_does():
    ys, xs = np.mgrid[0:6, 0:8]
    constituent, _, _ = to_constituent(xs, ys, division=MULTI_2SI, count=4, **_UHD)
    assert constituent.tolist() == _FIGURE_1


@pytest.mark.parametrize(
    ("essence", "expected"),
    [
        # Row 3 is odd, so constituent 3 or 4; pixels 6 and 7 are the second
        # pair of the second group of four, so 4; and constituent 4 already
        # holds pixels 2 and 3 of that row, so pixel 6 is its third, x = 2.
        ((6, 3), (4, 2, 1)),
        # Row 0, pixels 4 and 5 are constituent 1's second pair: x = 3 is
        # pixel 5, the second of that pair.
        ((5, 0), (1, 3, 0)),
        ((0, 0), (1, 0, 0)),
        ((7679, 4319), (4, 3839, 2159)),
    ],
)
def test_four_way_interleave_places_these_hand_counted_pixels(essence, expected):
    assert _point(*essence, division=MULTI_2SI, count=4, **_8K) == expected


@pytest.mark.parametrize(
    ("essence", "expected"),
    [
        ((3840, 2160), (4, 0, 0)),
        ((4000, 100), (2, 160, 100)),
        ((3839, 2159), (1, 3839, 2159)),
        ((0, 2160), (3, 0, 0)),
    ],
)
def test_four_way_square_division_cuts_the_8k_image_into_quadrants(essence, expected):
    assert _point(*essence, division=MULTI_SD, count=4, **_8K) == expected


@pytest.mark.parametrize(
    ("raster", "essence", "expected"),
    [
        # Figure 3's boundaries: 1919 is the last column of sub-images 1 and
        # 3, 1920 the first of 2 and 4; 1079 and 1080 the same for rows.
        (_UHD, (1919, 1079), (1, 1919, 1079)),
        (_UHD, (1920, 0), (2, 0, 0)),
        (_UHD, (0, 1080), (3, 0, 0)),
        (_UHD, (3839, 2159), (4, 1919, 1079)),
        # And its 4096-wide reading, 2047 and 2048.
        ({"width": 4096, "height": 2160}, (2047, 0), (1, 2047, 0)),
        ({"width": 4096, "height": 2160}, (2048, 0), (2, 0, 0)),
    ],
)
def test_four_way_square_division_follows_figure_3s_boundaries(
    raster, essence, expected
):
    assert _point(*essence, division=MULTI_SD, count=4, **raster) == expected


def test_sixteen_way_interleave_numbers_the_samples_as_figure_2_does():
    ys, xs = np.mgrid[0:4, 0:16]
    constituent, _, _ = to_constituent(xs, ys, division=MULTI_2SI, count=16, **_8K)
    assert constituent.tolist() == _FIGURE_2


@pytest.mark.parametrize(
    ("essence", "expected"),
    [
        # Figure 2 repeats every eight pixels and four rows, so sub-image 1
        # takes one pair in eight along a row and one row in four.
        ((0, 0), (1, 0, 0)),
        ((1, 0), (1, 1, 0)),
        ((8, 0), (1, 2, 0)),
        ((9, 4), (1, 3, 1)),
        # The first sample of sub-image 16 is row 3, pixel 6.
        ((6, 3), (16, 0, 0)),
        ((7679, 4319), (16, 1919, 1079)),
    ],
)
def test_sixteen_way_interleave_places_these_hand_counted_pixels(essence, expected):
    assert _point(*essence, division=MULTI_2SI, count=16, **_8K) == expected


def test_sixteen_way_square_division_numbers_the_tiles_as_figure_5_does():
    tile_ys, tile_xs = np.mgrid[0:4, 0:4]
    # The centre of each 1920x1080 tile of a 7680x4320 essence.
    xs = tile_xs * 1920 + 960
    ys = tile_ys * 1080 + 540
    constituent, sub_x, sub_y = to_constituent(
        xs, ys, division=MULTI_SD, count=16, **_8K
    )
    assert constituent.tolist() == _FIGURE_5
    assert (sub_x == 960).all()
    assert (sub_y == 540).all()


@pytest.mark.parametrize(
    ("essence", "expected"),
    [
        ((3840, 0), (5, 0, 0)),
        ((5000, 3000), (13, 1160, 840)),
        ((7679, 4319), (16, 1919, 1079)),
    ],
)
def test_sixteen_way_square_division_places_these_hand_counted_pixels(
    essence, expected
):
    assert _point(*essence, division=MULTI_SD, count=16, **_8K) == expected


@pytest.mark.parametrize(
    ("division", "count", "expected"),
    [
        (MULTI_2SI, 4, (3840, 2160)),
        (MULTI_2SI, 16, (1920, 1080)),
        (MULTI_SD, 4, (3840, 2160)),
        (MULTI_SD, 16, (1920, 1080)),
    ],
)
def test_the_sub_image_is_a_half_or_a_quarter_of_the_essence_each_way(
    division, count, expected
):
    assert sub_image_size(division=division, count=count, **_8K) == expected


@pytest.mark.parametrize(
    ("division", "count", "width", "height"),
    [
        # Four-way interleave deals out whole pixel pairs over two rows.
        (MULTI_2SI, 4, 3842, 2160),
        (MULTI_2SI, 4, 3840, 2161),
        # Sixteen-way does it twice.
        (MULTI_2SI, 16, 7684, 4320),
        (MULTI_2SI, 16, 7680, 4322),
        (MULTI_SD, 4, 3841, 2160),
        (MULTI_SD, 4, 3840, 2159),
        (MULTI_SD, 16, 7682, 4320),
        (MULTI_SD, 16, 7680, 4318),
    ],
)
def test_an_essence_the_division_cannot_cut_evenly_is_refused(
    division, count, width, height
):
    with pytest.raises(ValueError, match="divide"):
        sub_image_size(width, height, division=division, count=count)


def test_a_constituent_count_the_rp_does_not_define_is_refused():
    with pytest.raises(ValueError, match="4 or 16"):
        sub_image_size(division=MULTI_2SI, count=8, **_8K)


def test_the_temporal_division_is_refused_by_name():
    with pytest.raises(ValueError, match="PHASED"):
        sub_image_size(division="PHASED", count=4, **_8K)


def test_an_essence_pixel_outside_the_raster_is_refused():
    with pytest.raises(ValueError, match="outside"):
        to_constituent(7680, 0, division=MULTI_SD, count=4, **_8K)


def test_a_constituent_pixel_outside_its_sub_image_is_refused():
    with pytest.raises(ValueError, match="outside"):
        to_essence(1, 3840, 0, division=MULTI_SD, count=4, **_8K)


def test_a_constituent_the_group_does_not_have_is_refused():
    with pytest.raises(ValueError, match="constituent"):
        to_essence(5, 0, 0, division=MULTI_2SI, count=4, **_8K)


# The two properties, over every pixel of rasters small enough to enumerate:
# the smallest each mode and count admits, and one a few times larger.
_RASTERS = [(8, 4), (24, 12), (40, 16)]


@pytest.mark.parametrize(("division", "count"), _MODES)
@pytest.mark.parametrize(("width", "height"), _RASTERS)
def test_every_essence_pixel_lands_in_exactly_one_constituent_pixel(
    division, count, width, height
):
    ys, xs = np.mgrid[0:height, 0:width]
    constituent, sub_x, sub_y = to_constituent(
        xs, ys, division=division, count=count, width=width, height=height
    )
    sub_width, sub_height = sub_image_size(
        width, height, division=division, count=count
    )
    landed = set(zip(constituent.ravel(), sub_x.ravel(), sub_y.ravel(), strict=True))
    every = set(
        itertools.product(range(1, count + 1), range(sub_width), range(sub_height))
    )
    assert landed == every
    assert len(landed) == width * height


@pytest.mark.parametrize(("division", "count"), _MODES)
@pytest.mark.parametrize(("width", "height"), _RASTERS)
def test_the_inverse_mapping_returns_every_pixel_to_where_it_was(
    division, count, width, height
):
    ys, xs = np.mgrid[0:height, 0:width]
    raster = {"division": division, "count": count, "width": width, "height": height}
    back_x, back_y = to_essence(*to_constituent(xs, ys, **raster), **raster)
    np.testing.assert_array_equal(back_x, xs)
    np.testing.assert_array_equal(back_y, ys)


def _labelled(width: int, height: int) -> np.ndarray:
    """An essence whose every element is ten times its row plus its column."""
    ys, xs = np.mgrid[0:height, 0:width]
    return 10 * ys + xs


def test_four_way_interleave_splits_a_small_essence_as_counted_by_hand():
    """Rows 0 and 2 to constituents 1 and 2, rows 1 and 3 to 3 and 4; along
    a row, pairs 0-1 and 4-5 to the first of each, 2-3 and 6-7 to the
    second."""
    constituents = split(_labelled(8, 4), division=MULTI_2SI, count=4)
    assert [part.tolist() for part in constituents] == [
        [[0, 1, 4, 5], [20, 21, 24, 25]],
        [[2, 3, 6, 7], [22, 23, 26, 27]],
        [[10, 11, 14, 15], [30, 31, 34, 35]],
        [[12, 13, 16, 17], [32, 33, 36, 37]],
    ]


def test_four_way_square_division_splits_a_small_essence_into_quadrants():
    constituents = split(_labelled(4, 2), division=MULTI_SD, count=4)
    assert [part.tolist() for part in constituents] == [
        [[0, 1]],
        [[2, 3]],
        [[10, 11]],
        [[12, 13]],
    ]


@pytest.mark.parametrize(("division", "count"), _MODES)
@pytest.mark.parametrize(("width", "height"), _RASTERS)
def test_split_puts_each_element_where_the_mapping_says(division, count, width, height):
    essence = _labelled(width, height)
    constituents = split(essence, division=division, count=count)
    ys, xs = np.mgrid[0:height, 0:width]
    constituent, sub_x, sub_y = to_constituent(
        xs, ys, division=division, count=count, width=width, height=height
    )
    stacked = np.stack(constituents)
    np.testing.assert_array_equal(stacked[constituent - 1, sub_y, sub_x], essence)


@pytest.mark.parametrize(("division", "count"), _MODES)
@pytest.mark.parametrize(("width", "height"), _RASTERS)
def test_reassembling_the_split_returns_the_essence_unchanged(
    division, count, width, height
):
    essence = _labelled(width, height)
    constituents = split(essence, division=division, count=count)
    np.testing.assert_array_equal(reassemble(constituents, division=division), essence)


@pytest.mark.parametrize(("division", "count"), _MODES)
def test_split_moves_whole_elements_of_any_dtype_and_trailing_shape(division, count):
    """Three samples a pixel in a structured dtype: nothing is read, so
    nothing about the element matters."""
    rng = np.random.default_rng(2110)
    element = np.dtype([("luma", "<u2"), ("chroma", "<u2", (2,))])
    essence = np.zeros((16, 24, 3), dtype=element)
    essence["luma"] = rng.integers(0, 1024, essence.shape)
    essence["chroma"] = rng.integers(0, 1024, (*essence.shape, 2))
    constituents = split(essence, division=division, count=count)
    assert all(part.dtype == element for part in constituents)
    assert all(part.shape[2:] == (3,) for part in constituents)
    np.testing.assert_array_equal(reassemble(constituents, division=division), essence)


def test_reassembly_refuses_constituents_of_unequal_shape():
    parts = split(_labelled(8, 4), division=MULTI_2SI, count=4)
    parts[3] = parts[3][:, :2]
    with pytest.raises(ValueError, match="shape"):
        reassemble(parts, division=MULTI_2SI)


def test_reassembly_refuses_a_count_the_rp_does_not_define():
    parts = split(_labelled(8, 4), division=MULTI_2SI, count=4)
    with pytest.raises(ValueError, match="4 or 16"):
        reassemble(parts[:3], division=MULTI_2SI)


def test_split_refuses_an_essence_with_no_raster_axes():
    with pytest.raises(ValueError, match="rows and columns"):
        split(np.zeros(8), division=MULTI_2SI, count=4)
