"""Where an essence pixel sits when RP 2110-23 divides it over several flows.

SMPTE RP 2110-23 carries one picture as four or sixteen ST 2110-20 flows,
each a sub-image of the essence (SPEC §spec:multi-stream). This maps an
essence pixel to the constituent that carries it and its pixel within that
sub-image, and back, for two-sample interleave (``MULTI-2SI``, section 5.2.3)
and square division (``MULTI-SD``, section 5.2.4).

The mapping is closed form and vectorised over coordinate arrays, so a GPU
kernel and this reference compute it from the same arithmetic. It says where
a sample goes; moving it is the consumer's, as with the raster placement of
:mod:`pyst2110.geometry` (§spec:scope-boundary). :func:`split` and
:func:`reassemble` are the reference move, kept as the oracle an accelerated
gather is checked against.

Constituents are numbered from 1, as the RP numbers them and as an SDP
group's tags name them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "CONSTITUENT_COUNTS",
    "DIVISIONS",
    "MULTI_2SI",
    "MULTI_SD",
    "reassemble",
    "split",
    "sub_image_size",
    "to_constituent",
    "to_essence",
]

#: Two-sample interleave, RP 2110-23 section 5.2.3: rows alternate between
#: two pairs of constituents, and along a row each pixel pair goes to the
#: next constituent of its pair. The token is the RP's ``a=group:`` semantics.
MULTI_2SI = "MULTI-2SI"

#: Square division, RP 2110-23 section 5.2.4: the essence cut into quadrants.
#: The RP deprecates it for new designs and keeps it for signals already
#: built that way.
MULTI_SD = "MULTI-SD"

#: Every division this library maps. The RP's third, ``PHASED``, divides time
#: and not space, and is refused by name (§spec:multi-stream).
DIVISIONS = (MULTI_2SI, MULTI_SD)

#: The constituent counts RP 2110-23 defines for these divisions: one
#: division into four, or that division applied again to each of the four.
CONSTITUENT_COUNTS = (4, 16)

_PHASED = "PHASED"

# One four-way step halves each axis, so a constituent of a four-way group is
# a 2x2 tile of the essence's area and a sixteen-way one a 4x4.
_QUARTERS = 4
_HALF = 2
# Two-sample interleave deals whole pixel pairs: each group of four pixels
# along a row is a pair for each of two constituents.
_PAIR = 2
_PAIRS_PER_GROUP = 2

Coordinates = NDArray[np.int64]


def sub_image_size(
    width: int, height: int, *, division: str, count: int
) -> tuple[int, int]:
    """Returns the width and height of each constituent's sub-image.

    Raises ``ValueError`` for a division or a count RP 2110-23 does not
    define for a spatial division, and for an essence the division cannot cut
    into equal sub-images: four-way two-sample interleave deals pixel pairs
    over two rows, so it needs a width divisible by four and an even height,
    and square division needs equal quadrants. Sixteen-way applies the
    four-way requirement again to the quarter.
    """
    _check_mode(division, count)
    steps = _steps(count)
    for _ in range(steps):
        _check_step(width, height, division)
        width, height = width // _HALF, height // _HALF
    return width, height


def to_constituent(
    x: ArrayLike,
    y: ArrayLike,
    *,
    width: int,
    height: int,
    division: str,
    count: int,
) -> tuple[Coordinates, Coordinates, Coordinates]:
    """Returns the constituent, and the pixel within its sub-image, of each
    essence pixel ``(x, y)``.

    ``width`` and ``height`` are the essence's. The three arrays broadcast
    ``x`` against ``y``, and the constituent runs from 1 to ``count``. For
    four-way two-sample interleave, as RP 2110-23 Figure 1 draws it, the
    constituent is ``1 + 2·(y mod 2) + ((x div 2) mod 2)`` and the sub-image
    pixel ``(2·(x div 4) + x mod 2, y div 2)``; for square division it is
    the quadrant and the offset from the quadrant's origin. Sixteen-way
    applies the four-way mapping to the essence and again to the
    intermediate sub-image, numbering intermediate ``i`` and constituent
    ``j`` within it ``4·(i - 1) + j``, which is Figure 2's and Figure 5's
    numbering.

    Raises ``ValueError`` as :func:`sub_image_size` does, and for a pixel
    outside the essence.
    """
    sub_image_size(width, height, division=division, count=count)
    xs, ys = np.broadcast_arrays(np.asarray(x, np.int64), np.asarray(y, np.int64))
    _check_range(xs, width, "essence x")
    _check_range(ys, height, "essence y")

    constituent = np.zeros(xs.shape, np.int64)
    for _ in range(_steps(count)):
        quarter, xs, ys = _quarter(xs, ys, width, height, division)
        constituent = constituent * _QUARTERS + quarter
        width, height = width // _HALF, height // _HALF
    return constituent + 1, xs, ys


def to_essence(
    constituent: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    *,
    width: int,
    height: int,
    division: str,
    count: int,
) -> tuple[Coordinates, Coordinates]:
    """Returns the essence pixel that constituent ``constituent`` carries at
    sub-image pixel ``(x, y)``: the inverse of :func:`to_constituent`.

    ``width`` and ``height`` are the essence's, as for :func:`to_constituent`.
    Raises ``ValueError`` as :func:`sub_image_size` does, for a constituent
    outside 1 to ``count``, and for a pixel outside the sub-image.
    """
    sub_width, sub_height = sub_image_size(
        width, height, division=division, count=count
    )
    numbers, xs, ys = np.broadcast_arrays(
        np.asarray(constituent, np.int64),
        np.asarray(x, np.int64),
        np.asarray(y, np.int64),
    )
    if numbers.size and (numbers.min() < 1 or numbers.max() > count):
        raise ValueError(f"a {count}-way group has no constituent outside 1-{count}")
    _check_range(xs, sub_width, "sub-image x")
    _check_range(ys, sub_height, "sub-image y")

    # Undo the last step first: its quarter is the low base-4 digit of the
    # zero-based constituent number, applied over the smallest raster.
    steps = _steps(count)
    remaining = numbers - 1
    for step in reversed(range(steps)):
        step_width, step_height = width >> step, height >> step
        quarter = remaining % _QUARTERS
        remaining = remaining // _QUARTERS
        xs, ys = _unquarter(quarter, xs, ys, step_width, step_height, division)
    return xs, ys


def split(
    essence: ArrayLike, *, division: str, count: int
) -> list[NDArray[np.generic]]:
    """Returns each constituent's sub-image of an essence array, in
    constituent order.

    ``essence`` is ``(height, width, …)``: the first two axes are the raster
    and any further axes ride along, so a pixel may be several samples. The
    elements are moved and never read, so any dtype works. Raises
    ``ValueError`` as :func:`sub_image_size` does.
    """
    array = np.asarray(essence)
    if array.ndim < _HALF:
        raise ValueError("an essence array needs rows and columns as its first axes")
    height, width = array.shape[:2]
    return [
        array[np.ix_(*_essence_lines(number, width, height, division, count))]
        for number in range(1, count + 1)
    ]


def reassemble(
    constituents: Sequence[ArrayLike], *, division: str
) -> NDArray[np.generic]:
    """Returns the essence array that :func:`split` cut into these sub-images.

    The count is the number of constituents, which shall be 4 or 16, and each
    is ``(sub_height, sub_width, …)`` with one shape and dtype for all of
    them. Raises ``ValueError`` otherwise, and as :func:`sub_image_size` does.
    """
    parts = [np.asarray(part) for part in constituents]
    count = len(parts)
    _check_mode(division, count)
    shapes = {part.shape for part in parts}
    if len(shapes) != 1:
        raise ValueError(
            f"the constituents are of shapes {sorted(shapes)}; a division "
            f"assembles one essence only from sub-images of one shape"
        )
    (shape,) = shapes
    if len(shape) < _HALF:
        raise ValueError("a sub-image array needs rows and columns as its first axes")
    side = _HALF ** _steps(count)
    height, width = shape[0] * side, shape[1] * side
    essence = np.empty((height, width, *shape[2:]), dtype=parts[0].dtype)
    for number, part in enumerate(parts, start=1):
        essence[np.ix_(*_essence_lines(number, width, height, division, count))] = part
    return essence


def _essence_lines(
    number: int, width: int, height: int, division: str, count: int
) -> tuple[Coordinates, Coordinates]:
    """The essence rows and columns constituent ``number`` carries, in order.

    Both divisions deal rows and columns independently — a sub-image row is
    one essence row and a sub-image column one essence column — so the two
    lists index the constituent's whole sub-image as an open mesh.
    """
    sub_width, sub_height = sub_image_size(
        width, height, division=division, count=count
    )
    columns, _ = to_essence(
        number,
        np.arange(sub_width),
        0,
        width=width,
        height=height,
        division=division,
        count=count,
    )
    _, rows = to_essence(
        number,
        0,
        np.arange(sub_height),
        width=width,
        height=height,
        division=division,
        count=count,
    )
    return rows, columns


def _quarter(
    xs: Coordinates, ys: Coordinates, width: int, height: int, division: str
) -> tuple[Coordinates, Coordinates, Coordinates]:
    """One four-way step: the zero-based quarter and the pixel within it."""
    if division == MULTI_2SI:
        quarter = _HALF * (ys % _HALF) + (xs // _PAIR) % _PAIRS_PER_GROUP
        group_width = _PAIR * _PAIRS_PER_GROUP
        return quarter, _PAIR * (xs // group_width) + xs % _PAIR, ys // _HALF
    half_width, half_height = width // _HALF, height // _HALF
    column, row = xs // half_width, ys // half_height
    return _HALF * row + column, xs - column * half_width, ys - row * half_height


def _unquarter(
    quarter: Coordinates,
    xs: Coordinates,
    ys: Coordinates,
    width: int,
    height: int,
    division: str,
) -> tuple[Coordinates, Coordinates]:
    """The inverse of :func:`_quarter`, over a raster ``width`` by ``height``."""
    row, column = quarter // _HALF, quarter % _HALF
    if division == MULTI_2SI:
        group_width = _PAIR * _PAIRS_PER_GROUP
        essence_x = group_width * (xs // _PAIR) + _PAIR * column + xs % _PAIR
        return essence_x, _HALF * ys + row
    return xs + column * (width // _HALF), ys + row * (height // _HALF)


def _steps(count: int) -> int:
    """How many four-way divisions make ``count`` constituents."""
    return 1 if count == CONSTITUENT_COUNTS[0] else _HALF


def _check_mode(division: str, count: int) -> None:
    if division == _PHASED:
        raise ValueError(
            "PHASED divides an essence in time, carrying each phase with its "
            "own RTP timestamps, and this library models only the spatial "
            f"divisions {list(DIVISIONS)}"
        )
    if division not in DIVISIONS:
        raise ValueError(
            f"a division of {division!r} is not one of the {list(DIVISIONS)} "
            f"RP 2110-23 defines"
        )
    if count not in CONSTITUENT_COUNTS:
        raise ValueError(
            f"a group of {count!r} constituents is not one RP 2110-23 defines; "
            f"{division} divides into 4 or 16"
        )


def _check_step(width: int, height: int, division: str) -> None:
    """Refuse a raster that one four-way step cannot cut into equal parts."""
    column_unit = _PAIR * _PAIRS_PER_GROUP if division == MULTI_2SI else _HALF
    if width <= 0 or height <= 0 or width % column_unit or height % _HALF:
        raise ValueError(
            f"{division} cannot divide a {width}x{height} raster into four "
            f"equal sub-images: it needs a width divisible by {column_unit} "
            f"and an even height"
        )


def _check_range(values: Coordinates, limit: int, name: str) -> None:
    if values.size and (values.min() < 0 or values.max() >= limit):
        raise ValueError(f"the {name} coordinates run outside 0-{limit - 1}")
