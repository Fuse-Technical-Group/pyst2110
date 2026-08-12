"""Where a format's samples sit: pgroups, lines and packet counts.

ST 2110-20 packs samples into pgroups — the smallest whole number of pixels
filling a whole number of octets — and a payload header's offset is a sample
position, so turning one into a byte offset needs the format (§spec:geometry).

Every function here takes an :class:`~pyst2110.sdp.SdpVideo`, which is what an
SDP already yields; the format type is not redefined here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyst2110.sdp import SdpVideo

__all__ = [
    "byte_offset",
    "choose_payload_size",
    "fits_raster",
    "line_bytes",
    "packets_per_frame",
    "packets_per_line",
    "pgroup",
]

# SMPTE ST 2110-20 Table 1 (4:4:4) and Table 2 (4:2:2), as (octets, pixels).
# Both families group samples within a single row, which is what lets a line
# be a whole number of pgroups.
_444_PGROUPS = {8: (3, 1), 10: (15, 4), 12: (9, 2), 16: (6, 1)}
_422_PGROUPS = {8: (4, 2), 10: (5, 2), 12: (6, 2), 16: (8, 2)}
_444_SAMPLINGS = ("YCbCr-4:4:4", "CLYCbCr-4:4:4", "ICtCp-4:4:4", "RGB")
_422_SAMPLINGS = ("YCbCr-4:2:2", "CLYCbCr-4:2:2", "ICtCp-4:2:2")

# Which table a sampling reads from, so a sampling and a depth are looked up
# in that order — the depths a sampling covers are its own.
_FAMILY: dict[str, dict[int, tuple[int, int]]] = dict.fromkeys(
    _444_SAMPLINGS, _444_PGROUPS
) | dict.fromkeys(_422_SAMPLINGS, _422_PGROUPS)
# XYZ is 4:4:4-shaped, but ST 2110-20 Table 1 lists it only at 12 and 16 bits.
_FAMILY["XYZ"] = {depth: _444_PGROUPS[depth] for depth in (12, 16)}

# 4:2:0 is absent on purpose. Its pgroups span two sample rows (ST 2110-20
# Table 3), and ST 2110-20 section 6.1.5 signals only every other row, so a
# line is not a whole number of pgroups and none of the arithmetic below
# holds. Refusing it is better than returning a geometry that is quietly off
# by a factor of two.


def pgroup(video: SdpVideo) -> tuple[int, int]:
    """Octets and pixels in one pgroup of this format."""
    table = _FAMILY.get(video.sampling, {})
    if not table:
        raise ValueError(
            f"no pgroup for sampling={video.sampling!r}; ST 2110-20 Tables 1 "
            f"and 2 cover {sorted(_FAMILY)}"
        )
    if video.depth not in table:
        raise ValueError(
            f"no pgroup for sampling={video.sampling!r} at depth="
            f"{video.depth}; ST 2110-20 lists it at {sorted(table)} bits"
        )
    return table[video.depth]


def line_bytes(video: SdpVideo) -> int:
    """Octets one active sample row occupies."""
    group_bytes, group_pixels = pgroup(video)
    if video.width <= 0:
        raise ValueError(f"a width of {video.width} is not an image")
    if video.width % group_pixels:
        raise ValueError(
            f"a width of {video.width} is not a whole number of "
            f"{group_pixels}-pixel pgroups"
        )
    return video.width // group_pixels * group_bytes


def choose_payload_size(video: SdpVideo, limit: int) -> int:
    """The largest payload at or under ``limit`` that tiles a line exactly.

    A whole number of pgroups, as RFC 4175 requires an SRD length to be, and
    a divisor of the line — so a line the limit does not divide evenly gets
    more, smaller packets rather than a short final one. Any limit at or
    above one pgroup has an answer; below one it raises (§spec:geometry).
    """
    group_bytes, _ = pgroup(video)
    if limit < group_bytes:
        raise ValueError(
            f"a payload of {limit} octets cannot hold one {group_bytes}-octet pgroup"
        )
    groups = line_bytes(video) // group_bytes
    for count in range(min(limit // group_bytes, groups), 0, -1):
        if groups % count == 0:
            return count * group_bytes
    # The search starts at one pgroup per packet or more, and one divides
    # every line, so it is the answer the loop cannot fail to have found.
    return group_bytes


def packets_per_line(video: SdpVideo, payload_size: int) -> int:
    """Packets one sample row takes at this payload size."""
    total = line_bytes(video)
    if payload_size <= 0 or total % payload_size:
        raise ValueError(
            f"a payload of {payload_size} octets does not tile a "
            f"{total}-octet line; see choose_payload_size()"
        )
    return total // payload_size


def packets_per_frame(video: SdpVideo, payload_size: int) -> int:
    """Packets one frame takes at this payload size.

    Interlace does not change the total: the two fields between them carry
    every row of the frame once, so only the field each row is announced in
    differs (ST 2110-20 section 6.1.5).
    """
    return packets_per_line(video, payload_size) * video.height


def fits_raster(
    video: SdpVideo,
    line: NDArray[np.integer[Any]],
    offset: NDArray[np.integer[Any]],
) -> NDArray[np.bool_]:
    """Which descriptors name a place inside this format's raster.

    True where the row is inside the image and the sample position inside the
    row. Descriptor-aligned: one flag per pair, in the shape they came in
    (§spec:interface-shape).

    A payload header's row and offset are fifteen-bit fields the parse cannot
    check, having no format to check them against, so a packet may name row
    32767 of a 1080-row image. The arithmetic that places one is a scaled
    multiply either way, and a consumer's gather is a kernel over device
    memory where a row past the image writes outside the raster instead of
    raising (§spec:scope-boundary). Mask before placing, not after.

    An interlaced flow numbers rows within a field, so the bound there is
    half the frame's height; which field is the F bit's, reported separately
    (ST 2110-20 section 6.1.5).

    Segment lengths are not checked. Under a header-data split the payload
    sits in a buffer this library never sees, so bounding the extent stays
    the consumer's (§spec:payload-header).
    """
    rows = video.height // 2 if video.interlaced else video.height
    positions = np.asarray(line, dtype=np.int64)
    samples = np.asarray(offset, dtype=np.int64)
    within_image = (positions >= 0) & (positions < rows)
    within_row = (samples >= 0) & (samples < video.width)
    fits: NDArray[np.bool_] = within_image & within_row
    return fits


def byte_offset(
    video: SdpVideo, samples: NDArray[np.integer[Any]]
) -> NDArray[np.int64]:
    """Scale SRD sample positions into byte offsets by the format's pgroup.

    ST 2110-20 calls the SRD Offset a Full-Bandwidth Sample Position: it
    counts pixels, one per pixel. A conformant sender starts a packet on a
    pgroup boundary, so the division is exact; a position that is not on one
    floors to the pgroup containing it rather than raising.

    The scaling is unbounded and knows nothing of the row it scales within: a
    position past the image's width yields an offset past the line, and the
    two are indistinguishable here. Mask with :func:`fits_raster` before the
    result indexes anything.

    Array in, array out (§spec:interface-shape).
    """
    group_bytes, group_pixels = pgroup(video)
    positions = np.asarray(samples, dtype=np.int64)
    return (positions // group_pixels * group_bytes).astype(np.int64)
