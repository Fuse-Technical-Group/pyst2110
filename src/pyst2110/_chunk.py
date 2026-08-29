"""Shape checks and per-packet reads shared by the header parses.

Internal. Both parses take the same thing — a ``(packets, stride)`` uint8 view
(SPEC §spec:interface-shape) — and both need to read a big-endian field at an
offset that differs per packet, because CSRCs, a header extension and the
RFC 4175 continuation bit all move what follows them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

_U16_SIZE = 2


def per_packet(
    values: ArrayLike,
    noun: str,
    count: int | None = None,
    plural: str = "",
    dtype: DTypeLike = np.int64,
) -> NDArray[Any]:
    """One ``noun`` per packet, coerced — or a ValueError naming the field.

    Every array crossing this interface is packet-aligned
    (§spec:interface-shape), so every one of them is checked the same way.
    Pass ``count`` where the packet count is already known, which reports both
    numbers; without it the check is that the input is one-dimensional at all,
    which is what catches a ``(packets, stride)`` view passed by mistake.

    ``plural`` names ``noun`` in the plural where adding an *s* does not.
    """
    array = np.asarray(values, dtype=dtype)
    if count is None:
        if array.ndim != 1:
            raise ValueError(f"expected one {noun} per packet, got shape {array.shape}")
    elif array.shape != (count,):
        raise ValueError(
            f"expected one {noun} per packet: {count} packets, "
            f"{array.size} {plural or noun + 's'}"
        )
    return array


def validate(packets: NDArray[np.uint8], minimum: int) -> None:
    """Reject anything that is not a byte chunk of at least ``minimum`` columns."""
    if packets.ndim != 2:
        raise ValueError(
            f"expected a (packets, stride) byte view, got shape {packets.shape}"
        )
    if packets.dtype != np.uint8:
        raise ValueError(
            f"expected a uint8 byte view, got dtype {packets.dtype}; a wider "
            "dtype would misalign every field"
        )
    if packets.shape[1] < minimum:
        raise ValueError(
            f"expected at least {minimum} columns, got shape {packets.shape}"
        )


def limits(
    packets: NDArray[np.uint8], sizes: NDArray[np.integer[Any]] | None
) -> NDArray[np.int64]:
    """Each packet's usable length: its reported size, or the stride if unstated.

    A receiver writes at most the stride and reports the true size separately,
    so the columns past a packet still hold whatever the buffer held before
    (§spec:interface-shape). Deriving a field's position from those would read
    the previous frame's packet.
    """
    count, stride = packets.shape
    if sizes is None:
        return np.full(count, stride, dtype=np.int64)
    bounds: NDArray[np.int64] = per_packet(sizes, "size", count=count)
    return np.clip(bounds, 0, stride)


def read_u16(
    packets: NDArray[np.uint8],
    offsets: NDArray[np.int64],
    bounds: NDArray[np.int64],
    rows: NDArray[np.int64] | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """A big-endian 16-bit field read at a per-packet offset.

    Returns the values and a mask of which reads landed inside their packet.
    A read that does not fit is clamped to a valid index and its value zeroed
    rather than raising: one malformed packet in a chunk of thousands is a
    packet to account for, not a reason to drop the chunk.

    ``rows`` is the packet index the gather reads down, ``arange(count)``
    where a caller does not pass one. A caller reading several fields of the
    same chunk passes it once rather than paying an allocation per field:
    :func:`pyst2110.payload.parse_payload_headers` reads nine, and at a
    thousand chunks a second that is nine arange allocations per chunk on the
    frame path.
    """
    inside = (offsets >= 0) & (offsets + _U16_SIZE <= bounds)
    index = np.where(inside, offsets, 0).clip(0, packets.shape[1] - _U16_SIZE)
    if rows is None:
        rows = np.arange(packets.shape[0])
    high = packets[rows, index].astype(np.int64)
    low = packets[rows, index + 1].astype(np.int64)
    return np.where(inside, (high << 8) | low, 0).astype(np.int64), inside


def u16_view(packets: NDArray[np.uint8]) -> NDArray[np.uint16] | None:
    """The whole chunk as big-endian 16-bit words, or ``None`` if it cannot be.

    Reinterpreting the buffer is what turns a header field into a column
    slice, and a column slice is what the conforming fast path is made of
    (§spec:conforming-fast-path). It needs the rows contiguous and an even
    number of octets in each; a caller handing over an odd stride, or a
    sub-block sliced out of a wider buffer, gets ``None`` and the general path
    rather than an exception. Both are legitimate chunks — they are simply not
    ones a column slice can read.
    """
    if packets.shape[1] % _U16_SIZE != 0 or not packets.flags["C_CONTIGUOUS"]:
        return None
    view: NDArray[np.uint16] = packets.view(">u2")
    return view
