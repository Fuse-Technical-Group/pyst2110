#!/usr/bin/env python3
"""Time the conforming fast path against the general one (§road:parse-benchmark).

SPEC §spec:conforming-fast-path quotes a per-packet cost measured on one host.
A figure nobody else can reproduce is a claim, so this builds its own chunk of
conforming ST 2110-20 headers, parses it both ways through the public entry
points, and prints what each costs — per packet, and as a share of the frame
period at a named raster.

    uv run tools/parse-benchmark.py [--packets N] [--repeat N]

Not part of the suite: it measures a host rather than asserting anything about
the library, and a timing threshold in CI is a flake generator (§spec:testing).
Equality between the two paths is `tests/test_path_equality.py`'s, and that one
is a gate.
"""

from __future__ import annotations

import argparse
import timeit
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import fields
from fractions import Fraction
from typing import Any

import numpy as np

from pyst2110 import _chunk, geometry, payload, rtp
from pyst2110.framing import SequenceTracker
from pyst2110.sdp import SdpVideo

# The raster the measurement is quoted at: 2160p59.94 YCbCr-4:2:2 10-bit at
# the standard datagram limit, which is what a consumer of this library was
# running when the cost of the general path became visible.
RASTER = SdpVideo(
    width=3840,
    height=2160,
    frame_rate=Fraction(60000, 1001),
    depth=10,
    sampling="YCbCr-4:2:2",
    max_udp=1460,
)

_NS_PER_MS = 1_000_000
_US_PER_S = 1_000_000
_NS_PER_S = 1_000_000_000


def conforming_chunk(count: int, payload_size: int, per_frame: int) -> np.ndarray:
    """A chunk of header blocks a conforming ST 2110-20 sender would emit.

    Twenty octets a packet — the RFC 3550 fixed header with no CSRC list and
    no extension, two octets of extended sequence number, one SRD header —
    which is what a header-data-split receiver hands to a parse. Written field
    by field from RFC 3550 section 5.1 and RFC 4175 section 4.2 rather than
    through this library's own builder, so the benchmark measures the parse
    and nothing else.
    """
    per_line = geometry.packets_per_line(RASTER, payload_size)
    _, group_pixels = geometry.pgroup(RASTER)
    index = np.arange(count, dtype=np.int64)
    block = np.zeros((count, 20), dtype=np.uint8)

    sequence = index % (1 << 16)
    extended = (index // (1 << 16)) % (1 << 16)
    line = (index // per_line) % RASTER.height
    offset = (index % per_line) * (payload_size // 5) * group_pixels

    block[:, 0] = 0x80  # V=2, P=0, X=0, CC=0
    # The marker lands on a frame's last packet, as ST 2110-20 6.1.2 puts it.
    block[:, 1] = 96 | np.where((index + 1) % per_frame == 0, 0x80, 0)
    _put_u16(block, 2, sequence)  # RTP sequence number
    _put_u16(block, 8, np.full(count, 0x0A0B))  # SSRC, high half
    _put_u16(block, 10, np.full(count, 0x0C0D))  # SSRC, low half
    _put_u16(block, 12, extended)  # RFC 4175 Extended Sequence Number
    _put_u16(block, 14, np.full(count, payload_size))  # SRD Length
    _put_u16(block, 16, line)  # F=0, SRD Row Number
    _put_u16(block, 18, offset)  # C=0, SRD Offset
    return block


def _put_u16(block: np.ndarray, octet: int, values: np.ndarray) -> None:
    """Write a big-endian 16-bit field into every packet's column."""
    block[:, octet] = (values >> 8) & 0xFF
    block[:, octet + 1] = values & 0xFF


@contextmanager
def general_path() -> Iterator[None]:
    """Force both parses down the general path for the duration.

    Withholding the 16-bit view is the one lever: neither fast path runs
    without it, and both entry points stay exactly the ones a caller uses.
    """
    original = _chunk.u16_view
    _chunk.u16_view = lambda packets: None  # type: ignore[assignment]
    try:
        yield
    finally:
        _chunk.u16_view = original  # type: ignore[assignment]


def agree(block: np.ndarray, sizes: np.ndarray) -> bool:
    """Whether the two paths read this chunk identically, field for field.

    A number measured over a chunk the two paths disagree about measures
    nothing, so the tool says so on the spot. `tests/test_path_equality.py`
    is the gate that says it over every shape.
    """
    fast = _parse(block, sizes)
    with general_path():
        general = _parse(block, sizes)
    return all(
        getattr(mine, entry.name).dtype == getattr(theirs, entry.name).dtype
        and np.array_equal(getattr(mine, entry.name), getattr(theirs, entry.name))
        for mine, theirs in zip(fast, general, strict=True)
        for entry in fields(mine)
    )


def _parse(block: np.ndarray, sizes: np.ndarray) -> tuple[Any, Any]:
    """One chunk through both public entry points, the way a consumer calls."""
    headers = rtp.parse_rtp(block, sizes=sizes)
    return headers, payload.parse_payload_headers(
        block, headers.payload_offset, sizes=sizes
    )


def fastest(work: Callable[[], object], repeat: int, loops: int) -> float:
    """Seconds per call, best of ``repeat`` — the least-disturbed run."""
    return min(timeit.repeat(work, repeat=repeat, number=loops)) / loops


def report(name: str, seconds: float, count: int, per_frame: int) -> None:
    """One row: what a chunk costs, what a packet costs, what a frame costs."""
    print(
        f"  {name:<34}"
        f"{seconds * _US_PER_S:>10.1f}"
        f"{seconds / count * _NS_PER_S:>12.1f}"
        f"{seconds / count * per_frame * 1000:>11.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, default=4096, help="packets a chunk")
    parser.add_argument("--repeat", type=int, default=7, help="timing rounds")
    parser.add_argument("--loops", type=int, default=50, help="calls a round")
    args = parser.parse_args()

    size = geometry.choose_payload_size(RASTER, RASTER.max_udp)
    per_frame = geometry.packets_per_frame(RASTER, size)
    block = conforming_chunk(args.packets, size, per_frame)
    sizes = np.full(args.packets, block.shape[1], dtype=np.int64)

    def parse() -> tuple[object, object]:
        headers = rtp.parse_rtp(block, sizes=sizes)
        return headers, payload.parse_payload_headers(
            block, headers.payload_offset, sizes=sizes
        )

    offsets = rtp.parse_rtp(block, sizes=sizes).payload_offset

    def read_rtp() -> object:
        return rtp.parse_rtp(block, sizes=sizes)

    def read_payload() -> object:
        return payload.parse_payload_headers(block, offsets, sizes=sizes)

    def track() -> object:
        tracker = SequenceTracker()
        headers, descriptors = parse()
        tracker.observe(headers.sequence, extended=descriptors.extended_sequence)
        return tracker

    print(
        f"pyst2110 header parse — {args.packets} conforming packets a chunk, "
        f"{block.shape[1]} octets a header\n"
        f"raster {RASTER.width}x{RASTER.height} "
        f"{float(RASTER.frame_rate):.2f} fps {RASTER.sampling} {RASTER.depth}-bit, "
        f"MAXUDP {RASTER.max_udp} -> payload {size}, {per_frame} packets a frame, "
        f"{RASTER.frame_interval_ns / _NS_PER_MS:.2f} ms a frame\n"
    )
    print(f"  both paths agree over this chunk: {agree(block, sizes)}\n")
    print(f"  {'':<34}{'us/chunk':>10}{'ns/packet':>12}{'ms/frame':>11}")

    timings: dict[str, float] = {}
    for label, work in (
        ("parse_rtp", read_rtp),
        ("parse_payload_headers", read_payload),
        ("both", parse),
    ):
        with general_path():
            timings[label + ", general"] = fastest(work, args.repeat, args.loops)
        timings[label + ", fast"] = fastest(work, args.repeat, args.loops)
    timings["both + SequenceTracker.observe"] = fastest(track, args.repeat, args.loops)
    for name, seconds in timings.items():
        report(name, seconds, args.packets, per_frame)

    speedup = timings["both, general"] / timings["both, fast"]
    print(f"\n  fast path is {speedup:.1f}x the general path over the same chunk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
