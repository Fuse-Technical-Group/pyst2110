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
from collections.abc import Callable
from dataclasses import fields
from fractions import Fraction
from typing import Any

import numpy as np

from pyst2110 import geometry, payload, rtp
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

# The raster a header-data split is measured over: 1080p60 YCbCr-4:2:2 10-bit,
# which is what a Matrox ConvertIP was sending when a consumer found a quarter
# of every frame missing (§spec:split-segments).
SPLIT_RASTER = SdpVideo(
    width=1920,
    height=1080,
    frame_rate=Fraction(60, 1),
    depth=10,
    sampling="YCbCr-4:2:2",
    max_udp=1460,
)

#: Octets a segment, against SPLIT_RASTER's 4800-octet line. 1200 divides it,
#: so every packet carries one SRD header and nothing crosses the cut; 1320 is
#: the ConvertIP's own, and 4800 = 3 x 1320 + 840 puts two segments in every
#: fourth packet.
TILED_SEGMENT = 1200
STRADDLING_SEGMENT = 1320

#: Octets of header a split receiver keeps: the RFC 3550 fixed header, the
#: extended sequence number, and the one SRD header the cut clears.
SPLIT_CUT = 20
_SRD_SIZE = 6

_NS_PER_MS = 1_000_000
_US_PER_S = 1_000_000
_NS_PER_S = 1_000_000_000
_MS_PER_S = 1_000


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


def split_chunk(count: int, segment: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A chunk as a header-data-split receiver holds it, at ``segment`` octets.

    Two buffers: twenty octets of header a packet, and the payload row that
    follows the cut. Where ``segment`` does not divide the line, a packet spans
    two rows and its second SRD header lands at the head of its payload row —
    which is the shape this benchmark exists to time (§spec:split-segments).

    Written field by field like :func:`conforming_chunk`, so the number
    measures the parse rather than a round trip through our own builder.
    """
    octets, pixels = geometry.pgroup(SPLIT_RASTER)
    line = geometry.line_bytes(SPLIT_RASTER)
    heads = np.zeros((count, SPLIT_CUT), dtype=np.uint8)
    payloads = np.zeros((count, _SRD_SIZE + segment), dtype=np.uint8)

    position = 0
    for index in range(count):
        remaining = segment
        segments: list[tuple[int, int, int]] = []
        while remaining:
            row, within = divmod(position, line)
            take = min(remaining, line - within)
            segments.append(
                (row % SPLIT_RASTER.height, within // octets * pixels, take)
            )
            position += take
            remaining -= take
        _write_split_packet(heads[index], payloads[index], index, segments)
    return heads, payloads, np.full(count, SPLIT_CUT, dtype=np.int64)


def _write_split_packet(
    head: np.ndarray,
    payload_row: np.ndarray,
    sequence: int,
    segments: list[tuple[int, int, int]],
) -> None:
    """One packet's headers, the first SRD in the header row and the rest past
    the cut, where a fixed-offset split leaves them."""
    head[0] = 0x80  # V=2, P=0, X=0, CC=0
    head[1] = 96  # M=0, PT=96
    _put(head, 2, sequence % (1 << 16))  # RTP sequence number
    _put(head, 8, 0x0A0B)  # SSRC, high half
    _put(head, 10, 0x0C0D)  # SSRC, low half
    _put(head, 12, (sequence // (1 << 16)) % (1 << 16))  # Extended Sequence
    for index, (row, offset, length) in enumerate(segments):
        # The C bit says another SRD header follows.
        carry = 0x0000 if index == len(segments) - 1 else 0x8000
        target, at = (
            (head, 14) if index == 0 else (payload_row, (index - 1) * _SRD_SIZE)
        )
        _put(target, at, length)  # SRD Length
        _put(target, at + 2, row)  # F=0, SRD Row Number
        _put(target, at + 4, offset | carry)  # C, SRD Offset


def _put(row: np.ndarray, octet: int, value: int) -> None:
    """Write a big-endian 16-bit field into one packet's row."""
    row[octet] = (value >> 8) & 0xFF
    row[octet + 1] = value & 0xFF


def odd_strided(block: np.ndarray) -> np.ndarray:
    """The same packets in a chunk no column slice can read.

    One octet wider, so the stride is odd and the parse takes its general
    path — the library's own fallback, rather than a private function rebound
    at runtime. The sizes passed alongside keep the bound what it was, so the
    two paths read the same octets and neither is handed one the other could
    not reach.
    """
    packets, stride = block.shape
    wide = np.zeros((packets, stride + 1), dtype=block.dtype)
    wide[:, :stride] = block
    return wide


def agree(block: np.ndarray, sizes: np.ndarray) -> bool:
    """Whether the two paths read this chunk identically, field for field.

    A number measured over a chunk the two paths disagree about measures
    nothing, so the tool says so on the spot. `tests/test_path_equality.py`
    is the gate that says it over every shape.
    """
    fast = _parse(block, sizes)
    general = _parse(odd_strided(block), sizes)
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
        f"{seconds / count * per_frame * _MS_PER_S:>11.3f}"
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

    # The same packets in a chunk the column read declines, so the general
    # path is reached the way a caller reaches it rather than by rebinding
    # anything (:func:`odd_strided`).
    wide = odd_strided(block)
    offsets = rtp.parse_rtp(block, sizes=sizes).payload_offset

    def parse() -> tuple[object, object]:
        return _parse(block, sizes)

    def parse_general() -> tuple[object, object]:
        return _parse(wide, sizes)

    def read_rtp() -> object:
        return rtp.parse_rtp(block, sizes=sizes)

    def read_rtp_general() -> object:
        return rtp.parse_rtp(wide, sizes=sizes)

    def read_payload() -> object:
        return payload.parse_payload_headers(block, offsets, sizes=sizes)

    def read_payload_general() -> object:
        return payload.parse_payload_headers(wide, offsets, sizes=sizes)

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
    for label, fast_work, general_work in (
        ("parse_rtp", read_rtp, read_rtp_general),
        ("parse_payload_headers", read_payload, read_payload_general),
        ("both", parse, parse_general),
    ):
        timings[label + ", general"] = fastest(general_work, args.repeat, args.loops)
        timings[label + ", fast"] = fastest(fast_work, args.repeat, args.loops)
    timings["both + SequenceTracker.observe"] = fastest(track, args.repeat, args.loops)
    for name, seconds in timings.items():
        report(name, seconds, args.packets, per_frame)

    speedup = timings["both, general"] / timings["both, fast"]
    print(f"\n  fast path is {speedup:.1f}x the general path over the same chunk")
    split(args)
    return 0


def split(args: argparse.Namespace) -> None:
    """What crossing a header-data split costs, and what declining to costs.

    Two chunks of the same raster: one whose segment divides the line, so no
    packet crosses the cut, and one at the ConvertIP's 1320 octets, where a
    quarter of them do. Offering the payload buffer to the first shall not
    move its number — the selection reads the first segment's continuation bit
    on the header buffer and finds nothing (§spec:split-segments).
    """
    line = geometry.line_bytes(SPLIT_RASTER)
    print(
        f"\nheader-data split — raster {SPLIT_RASTER.width}x{SPLIT_RASTER.height} "
        f"{float(SPLIT_RASTER.frame_rate):.2f} fps, {line} octets a line, "
        f"cut at {SPLIT_CUT}\n"
    )
    print(f"  {'':<34}{'us/chunk':>10}{'ns/packet':>12}{'ms/frame':>11}")

    for label, segment in (
        ("tiled", TILED_SEGMENT),
        ("straddling", STRADDLING_SEGMENT),
    ):
        heads, payloads, sizes = split_chunk(args.packets, segment)
        offsets = rtp.parse_rtp(heads, sizes=sizes).payload_offset
        per_frame = -(-(SPLIT_RASTER.height * line) // segment)
        parsed = payload.parse_payload_headers(
            heads, offsets, sizes=sizes, payloads=payloads
        )
        crossed = int(np.count_nonzero(parsed.segments > 1))

        def withheld(heads=heads, offsets=offsets, sizes=sizes) -> object:
            return payload.parse_payload_headers(heads, offsets, sizes=sizes)

        def offered(
            heads=heads, offsets=offsets, sizes=sizes, payloads=payloads
        ) -> object:
            return payload.parse_payload_headers(
                heads, offsets, sizes=sizes, payloads=payloads
            )

        print(
            f"  {label} at {segment} octets a segment: "
            f"{crossed} of {args.packets} packets cross the cut"
        )
        for shown, work in (("withheld", withheld), ("payloads offered", offered)):
            seconds = fastest(work, args.repeat, args.loops)
            report(f"  {shown}", seconds, args.packets, per_frame)


if __name__ == "__main__":
    raise SystemExit(main())
