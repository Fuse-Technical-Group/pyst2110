"""The two parses agree field for field (SPEC §spec:conforming-fast-path).

A conforming chunk is read by column and every other chunk by the general
walk, and which one ran is not visible in the result — that is the promise, and
this is the gate on it. The general path is the reference: where the two
disagree, the fast path is what is wrong.

A gate rather than a benchmark. Nothing here times anything, and nothing here
asserts a speed; the comparison is over values, dtypes and shapes, so it fails
the same way on any host. `tools/parse-benchmark.py` is where the cost is
measured, and it is not part of the suite (§spec:testing).

The vectors are the ones the two parse tests already write by hand from RFC
3550 section 5.1 and RFC 4175 section 4.2, plus the real-sender capture in
`captures/` — whose sequence numbers, outage included, drive a chunk built here
field by field.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import numpy as np
import pytest

from conftest import GeneralPathError, chunk, general_only, watch_paths
from pyst2110 import payload as payload_module
from pyst2110 import rtp as rtp_module
from pyst2110.framing import SequenceTracker
from pyst2110.payload import PayloadHeaders, parse_payload_headers
from pyst2110.rtp import RtpHeaders, parse_rtp

# --- vectors ------------------------------------------------------------------
#
# Twelve octets of RFC 3550 fixed header — V=2, no padding, no extension, no
# CSRCs; marker clear; payload type 96 — then the RFC 4175 payload header.

_RTP = [0x80, 0x60, 0x00, 0x2A, 0x11, 0x22, 0x33, 0x44, 0x0A, 0x0B, 0x0C, 0x0D]

# One SRD: 1200 octets of row 42 from sample 480. The shape a conforming
# ST 2110-20 sender emits, and the one the fast path is for.
_CONFORMING = [
    *_RTP,
    0x00,
    0x07,  # Extended Sequence Number 7
    0x04,
    0xB0,  # SRD Length 1200
    0x00,
    0x2A,  # F=0, SRD Row Number 42
    0x01,
    0xE0,  # C=0, SRD Offset 480
]

# The same packet with the marker set and both flagged fields at the top of
# their fifteen bits, which is where a mask taking the octet whole goes wrong.
_MARKED_WIDE = [
    *_RTP[:1],
    0xE0,  # M=1, PT=96
    *_RTP[2:],
    0xFF,
    0xFF,  # Extended Sequence Number 65535
    0xFF,
    0xFF,  # SRD Length 65535
    0xFF,
    0xFF,  # F=1, SRD Row Number 32767
    0x7F,
    0xFF,  # C=0, SRD Offset 32767
]

# Two SRDs: the C bit on the first says a second follows, and the second
# carries the F bit. Segment-aligned descriptors, so the general path's.
_TWO_SRDS = [
    *_RTP,
    0x00,
    0x02,  # Extended Sequence Number 2
    0x00,
    0x0A,  # SRD Length 10
    0x00,
    0x05,  # F=0, SRD Row Number 5
    0x80,
    0x00,  # C=1, SRD Offset 0
    0x00,
    0x14,  # SRD Length 20
    0x80,
    0x06,  # F=1, SRD Row Number 6
    0x00,
    0x64,  # C=0, SRD Offset 100
]

# Two CSRCs, so the payload begins eight octets later (RFC 3550 §5.1).
_TWO_CSRCS = [
    0x82,  # V=2, P=0, X=0, CC=2
    *_RTP[1:],
    0x11,
    0x11,
    0x11,
    0x11,  # CSRC[0]
    0x22,
    0x22,
    0x22,
    0x22,  # CSRC[1]
    *_CONFORMING[12:],
]

# The X bit set, with RFC 3550 §5.3.1's extension header: two profile octets
# and a length counting the words after them.
_EXTENSION = [
    0x90,  # V=2, P=0, X=1, CC=0
    *_RTP[1:],
    0xBE,
    0xDE,  # extension profile
    0x00,
    0x01,  # extension length: one 32-bit word
    0xDE,
    0xAD,
    0xBE,
    0xEF,  # the word
    *_CONFORMING[12:],
]

# Four SRDs, one past what ST 2110-20 §6.1.4 permits: the third still sets C,
# so a parse bounded at three ends with a continuation outstanding.
_FOUR_SRDS = [
    *_RTP,
    0x00,
    0x00,  # Extended Sequence Number 0
    *([0x00, 0x01, 0x00, 0x00, 0x80, 0x00] * 3),  # three SRDs, C set on each
    0x00,
    0x01,  # SRD Length 1
    0x00,
    0x03,  # F=0, SRD Row Number 3
    0x00,
    0x00,  # C=0, SRD Offset 0
]


# --- the comparison -----------------------------------------------------------


def parse_both(
    rows: np.ndarray,
    sizes: np.ndarray | None = None,
    max_segments: int = 3,
) -> tuple[tuple[RtpHeaders, PayloadHeaders], tuple[RtpHeaders, PayloadHeaders]]:
    """Parse one chunk as the library would, and again with the fast path shut.

    Both go through the public entry points, neither of which takes a path
    argument — the same call twice, with the 16-bit view withheld the second
    time (§spec:conforming-fast-path).
    """
    fast = _parse(rows, sizes, max_segments)
    with pytest.MonkeyPatch.context() as patch:
        general_only(patch)
        general = _parse(rows, sizes, max_segments)
    return fast, general


def _parse(
    rows: np.ndarray, sizes: np.ndarray | None, max_segments: int
) -> tuple[RtpHeaders, PayloadHeaders]:
    headers = parse_rtp(rows, sizes=sizes)
    return headers, parse_payload_headers(
        rows, headers.payload_offset, sizes=sizes, max_segments=max_segments
    )


def assert_identical(fast: object, general: object) -> None:
    """Every field of one parse result against the other's, dtype included.

    A field that agrees in value and differs in width is still a difference a
    caller can trip over — a sixteen-bit array wraps where a sixty-four-bit one
    does not — so the width is compared rather than the values alone.
    """
    for entry in dataclasses.fields(fast):  # type: ignore[arg-type]
        mine = getattr(fast, entry.name)
        theirs = getattr(general, entry.name)
        assert mine.dtype == theirs.dtype, (
            f"{type(fast).__name__}.{entry.name}: {mine.dtype} against {theirs.dtype}"
        )
        assert mine.shape == theirs.shape, f"{entry.name}: shapes differ"
        assert np.array_equal(mine, theirs), f"{entry.name}: values differ"


def assert_paths_agree(
    rows: np.ndarray, sizes: np.ndarray | None = None, max_segments: int = 3
) -> tuple[RtpHeaders, PayloadHeaders]:
    """Both parses over one chunk, compared field for field. Returns the fast."""
    fast, general = parse_both(rows, sizes, max_segments)
    for mine, theirs in zip(fast, general, strict=True):
        assert_identical(mine, theirs)
    return fast


def test_the_conforming_vectors_really_take_the_fast_path(monkeypatch):
    """Without this the gate is vacuous: two general parses agree trivially.

    Both parses are made to announce their general path, so a conforming chunk
    that returns at all is one the columns read.
    """
    watch_paths(monkeypatch, rtp_module, payload_module)
    rows = chunk(_CONFORMING, _MARKED_WIDE, _CONFORMING)
    headers = parse_rtp(rows)
    assert parse_payload_headers(rows, headers.payload_offset).line.tolist() == [
        42,
        32767,
        42,
    ]

    with pytest.raises(GeneralPathError):
        parse_rtp(chunk(_TWO_CSRCS))


# --- over the hand-written vectors --------------------------------------------


def test_a_conforming_chunk_reads_the_same_both_ways():
    result = assert_paths_agree(chunk(_CONFORMING, _MARKED_WIDE, _CONFORMING))
    # Not vacuous: the chunk really does carry what it claims to.
    assert result[1].line.tolist() == [42, 32767, 42]
    assert result[0].marker.tolist() == [False, True, False]


def test_the_flagged_fields_at_the_top_of_their_range_agree():
    """Where a mask taking the word whole reports a row 32768 too many, and a
    continuation bit where none was set."""
    result = assert_paths_agree(chunk(_MARKED_WIDE))
    assert result[1].line.tolist() == [32767]
    assert result[1].offset_samples.tolist() == [32767]
    assert result[1].field.tolist() == [True]
    assert result[1].extended_sequence.tolist() == [65535]


def test_a_csrc_list_agrees():
    result = assert_paths_agree(chunk(_TWO_CSRCS))
    assert result[0].payload_offset.tolist() == [20]
    assert result[1].line.tolist() == [42]


def test_an_rtp_extension_agrees():
    result = assert_paths_agree(chunk(_EXTENSION))
    # 12 fixed, then the extension's own four octets and its one word.
    assert result[0].payload_offset.tolist() == [20]
    assert result[1].line.tolist() == [42]


def test_two_segments_agree():
    result = assert_paths_agree(chunk(_TWO_SRDS))
    assert result[1].segments.tolist() == [2]
    assert result[1].line.tolist() == [5, 6]


def test_a_chunk_of_every_shape_at_once_agrees():
    """A conforming packet beside a CSRC list, an extension, two segments and
    an overflow: one chunk takes one path, so the mixture is the general
    path's and every packet in it has to come back unchanged."""
    result = assert_paths_agree(
        chunk(_CONFORMING, _TWO_CSRCS, _EXTENSION, _TWO_SRDS, _FOUR_SRDS)
    )
    assert result[1].overflowed.tolist() == [False, False, False, False, True]
    assert result[1].segments.tolist() == [1, 1, 1, 2, 0]


def test_the_bound_on_the_walk_does_not_change_the_agreement():
    for bound in (1, 2, 3, 4, 8):
        assert_paths_agree(
            chunk(_CONFORMING, _TWO_SRDS, _FOUR_SRDS), max_segments=bound
        )


def test_a_truncated_packet_agrees():
    """Rows past a packet's true length are ignored on both paths, so a chunk
    whose sizes cut into the headers reports what it reported before the fast
    path existed (§spec:interface-shape)."""
    rows = chunk(_CONFORMING, _CONFORMING, _CONFORMING, _CONFORMING, stride=32)
    rows[:, 20:] = 0xFF  # what a reused ring still holds past the packet
    sizes = np.array([20, 19, 12, 11], dtype=np.int64)

    result = assert_paths_agree(rows, sizes=sizes)
    assert result[1].segments.tolist() == [1, 0, 0, 0]
    assert result[0].version.tolist() == [2, 2, 2, 0]
    assert result[0].payload_offset.tolist() == [12, 12, 12, 11]


def test_a_chunk_with_no_packets_agrees():
    assert_paths_agree(np.zeros((0, 20), dtype=np.uint8))


def test_a_chunk_that_cannot_be_read_as_words_agrees():
    """An odd stride and a sub-block of a wider buffer both fall back rather
    than raising, and the fallback has to be the same parse."""
    assert_paths_agree(chunk(_CONFORMING, _CONFORMING, stride=21))

    buffer = chunk(_CONFORMING, _CONFORMING, stride=64)
    assert_paths_agree(buffer[:, :24])


# --- over a capture from real senders -----------------------------------------
#
# The strongest evidence available without a NIC (§spec:testing). The capture
# carries the sequence numbers of both legs of a ST 2022-7 pair, one of which
# lost 66 ms mid-recording; the packets are rebuilt around them here, so the
# outage travels into the parse rather than being invented for it.

CAPTURE = pathlib.Path(__file__).parent / "captures" / "convertip-2022-7-outage.npz"

# The format the capture's provenance names: 1280x720 YCbCr-4:2:2 10-bit, which
# is 3200 octets a line, four 800-octet packets to a line and 2880 to a frame.
_SRD_LENGTH = 800
_PACKETS_PER_LINE = 4
_ROWS = 720
_SAMPLES_PER_PACKET = 320


@pytest.fixture(scope="module")
def outage() -> dict[str, Any]:
    with np.load(CAPTURE) as data:
        return {name: data[name] for name in data.files}


def captured_chunk(sequence: np.ndarray) -> np.ndarray:
    """Conforming ST 2110-20 headers carrying a captured sequence stream.

    One packet per captured number, with the row and sample position the
    number implies for the capture's format. Written field by field from the
    standards' diagrams, not through this library's builder — what is borrowed
    from the capture is the sequence stream, gap included.
    """
    numbers = np.asarray(sequence, dtype=np.int64)
    within = numbers % (_ROWS * _PACKETS_PER_LINE)
    block = np.zeros((numbers.size, 20), dtype=np.uint8)

    block[:, 0] = 0x80  # V=2, P=0, X=0, CC=0
    block[:, 1] = 96 | np.where(within == _ROWS * _PACKETS_PER_LINE - 1, 0x80, 0)
    _put_u16(block, 2, numbers & 0xFFFF)  # RTP sequence number
    _put_u16(block, 8, np.full(numbers.size, 0x0A0B))  # SSRC, high half
    _put_u16(block, 10, np.full(numbers.size, 0x0C0D))  # SSRC, low half
    _put_u16(block, 12, (numbers >> 16) & 0xFFFF)  # Extended Sequence Number
    _put_u16(block, 14, np.full(numbers.size, _SRD_LENGTH))  # SRD Length
    _put_u16(block, 16, within // _PACKETS_PER_LINE)  # F=0, SRD Row Number
    _put_u16(block, 18, (within % _PACKETS_PER_LINE) * _SAMPLES_PER_PACKET)
    return block


def _put_u16(block: np.ndarray, octet: int, values: np.ndarray) -> None:
    block[:, octet] = (values >> 8) & 0xFF
    block[:, octet + 1] = values & 0xFF


def test_the_capture_is_the_one_this_gate_was_written_against(
    outage: dict[str, Any],
) -> None:
    """A capture whose origin is unknown proves nothing, so the fixture carries
    its own provenance and this fails if it is swapped for another."""
    recorded = json.loads(str(outage["provenance"]))
    assert "1280x720" in recorded["format"]
    assert recorded["outage"]


@pytest.mark.parametrize("leg", ["a", "b"])
def test_the_two_paths_agree_across_the_captured_stream(
    outage: dict[str, Any], leg: str
) -> None:
    """Every field, over 5748 packets of leg A and the 4002 of leg B — whose
    66 ms hole is the discontinuity the parse has to carry unchanged."""
    rows = captured_chunk(outage[f"{leg}_sequence"])
    with pytest.MonkeyPatch.context() as patch:
        # The captured stream is conforming, so the columns are what read it;
        # a general parse of it would make the comparison below vacuous.
        watch_paths(patch, rtp_module, payload_module)
        headers = parse_rtp(rows)
        parse_payload_headers(rows, headers.payload_offset)

    result = assert_paths_agree(rows)
    assert result[1].segments.sum() == rows.shape[0]
    assert result[1].line.max() == _ROWS - 1


def test_the_loss_window_is_counted_the_same_from_either_path(
    outage: dict[str, Any],
) -> None:
    """What a receiver does with the parse, not just what the parse says: the
    sequence tracker fed from each path has to reach the same numbers, and the
    leg that lost 66 ms has to still show the loss."""
    rows = captured_chunk(outage["b_sequence"])
    fast, general = parse_both(rows)

    counts = []
    for headers, descriptors in (fast, general):
        tracker = SequenceTracker()
        tracker.observe(headers.sequence, extended=descriptors.extended_sequence)
        counts.append(
            (tracker.received, tracker.lost, tracker.discontinuities, tracker.resyncs)
        )

    assert counts[0] == counts[1]
    received, lost, discontinuities, _ = counts[0]
    assert received == rows.shape[0]
    assert lost == 1746  # the outage, exactly as `test_redundancy` measures it
    assert discontinuities == 1
