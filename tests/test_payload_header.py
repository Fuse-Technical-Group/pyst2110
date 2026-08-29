"""The RFC 4175 payload header, parsed over a chunk (SPEC §spec:payload-header).

Vectors are written by hand from RFC 4175 section 4.2 and SMPTE ST 2110-20
section 6.1.4, with the field each byte carries named beside it. Nothing here
is round-tripped through a writer of ours — that would prove only that two of
our modules agree with each other (§spec:testing).

Layout, for reading the vectors below: two octets of Extended Sequence Number,
then six-octet Sample Row Data headers of ``Length(16) | F(1) Row(15) |
C(1) Offset(15)``. Length is octets and must be a multiple of the pgroup;
Offset is a sample position. The C bit says another SRD header follows.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import GeneralPathError, chunk, watch_paths
from pyst2110 import payload
from pyst2110.payload import PayloadHeaders, parse_payload_headers
from pyst2110.rtp import parse_rtp

# Twelve octets of RFC 3550 fixed header: V=2, no padding, no extension, no
# CSRCs; marker clear; payload type 96. The payload header follows at 12.
_RTP = [
    0x80,  # V=2, P=0, X=0, CC=0
    0x60,  # M=0, PT=96
    0x00,
    0x2A,  # sequence number 42
    0x00,
    0x00,
    0x00,
    0x00,  # timestamp
    0x00,
    0x00,
    0x00,
    0x01,  # SSRC
]

# One SRD: 1200 octets of row 42 starting at sample 480, which is one packet
# of a 1080p YCbCr-4:2:2 10-bit flow.
_ONE_SRD = [
    *_RTP,
    0x00,
    0x01,  # Extended Sequence Number 1
    0x04,
    0xB0,  # SRD Length 1200
    0x00,
    0x2A,  # F=0, SRD Row Number 42
    0x01,
    0xE0,  # C=0, SRD Offset 480
]

# Two SRDs: the C bit on the first says a second follows, and the second
# carries the F bit for the temporally second field.
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
    *([0xAA] * 30),  # 10 + 20 octets of sample row data
]

# Three SRDs, the most ST 2110-20 section 6.1.4 permits.
_THREE_SRDS = [
    *_RTP,
    0x00,
    0x03,  # Extended Sequence Number 3
    0x00,
    0x05,  # SRD Length 5
    0x00,
    0x00,  # F=0, SRD Row Number 0
    0x80,
    0x00,  # C=1, SRD Offset 0
    0x00,
    0x05,  # SRD Length 5
    0x00,
    0x01,  # F=0, SRD Row Number 1
    0x80,
    0x02,  # C=1, SRD Offset 2
    0x00,
    0x05,  # SRD Length 5
    0x00,
    0x02,  # F=0, SRD Row Number 2
    0x00,
    0x04,  # C=0, SRD Offset 4
    *([0xBB] * 15),  # three segments of five octets
]

# Four SRDs, one past what ST 2110-20 section 6.1.4 permits: the third still
# sets the C bit, so a parse bounded at three ends with a continuation
# outstanding. Its first three descriptors are indistinguishable from a
# conformant three-SRD packet's.
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
    *([0x00] * 4),  # four segments of one octet
]

# Every bit of both flagged fields set, which is where a mask that took the
# octet whole would report a line number of 32768 too many.
_WIDE_FIELDS = [
    *_RTP,
    0xFF,
    0xFF,  # Extended Sequence Number 65535
    0x00,
    0x00,  # SRD Length 0 — legal with exactly one SRD header
    0xFF,
    0xFF,  # F=1, SRD Row Number 32767
    0x7F,
    0xFF,  # C=0, SRD Offset 32767
]


def parse(*packets: list[int], max_segments: int = 3) -> PayloadHeaders:
    """Parse a chunk the way a consumer does: RTP first, then the payload."""
    rows = chunk(*packets)
    sizes = np.array([len(packet) for packet in packets], dtype=np.int64)
    headers = parse_rtp(rows, sizes=sizes)
    return parse_payload_headers(
        rows, headers.payload_offset, sizes=sizes, max_segments=max_segments
    )


def test_one_segment_reads_back_every_field():
    result = parse(_ONE_SRD)

    assert result.extended_sequence.tolist() == [1]
    assert result.segments.tolist() == [1]
    assert result.packet.tolist() == [0]
    assert result.length.tolist() == [1200]
    assert result.line.tolist() == [42]
    assert result.field.tolist() == [False]
    assert result.offset_samples.tolist() == [480]
    # 12 of RTP, 2 of extended sequence, 6 of the one SRD header.
    assert result.source.tolist() == [20]


def test_the_continuation_bit_makes_the_segment_count_data_dependent():
    result = parse(_TWO_SRDS)

    assert result.segments.tolist() == [2]
    assert result.packet.tolist() == [0, 0]
    assert result.length.tolist() == [10, 20]
    assert result.line.tolist() == [5, 6]
    # The first SRD's offset word is 0x8000: C set, offset zero.
    assert result.offset_samples.tolist() == [0, 100]


def test_the_field_bit_is_separated_from_the_line_number():
    """F rides in the top bit of the row word, so a reader taking the word
    whole would report row 32774 for row 6 of the second field."""
    result = parse(_TWO_SRDS)
    assert result.field.tolist() == [False, True]
    assert result.line.tolist() == [5, 6]


def test_both_flags_set_still_yield_fifteen_bit_values():
    result = parse(_WIDE_FIELDS)
    assert result.extended_sequence.tolist() == [65535]
    assert result.field.tolist() == [True]
    assert result.line.tolist() == [32767]
    assert result.offset_samples.tolist() == [32767]
    # ST 2110-20 permits a zero length only with exactly one SRD header, and
    # it means no sample row data follows.
    assert result.length.tolist() == [0]
    assert result.segments.tolist() == [1]


def test_a_segments_source_follows_the_whole_payload_header():
    """Sample row data for every segment starts after the last SRD header,
    so a two-segment packet's first byte of data is six octets later than a
    one-segment packet's."""
    result = parse(_TWO_SRDS)
    # 12 RTP + 2 extended sequence + 12 for two SRD headers = 26.
    assert result.source.tolist() == [26, 36]
    # And the second segment starts where the first one's ten octets end.
    assert result.source[1] - result.source[0] == result.length[0]


def test_three_segments_chain_their_sources():
    result = parse(_THREE_SRDS)
    assert result.segments.tolist() == [3]
    assert result.line.tolist() == [0, 1, 2]
    assert result.offset_samples.tolist() == [0, 2, 4]
    # 12 RTP + 2 + 18 for three SRD headers = 32, then five octets each.
    assert result.source.tolist() == [32, 37, 42]


def test_packets_with_different_segment_counts_stay_associated():
    """Descriptors are segment-aligned, and ``packet`` is what maps each one
    back to the row it came from."""
    result = parse(_ONE_SRD, _TWO_SRDS, _THREE_SRDS)

    assert result.segments.tolist() == [1, 2, 3]
    assert result.packet.tolist() == [0, 1, 1, 2, 2, 2]
    assert result.line.tolist() == [42, 5, 6, 0, 1, 2]
    assert result.extended_sequence.tolist() == [1, 2, 3]


def test_one_continuing_packet_keeps_the_whole_chunk_walking():
    """The walk leaves its loop only when *no* packet continues.

    The bound is over the chunk, so a single continuing packet among
    single-segment ones still gets every segment it declared. Guards the
    early exit: a test where all packets continue, or none do, passes
    whether the bound is read per chunk or per packet.
    """
    # The continuing packet last, so an exit taken on the first packet's
    # continuation bit rather than the chunk's would drop it.
    result = parse(_ONE_SRD, _ONE_SRD, _THREE_SRDS)

    assert result.segments.tolist() == [1, 1, 3]
    assert result.packet.tolist() == [0, 1, 2, 2, 2]
    assert result.line.tolist() == [42, 42, 0, 1, 2]
    assert not result.overflowed.any()


def test_extended_sequence_is_packet_aligned_not_segment_aligned():
    result = parse(_ONE_SRD, _THREE_SRDS)
    assert result.extended_sequence.shape == (2,)
    assert result.segments.shape == (2,)
    assert result.packet.shape == (4,)


def test_the_extended_sequence_is_the_high_half_of_a_thirty_two_bit_count():
    """What makes loss detection survive a flow fast enough to wrap the RTP
    field inside a frame (§spec:payload-header)."""
    rows = chunk(_WIDE_FIELDS)
    sizes = np.array([len(_WIDE_FIELDS)], dtype=np.int64)
    rtp = parse_rtp(rows, sizes=sizes)
    payload = parse_payload_headers(rows, rtp.payload_offset, sizes=sizes)
    full = (int(payload.extended_sequence[0]) << 16) | int(rtp.sequence[0])
    assert full == (65535 << 16) | 42


def test_an_rtp_extension_shifts_the_payload_header():
    """The X bit is read rather than assumed clear, so the payload header is
    found past the extension instead of inside it (§spec:rtp)."""
    packet = list(_ONE_SRD)
    packet[0] = 0x90  # V=2, X=1, CC=0
    packet[12:12] = [
        0xBE,
        0xDE,  # extension profile
        0x00,
        0x01,  # extension length: one 32-bit word
        0xDE,
        0xAD,
        0xBE,
        0xEF,  # the word
    ]
    result = parse(packet)
    assert result.line.tolist() == [42]
    assert result.offset_samples.tolist() == [480]
    # Eight octets of extension, so everything after it moves by eight.
    assert result.source.tolist() == [20 + 8]


def test_a_packet_too_short_for_its_payload_header_yields_no_descriptor():
    """A truncated packet is accounted for rather than read past the end of."""
    result = parse([*_RTP, 0x00, 0x01, 0x04])
    assert result.segments.tolist() == [0]
    assert result.packet.tolist() == []


def test_a_header_only_view_still_yields_descriptors():
    """A header-data-split receiver hands out headers and payloads in
    separate buffers, so the view parsed here carries no sample row data at
    all — and a length checked against it would reject every packet."""
    result = parse(_ONE_SRD)  # declares 1200 octets and carries none
    assert result.segments.tolist() == [1]
    assert result.length.tolist() == [1200]


def test_the_payload_relative_offset_is_the_split_case_answer():
    """``source`` counts from the packet's start, which is what a contiguous
    packet needs. Subtracting ``data_offset`` gives the offset into the
    payload alone, which is what a split buffer needs."""
    result = parse(_TWO_SRDS)
    relative = result.source - result.data_offset[result.packet]
    assert relative.tolist() == [0, 10]


def test_lengths_are_reported_as_declared_not_clamped():
    """Bounding the gather is the consumer's, against the buffer it reads."""
    result = parse([*_ONE_SRD, *([0x11] * 4)])
    assert result.length.tolist() == [1200]


def test_a_continuation_past_the_bound_is_reported_not_ignored():
    """A fourth SRD is outside ST 2110-20, and dropping a quarter of a
    packet's raster in silence is the failure this flag exists for."""
    result = parse(_THREE_SRDS, max_segments=2)
    assert result.overflowed.tolist() == [True]

    within = parse(_THREE_SRDS)
    assert within.overflowed.tolist() == [False]


def test_the_bound_defaults_to_the_three_st2110_allows():
    result = parse(_FOUR_SRDS)
    assert result.overflowed.tolist() == [True]


def test_an_overflowed_packet_is_named_rather_than_counted():
    """The instruction is to discard the packet, so the flag says which one.
    A conformant three-SRD packet and a truncated four-SRD one carry the same
    three descriptors and the same segment count — nothing else tells them
    apart."""
    result = parse(_THREE_SRDS, _FOUR_SRDS)
    assert result.overflowed.tolist() == [False, True]
    assert result.overflowed.shape == result.segments.shape
    assert result.overflowed_count == 1


def test_an_overflowed_packets_descriptors_are_withheld():
    """Every unparsed SRD header leaves the survivors' source six octets
    early, so gathering one gathers the wrong bytes. None is emitted."""
    result = parse(_THREE_SRDS, _FOUR_SRDS)
    assert result.segments.tolist() == [3, 0]
    assert result.packet.tolist() == [0, 0, 0]
    assert result.line.tolist() == [0, 1, 2]
    assert result.source.tolist() == [32, 37, 42]


def test_an_overflowed_packet_still_reports_its_packet_aligned_fields():
    """Sequence accounting counts the packet as arrived; only its descriptors
    are withheld."""
    result = parse(_FOUR_SRDS)
    assert result.extended_sequence.tolist() == [0]
    assert result.packet.tolist() == []
    assert result.overflowed_count == 1


def test_an_empty_chunk_yields_nothing():
    rows = np.zeros((0, 20), dtype=np.uint8)
    result = parse_payload_headers(rows, np.zeros(0, dtype=np.int64))
    assert result.packet.shape == (0,)
    assert result.segments.shape == (0,)
    assert result.extended_sequence.shape == (0,)
    assert result.overflowed.shape == (0,)
    assert result.overflowed_count == 0


def test_a_flat_array_is_refused():
    with pytest.raises(ValueError, match="packets, stride"):
        parse_payload_headers(np.zeros(20, dtype=np.uint8), np.zeros(1, dtype=np.int64))


def test_an_offset_per_packet_is_required():
    with pytest.raises(ValueError, match="one payload offset per packet"):
        parse_payload_headers(
            np.zeros((3, 20), dtype=np.uint8), np.zeros(2, dtype=np.int64)
        )


def test_a_bound_below_one_segment_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        parse(_ONE_SRD, max_segments=0)


# --- which path the chunk chooses (SPEC §spec:conforming-fast-path) ----------
#
# The parse reads the choice off the chunk: every payload starting where the
# fixed header ends says no packet carried a CSRC list or an extension, and no
# first SRD setting its continuation bit says every packet carries exactly one
# segment. A call that returns took the fast path; one that raises
# `GeneralPathError` took the general one. That the two agree field for field
# is `test_path_equality.py`'s.


def _paths_watched(monkeypatch):
    """Watch the payload parse alone: the RTP parse chooses for itself."""
    watch_paths(monkeypatch, payload)


def test_a_conforming_packet_is_read_by_column(monkeypatch):
    rows = chunk(_ONE_SRD, _ONE_SRD)
    offsets = parse_rtp(rows).payload_offset
    _paths_watched(monkeypatch)
    result = parse_payload_headers(rows, offsets)

    assert result.extended_sequence.tolist() == [1, 1]
    assert result.line.tolist() == [42, 42]
    assert result.offset_samples.tolist() == [480, 480]
    assert result.length.tolist() == [1200, 1200]
    assert result.segments.tolist() == [1, 1]
    assert result.source.tolist() == [20, 20]


def test_a_second_segment_takes_the_general_path(monkeypatch):
    """The continuation bit makes the descriptors segment-aligned, which no
    column slice produces."""
    rows = chunk(_ONE_SRD, _TWO_SRDS)
    offsets = parse_rtp(rows).payload_offset
    _paths_watched(monkeypatch)
    with pytest.raises(GeneralPathError):
        parse_payload_headers(rows, offsets)


def test_a_payload_pushed_back_takes_the_general_path(monkeypatch):
    """A CSRC list or an extension moves where the payload begins, and the
    offsets are what say so — the flags octet is not read twice."""
    rows = chunk(_ONE_SRD)
    _paths_watched(monkeypatch)
    with pytest.raises(GeneralPathError):
        parse_payload_headers(rows, np.array([16]))


def test_a_packet_short_of_a_whole_header_takes_the_general_path(monkeypatch):
    rows = chunk(_ONE_SRD, stride=20)
    _paths_watched(monkeypatch)
    with pytest.raises(GeneralPathError):
        parse_payload_headers(rows, np.array([12]), sizes=np.array([19]))


def test_an_odd_stride_falls_back_rather_than_raising(monkeypatch):
    rows = chunk(_ONE_SRD, stride=21)
    offsets = parse_rtp(rows).payload_offset
    _paths_watched(monkeypatch)
    with pytest.raises(GeneralPathError):
        parse_payload_headers(rows, offsets)

    monkeypatch.undo()
    assert parse_payload_headers(rows, offsets).line.tolist() == [42]


def test_a_sub_block_of_a_wider_buffer_falls_back(monkeypatch):
    buffer = chunk(_ONE_SRD, _ONE_SRD, stride=64)
    rows = buffer[:, :24]
    assert not rows.flags["C_CONTIGUOUS"]
    offsets = parse_rtp(rows).payload_offset

    _paths_watched(monkeypatch)
    with pytest.raises(GeneralPathError):
        parse_payload_headers(rows, offsets)

    monkeypatch.undo()
    assert parse_payload_headers(rows, offsets).line.tolist() == [42, 42]


def test_the_bound_does_not_change_the_conforming_read(monkeypatch):
    """A chunk where nothing continues yields one descriptor a packet at any
    bound of one or more, so ``max_segments`` never forces the general path."""
    rows = chunk(_ONE_SRD)
    offsets = parse_rtp(rows).payload_offset
    _paths_watched(monkeypatch)

    for bound in (1, 2, 3, 8):
        result = parse_payload_headers(rows, offsets, max_segments=bound)
        assert result.segments.tolist() == [1]
        assert result.line.tolist() == [42]


def test_every_descriptor_is_wide_enough_to_scale_into_a_raster():
    """Sixteen bits on the wire is not sixteen bits in a consumer's hands.

    A descriptor is scaled into a raster the moment it is used — a row times
    the pgroups in a line, an offset times a pgroup's octets — and under NEP 50
    a Python multiplier adopts the array's own width rather than widening it.
    Row 2159 of a 2160-line raster times 1152 pgroups a line is 2,487,168,
    which an unsigned sixteen-bit array reports as 62,336: not an error, a
    different part of the picture. So every descriptor is reported wide, and
    the narrow read the fast path makes is its own business
    (§spec:conforming-fast-path)."""
    result = parse(_ONE_SRD)

    for name in (
        "length",
        "line",
        "offset_samples",
        "extended_sequence",
        "source",
        "data_offset",
        "packet",
        "segments",
    ):
        assert getattr(result, name).dtype == np.int64, name

    # The scaling that motivates it, on the widest row ST 2110-20 permits.
    assert (result.line * 1152).dtype == np.int64
