"""The RFC 3550 fixed header, parsed over a chunk (SPEC §spec:rtp).

Every vector is written by hand from RFC 3550 section 5.1's field diagram and
SMPTE ST 2110-20 section 6.1.2, byte by byte, with the field each byte carries
named beside it. Round-tripping our own writer would prove only that two of
our modules agree with each other (§spec:testing).
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import GeneralPathError, chunk, watch_paths
from pyst2110 import rtp
from pyst2110.rtp import FIXED_HEADER_SIZE, parse_rtp

# RFC 3550 §5.1. Version 2, no padding, no extension, no CSRCs; marker set;
# payload type 96, the dynamic type ST 2110-10 allocates.
_PLAIN = [
    0x80,  # V=2 (10), P=0, X=0, CC=0
    0xE0,  # M=1, PT=96 (0x60)
    0x12,
    0x34,  # sequence number 0x1234
    0xDE,
    0xAD,
    0xBE,
    0xEF,  # timestamp 0xDEADBEEF
    0x0A,
    0x0B,
    0x0C,
    0x0D,  # SSRC 0x0A0B0C0D
]

# The same header with the marker clear and the sequence number at the top of
# the space, which is where a 16-bit counter is about to wrap.
_NO_MARKER = [
    0x80,  # V=2, P=0, X=0, CC=0
    0x60,  # M=0, PT=96
    0xFF,
    0xFF,  # sequence number 0xFFFF
    0x00,
    0x00,
    0x00,
    0x01,  # timestamp 1
    0xFF,
    0xFF,
    0xFF,
    0xFF,  # SSRC 0xFFFFFFFF
]

# Two CSRCs, so the payload starts eight bytes later than the fixed header ends
# (RFC 3550 §5.1: the CSRC list is CC entries of four octets).
_TWO_CSRCS = [
    0x82,  # V=2, P=0, X=0, CC=2
    0x60,  # M=0, PT=96
    0x00,
    0x05,  # sequence number 5
    0x00,
    0x00,
    0x00,
    0x00,  # timestamp 0
    0x00,
    0x00,
    0x00,
    0x02,  # SSRC 2
    0x11,
    0x11,
    0x11,
    0x11,  # CSRC[0]
    0x22,
    0x22,
    0x22,
    0x22,  # CSRC[1]
]

# The extension bit set, with the RFC 3550 §5.3.1 extension header: two
# profile-defined octets then a length counting 32-bit words *after* those
# four octets. Three words here, so the extension occupies 4 + 12 = 16 bytes.
_EXTENSION = [
    0x90,  # V=2, P=0, X=1, CC=0
    0x60,  # M=0, PT=96
    0x00,
    0x07,  # sequence number 7
    0x00,
    0x00,
    0x00,
    0x00,  # timestamp 0
    0x00,
    0x00,
    0x00,
    0x03,  # SSRC 3
    0xBE,
    0xDE,  # extension profile ("defined by profile")
    0x00,
    0x03,  # extension length: 3 words of 4 bytes
    0x00,
    0x00,
    0x00,
    0x00,  # extension word 0
    0x00,
    0x00,
    0x00,
    0x00,  # extension word 1
    0x00,
    0x00,
    0x00,
    0x00,  # extension word 2
]

# Padding and extension together, with one CSRC: the payload starts at
# 12 + 4 + 4 + 0 = 20.
_PADDED_EXTENSION_ONE_CSRC = [
    0xB1,  # V=2, P=1, X=1, CC=1
    0xE0,  # M=1, PT=96
    0x00,
    0x09,  # sequence number 9
    0x00,
    0x00,
    0x00,
    0x00,  # timestamp 0
    0x00,
    0x00,
    0x00,
    0x04,  # SSRC 4
    0x33,
    0x33,
    0x33,
    0x33,  # CSRC[0]
    0xBE,
    0xDE,  # extension profile
    0x00,
    0x00,  # extension length: zero words, which RFC 3550 permits
]


def test_the_fixed_header_is_twelve_octets():
    assert FIXED_HEADER_SIZE == 12


def test_every_field_of_a_plain_header_reads_back():
    headers = parse_rtp(chunk(_PLAIN))

    assert headers.version.tolist() == [2]
    assert headers.padding.tolist() == [False]
    assert headers.extension.tolist() == [False]
    assert headers.csrc_count.tolist() == [0]
    assert headers.marker.tolist() == [True]
    assert headers.payload_type.tolist() == [96]
    assert headers.sequence.tolist() == [0x1234]
    assert headers.timestamp.tolist() == [0xDEADBEEF]
    assert headers.ssrc.tolist() == [0x0A0B0C0D]
    assert headers.payload_offset.tolist() == [12]


def test_the_marker_bit_is_separated_from_the_payload_type():
    """0xE0 is the marker plus type 96; reading the octet whole gives 224."""
    headers = parse_rtp(chunk(_PLAIN, _NO_MARKER))
    assert headers.marker.tolist() == [True, False]
    assert headers.payload_type.tolist() == [96, 96]


def test_the_top_of_the_sequence_space_is_not_sign_extended():
    headers = parse_rtp(chunk(_NO_MARKER))
    assert headers.sequence.tolist() == [0xFFFF]
    assert headers.ssrc.tolist() == [0xFFFFFFFF]
    assert headers.timestamp.tolist() == [1]


def test_a_chunk_parses_packet_aligned():
    """One array per field, one row per packet — the interface shape."""
    headers = parse_rtp(chunk(_PLAIN, _NO_MARKER, _TWO_CSRCS))
    assert headers.sequence.tolist() == [0x1234, 0xFFFF, 5]
    for field in (headers.marker, headers.timestamp, headers.payload_offset):
        assert field.shape == (3,)


def test_csrcs_push_the_payload_back_four_octets_each():
    headers = parse_rtp(chunk(_TWO_CSRCS))
    assert headers.csrc_count.tolist() == [2]
    assert headers.payload_offset.tolist() == [12 + 8]


def test_an_extension_shifts_everything_after_it():
    """§spec:rtp: the X bit is read rather than assumed clear.

    ST 2110-20 senders do not set it, so a receiver that assumed it clear
    would misparse anything that did — and would do so silently, reading the
    extension's first octets as a payload header.
    """
    headers = parse_rtp(chunk(_EXTENSION))
    assert headers.extension.tolist() == [True]
    # 12 fixed + 4 extension header + 3 words of 4 bytes.
    assert headers.payload_offset.tolist() == [28]


def test_a_zero_length_extension_still_costs_its_own_four_octets():
    headers = parse_rtp(chunk(_PADDED_EXTENSION_ONE_CSRC))
    assert headers.padding.tolist() == [True]
    assert headers.extension.tolist() == [True]
    assert headers.csrc_count.tolist() == [1]
    # 12 fixed + 4 for the one CSRC + 4 for the extension header + no words.
    assert headers.payload_offset.tolist() == [20]


def test_csrcs_and_an_extension_compose():
    """The extension length lives after the CSRC list, not after the fixed
    header — reading it at a fixed offset would give a CSRC's octets."""
    packet = list(_TWO_CSRCS)
    packet[0] = 0x92  # V=2, X=1, CC=2
    packet += [
        0xBE,
        0xDE,  # extension profile
        0x00,
        0x02,  # extension length: two words
        *([0x00] * 8),  # the two words
    ]
    headers = parse_rtp(chunk(packet))
    assert headers.payload_offset.tolist() == [12 + 8 + 4 + 8]


def test_mixed_packets_each_get_their_own_payload_offset():
    """The offset is per packet, so it cannot be hoisted out of the chunk."""
    headers = parse_rtp(chunk(_PLAIN, _TWO_CSRCS, _EXTENSION))
    assert headers.payload_offset.tolist() == [12, 20, 28]


def test_sizes_bound_the_extension_read():
    """A receiver writes at most the stride and reports the true size
    separately (§spec:interface-shape), so a truncated packet's extension
    length is not read out of the bytes that follow it in the buffer."""
    rows = chunk(_EXTENSION, stride=64)
    # Poison the buffer past the true packet, as a reused ring would hold.
    rows[0, 28:] = 0xFF
    headers = parse_rtp(rows, sizes=np.array([28]))
    assert headers.payload_offset.tolist() == [28]

    # A packet cut off before its extension length cannot be trusted to have
    # one, so the offset stops at the size rather than reading past it.
    truncated = parse_rtp(rows, sizes=np.array([14]))
    assert truncated.payload_offset.tolist() == [14]


def test_sizes_bound_the_fixed_fields_to_a_whole_header():
    """The fixed header is one twelve-octet unit, so a packet reporting less
    carries none of it and every field reads zero — not the bytes a reused
    ring still holds there (§spec:interface-shape). Version zero is what says
    so: no sender emits it."""
    rows = chunk(_PLAIN, _NO_MARKER, stride=64)
    headers = parse_rtp(rows, sizes=np.array([12, 11]))

    assert headers.version.tolist() == [2, 0]
    assert headers.marker.tolist() == [True, False]
    assert headers.payload_type.tolist() == [96, 0]
    assert headers.extension.tolist() == [False, False]
    assert headers.csrc_count.tolist() == [0, 0]
    assert headers.sequence.tolist() == [0x1234, 0]
    assert headers.timestamp.tolist() == [0xDEADBEEF, 0]
    assert headers.ssrc.tolist() == [0x0A0B0C0D, 0]
    # The payload starts where a packet that short ends: at nothing.
    assert headers.payload_offset.tolist() == [12, 11]


def test_a_flat_array_is_refused():
    with pytest.raises(ValueError, match="packets, stride"):
        parse_rtp(np.zeros(12, dtype=np.uint8))


def test_a_view_narrower_than_the_fixed_header_is_refused():
    with pytest.raises(ValueError, match="at least 12 columns"):
        parse_rtp(np.zeros((4, 11), dtype=np.uint8))


def test_a_non_byte_array_is_refused():
    """The interface is a byte view; a wider dtype would silently misalign."""
    with pytest.raises(ValueError, match="uint8"):
        parse_rtp(np.zeros((4, 12), dtype=np.uint16))


def test_sizes_must_match_the_packet_count():
    with pytest.raises(ValueError, match="one size per packet"):
        parse_rtp(chunk(_PLAIN, _NO_MARKER), sizes=np.array([12]))


def test_an_empty_chunk_yields_empty_fields():
    headers = parse_rtp(np.zeros((0, 12), dtype=np.uint8))
    assert headers.sequence.shape == (0,)
    assert headers.payload_offset.shape == (0,)


def test_a_version_other_than_two_is_reported_not_corrected():
    """The parse reports what the wire said. A receiver deciding what to do
    with a version-1 packet has more context than this does."""
    packet = list(_PLAIN)
    packet[0] = 0x40  # V=1
    assert parse_rtp(chunk(packet)).version.tolist() == [1]


# --- which path the chunk chooses (SPEC §spec:conforming-fast-path) ----------
#
# The two parses agree field for field — `test_path_equality.py` is the gate on
# that — so nothing in the *result* says which one ran. These pin the choice
# itself: a call that returns took the fast path, one that raises
# `GeneralPathError` took the general one.


def test_a_conforming_chunk_is_read_by_column(monkeypatch):
    """No CSRC list, no extension, a whole fixed header in every packet."""
    watch_paths(monkeypatch, rtp)
    headers = parse_rtp(chunk(_PLAIN, _NO_MARKER))

    assert headers.sequence.tolist() == [0x1234, 0xFFFF]
    assert headers.marker.tolist() == [True, False]
    assert headers.payload_offset.tolist() == [12, 12]


def test_a_csrc_list_takes_the_general_path(monkeypatch):
    """The payload offset stops being a constant, which is the whole reason
    the general path exists."""
    watch_paths(monkeypatch, rtp)
    with pytest.raises(GeneralPathError):
        parse_rtp(chunk(_TWO_CSRCS))


def test_an_extension_takes_the_general_path(monkeypatch):
    watch_paths(monkeypatch, rtp)
    with pytest.raises(GeneralPathError):
        parse_rtp(chunk(_EXTENSION))


def test_padding_takes_the_general_path(monkeypatch):
    """P does not move the payload, but reading the flags octet whole is one
    pass where four masks are four, so a padded packet is the general path's
    rather than a fifth test on the fast one."""
    padded = list(_PLAIN)
    padded[0] = 0xA0  # V=2, P=1, X=0, CC=0
    watch_paths(monkeypatch, rtp)
    with pytest.raises(GeneralPathError):
        parse_rtp(chunk(padded))


def test_a_version_other_than_two_takes_the_general_path(monkeypatch):
    """Version is reported rather than corrected, and only the general path
    can report one the fast path's constant does not hold."""
    wrong = list(_PLAIN)
    wrong[0] = 0x40  # V=1
    watch_paths(monkeypatch, rtp)
    with pytest.raises(GeneralPathError):
        parse_rtp(chunk(wrong))


def test_a_packet_short_of_a_fixed_header_takes_the_general_path(monkeypatch):
    """Every field of such a packet reads zero, which is a rule about a
    packet rather than about a column."""
    watch_paths(monkeypatch, rtp)
    with pytest.raises(GeneralPathError):
        parse_rtp(chunk(_PLAIN, stride=20), sizes=np.array([11]))


def test_an_odd_stride_falls_back_rather_than_raising(monkeypatch):
    """A 16-bit view needs an even number of octets a row. An odd stride is a
    legitimate chunk that no column slice can read, so it takes the general
    path and reads the same."""
    rows = chunk(_PLAIN, stride=13)
    watch_paths(monkeypatch, rtp)
    with pytest.raises(GeneralPathError):
        parse_rtp(rows)

    monkeypatch.undo()
    assert parse_rtp(rows).sequence.tolist() == [0x1234]


def test_a_sub_block_of_a_wider_buffer_falls_back(monkeypatch):
    """A header sub-block sliced out of a ring is not contiguous, so its rows
    are not words either — the same fallback, for the same reason."""
    buffer = chunk(_PLAIN, _NO_MARKER, stride=64)
    rows = buffer[:, :16]
    assert not rows.flags["C_CONTIGUOUS"]

    watch_paths(monkeypatch, rtp)
    with pytest.raises(GeneralPathError):
        parse_rtp(rows)

    monkeypatch.undo()
    assert parse_rtp(rows).sequence.tolist() == [0x1234, 0xFFFF]


def test_an_empty_chunk_takes_the_general_path(monkeypatch):
    """Nothing to read a column of, and the general path already answers it."""
    watch_paths(monkeypatch, rtp)
    with pytest.raises(GeneralPathError):
        parse_rtp(np.zeros((0, 12), dtype=np.uint8))
