"""Reconstructing one stream from two redundant legs (SPEC §spec:redundancy).

ST 2022-7 sends the same essence twice by two paths and lets the receiver
take whichever packet arrives first. What is measured here is what the pair
delivered together, what each leg would have missed alone, and how far apart
in time the two copies of one packet arrive — the path differential a
receiver has to buffer for.

Synthetic legs throughout: the arithmetic is over sequence numbers and
arrival times, and neither needs a NIC (§spec:testing).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyst2110.redundancy import reconstruct

#: A nanosecond per packet, so a skew reads as a count of packets.
STEP = 1_000


def _leg(sequence, *, offset_ns=0, step=STEP):
    seq = np.asarray(sequence, dtype=np.int64)
    return seq, seq * step + offset_ns


def test_two_perfect_legs_deliver_the_span_once() -> None:
    """Neither leg lost anything, so the union is the span and nothing was
    recovered by the pairing."""
    a = _leg(range(100))
    b = _leg(range(100))
    r = reconstruct(*a, *b)
    assert r.delivered == 100
    assert r.duplicated == 100
    assert r.recovered_a == 0
    assert r.recovered_b == 0
    assert r.lost == 0


def test_a_gap_on_one_leg_is_recovered_from_the_other() -> None:
    """The whole point of the standard: a packet missing from one path and
    present on the other reaches the receiver."""
    a = _leg([0, 1, 3, 4])
    b = _leg([0, 1, 2, 3, 4])
    r = reconstruct(*a, *b)
    assert r.delivered == 5
    assert r.recovered_a == 1  # leg A would have missed sequence 2
    assert r.recovered_b == 0
    assert r.lost == 0


def test_a_gap_on_both_legs_is_unrecoverable() -> None:
    a = _leg([0, 1, 3])
    b = _leg([0, 1, 3])
    r = reconstruct(*a, *b)
    assert r.lost == 1
    assert r.delivered == 3


def test_the_measured_span_is_the_overlap_not_the_union() -> None:
    """A capture stopped on one leg before the other must not read the tail
    as loss. The span both legs cover is what can be judged."""
    a = _leg(range(0, 100))
    b = _leg(range(0, 60))
    r = reconstruct(*a, *b)
    assert r.span == (0, 59)
    assert r.lost == 0
    assert r.recovered_b == 0


def test_the_path_differential_is_signed_and_per_packet() -> None:
    """Leg B arrives a fixed 5 us later, so the skew is that, and its sign
    says which leg leads."""
    a = _leg(range(50))
    b = _leg(range(50), offset_ns=5_000)
    r = reconstruct(*a, *b)
    assert r.skew_max_ns == 5_000
    assert r.skew_min_ns == 5_000


def test_the_differential_is_measured_only_where_both_legs_have_the_packet() -> None:
    """A packet one leg never carried has no pair, so it contributes no
    skew rather than contributing a wrong one."""
    a = _leg([0, 1, 2])
    b = _leg([0, 2])
    r = reconstruct(*a, *b)
    assert r.paired == 2


def test_legs_with_no_overlap_at_all_are_refused() -> None:
    """Two captures of different windows are not a redundant pair, and
    reading them as one would report the whole of each as recovered."""
    a = _leg(range(0, 50))
    b = _leg(range(100, 150))
    with pytest.raises(ValueError, match="no sequence numbers in common"):
        reconstruct(*a, *b)


def test_an_empty_leg_is_refused() -> None:
    with pytest.raises(ValueError, match="empty"):
        reconstruct(np.zeros(0, np.int64), np.zeros(0, np.int64), *_leg(range(10)))


def test_arrival_arrays_have_to_match_their_sequences() -> None:
    with pytest.raises(ValueError, match="same length"):
        reconstruct(np.arange(5), np.arange(4), *_leg(range(5)))
