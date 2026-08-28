"""What two redundant legs delivered together (SPEC §spec:redundancy).

ST 2022-7 sends one essence twice by two paths and lets the receiver take
whichever copy of a packet arrives first. Reconstruction is therefore a
question about sequence numbers rather than about pixels: which numbers the
pair delivered between them, which each leg would have missed alone, which
neither carried, and how far apart in time the two copies of one packet
arrive.

That last figure is the one a deployment turns on. A receiver has to hold a
packet from the leading leg until the lagging leg's copy could still arrive,
so the **path differential** sizes its buffer, and the standard bounds what
a sender may impose. It is reported signed: a positive skew means the second
leg lags.

**The span both legs cover is what can be judged, and it is not the union.**
Two captures rarely start and stop on the same packet, so a leg whose
recording ended earlier has a tail the other lacks — read as loss, that tail
would dwarf everything real. So the measurement is bounded to the sequence
range present on both, and a pair sharing no range at all is refused rather
than reported as total loss on both sides.

Numpy only, over the thirty-two-bit extended sequence number
(§spec:payload-header). Nothing here reads a payload, and nothing here needs
a NIC (§spec:testing).
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import NDArray

__all__ = ["LegPair", "reconstruct"]


@dataclasses.dataclass(frozen=True)
class LegPair:
    """The account of one redundant pair over the span both legs cover."""

    #: Distinct sequence numbers the pair delivered between them.
    delivered: int
    #: Numbers both legs carried. The receiver discards one copy of each.
    duplicated: int
    #: Numbers only the second leg carried — what the first would have lost.
    recovered_a: int
    #: Numbers only the first leg carried.
    recovered_b: int
    #: Numbers neither leg carried. This is the loss redundancy cannot mend.
    lost: int
    #: Packets present on both legs, and so the population the skew is
    #: measured over.
    paired: int
    #: The inclusive sequence range measured, being the overlap of the two.
    span: tuple[int, int]
    #: The path differential, in nanoseconds, over the paired packets. Signed:
    #: positive where the second leg arrived later.
    skew_min_ns: int
    skew_max_ns: int


def reconstruct(
    a_sequence: NDArray[np.int64],
    a_arrival: NDArray[np.int64],
    b_sequence: NDArray[np.int64],
    b_arrival: NDArray[np.int64],
) -> LegPair:
    """Account for what two legs of one flow delivered.

    Each leg is an array of extended sequence numbers and an array of arrival
    times in nanoseconds, in any order and of any length. The two legs need
    not cover the same window; the overlap is what is measured.
    """
    a_seq = np.asarray(a_sequence, dtype=np.int64)
    b_seq = np.asarray(b_sequence, dtype=np.int64)
    a_ts = np.asarray(a_arrival, dtype=np.int64)
    b_ts = np.asarray(b_arrival, dtype=np.int64)

    for seq, ts, name in ((a_seq, a_ts, "first"), (b_seq, b_ts, "second")):
        if seq.shape != ts.shape:
            raise ValueError(
                f"the {name} leg's sequence and arrival arrays are not the "
                f"same length: {seq.size} and {ts.size}"
            )
        if seq.size == 0:
            raise ValueError(f"the {name} leg is empty")

    # One arrival per number, the earliest, so a leg that duplicated a packet
    # on its own path does not count twice or pick the later copy: a receiver
    # takes the first that reaches it.
    a_unique, a_first = _earliest(a_seq, a_ts)
    b_unique, b_first = _earliest(b_seq, b_ts)

    low = max(int(a_unique[0]), int(b_unique[0]))
    high = min(int(a_unique[-1]), int(b_unique[-1]))
    if low > high:
        raise ValueError(
            "the two legs have no sequence numbers in common — "
            f"the first spans {a_unique[0]}..{a_unique[-1]} and the second "
            f"{b_unique[0]}..{b_unique[-1]}, which is two recordings of "
            "different windows rather than one redundant pair"
        )

    a_in = (a_unique >= low) & (a_unique <= high)
    b_in = (b_unique >= low) & (b_unique <= high)
    a_span, a_span_ts = a_unique[a_in], a_first[a_in]
    b_span, b_span_ts = b_unique[b_in], b_first[b_in]

    both = np.intersect1d(a_span, b_span, assume_unique=True)
    delivered = int(np.union1d(a_span, b_span).size)
    expected = high - low + 1

    skew = (
        b_span_ts[np.isin(b_span, both, assume_unique=True)]
        - a_span_ts[np.isin(a_span, both, assume_unique=True)]
    )
    return LegPair(
        delivered=delivered,
        duplicated=int(both.size),
        recovered_a=int(b_span.size - both.size),
        recovered_b=int(a_span.size - both.size),
        lost=int(expected - delivered),
        paired=int(both.size),
        span=(low, high),
        skew_min_ns=int(skew.min()) if skew.size else 0,
        skew_max_ns=int(skew.max()) if skew.size else 0,
    )


def _earliest(
    sequence: NDArray[np.int64], arrival: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Sorted unique numbers, each with the earliest time it arrived at."""
    order = np.lexsort((arrival, sequence))
    seq, ts = sequence[order], arrival[order]
    first = np.ones(seq.shape, dtype=np.bool_)
    first[1:] = seq[1:] != seq[:-1]
    return seq[first], ts[first]
