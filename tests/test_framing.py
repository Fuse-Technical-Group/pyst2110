"""Sequence continuity and frame boundaries across chunks (SPEC §spec:rtp).

The sequence cases are ported from an earlier implementation of the same
arithmetic. Modular arithmetic over a 16-bit space that wraps every fifth
of a second at ST 2110 rates is where an off-by-one hides, and none of it
needs a NIC (§spec:testing).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyst2110.framing import FrameTracker, SequenceTracker, frame_boundaries


def test_a_clean_run_has_no_discontinuities():
    tracker = SequenceTracker()
    tracker.observe(np.arange(1000, 1100))
    assert tracker.received == 100
    assert tracker.lost == 0
    assert tracker.discontinuities == 0
    assert tracker.expected == 100


def test_a_gap_counts_the_missing_packets():
    tracker = SequenceTracker()
    tracker.observe(np.array([10, 11, 15, 16]))
    assert tracker.lost == 3
    assert tracker.discontinuities == 1
    assert tracker.expected == 7


def test_a_gap_across_two_chunks_is_counted_once():
    """The tracker carries the last number it saw, so a gap straddling two
    acquisitions is classified where it happened and only there."""
    tracker = SequenceTracker()
    tracker.observe(np.array([10, 11]))
    tracker.observe(np.array([14, 15]))
    assert tracker.lost == 2
    assert tracker.discontinuities == 1


def test_the_first_chunk_has_nothing_to_compare_against():
    tracker = SequenceTracker()
    tracker.observe(np.array([500]))
    assert tracker.received == 1
    assert tracker.discontinuities == 0
    assert tracker.expected == 1
    assert tracker.lost == 0


def test_wraparound_is_continuous():
    tracker = SequenceTracker()
    tracker.observe(np.array([65534, 65535, 0, 1]))
    assert tracker.lost == 0
    assert tracker.discontinuities == 0


def test_a_gap_across_the_wrap_is_still_a_gap():
    tracker = SequenceTracker()
    tracker.observe(np.array([65534, 2]))
    assert tracker.lost == 3
    assert tracker.discontinuities == 1


def test_wraparound_across_two_chunks_is_continuous():
    tracker = SequenceTracker()
    tracker.observe(np.array([65535]))
    tracker.observe(np.array([0, 1]))
    assert tracker.lost == 0
    assert tracker.discontinuities == 0


def test_many_wraps_keep_accumulating():
    """The span is kept unwrapped, so a run longer than the sequence space is
    ordinary arithmetic rather than a special case per wrap."""
    tracker = SequenceTracker()
    for start in range(0, 200_000, 1000):
        tracker.observe(np.arange(start, start + 1000) % 65536)
    assert tracker.received == 200_000
    assert tracker.expected == 200_000
    assert tracker.lost == 0
    assert tracker.discontinuities == 0


def test_a_repeated_number_is_a_duplicate_not_a_loss():
    """RFC 3550 counts expected less received, so a duplicate goes negative.

    That is the definition rather than a defect: three packets arrived
    spanning two sequence numbers.
    """
    tracker = SequenceTracker()
    tracker.observe(np.array([7, 7, 8]))
    assert tracker.duplicated == 1
    assert tracker.expected == 2
    assert tracker.received == 3
    assert tracker.lost == -1
    assert tracker.discontinuities == 1


def test_a_step_backwards_is_reordering_not_a_huge_loss():
    """A late packet is not a loss, and not most of the sequence space."""
    tracker = SequenceTracker()
    tracker.observe(np.array([100, 99, 101]))
    assert tracker.reordered == 1
    # 99..101 is three numbers and three packets arrived: nothing is missing.
    assert tracker.lost == 0
    assert tracker.discontinuities == 2


def test_an_empty_chunk_changes_nothing():
    tracker = SequenceTracker()
    tracker.observe(np.array([5, 6]))
    tracker.observe(np.array([], dtype=np.int64))
    tracker.observe(np.array([7]))
    assert tracker.received == 3
    assert tracker.discontinuities == 0


def test_a_jump_too_large_to_read_is_a_resync_not_a_loss():
    """Beyond RFC 3550's MAX_DROPOUT a gap and a restarted sender are
    indistinguishable, so the jump is reported as its own number rather than
    charged to the loss count as fiction."""
    tracker = SequenceTracker()
    tracker.observe(np.array([10, 11, 40_000, 40_001]))
    assert tracker.resyncs == 1
    assert tracker.received == 4
    # Two spans of two, not a span of forty thousand.
    assert tracker.expected == 4
    assert tracker.lost == 0


def test_loss_either_side_of_a_resync_still_counts():
    tracker = SequenceTracker()
    tracker.observe(np.array([10, 12, 40_000, 40_003]))
    assert tracker.resyncs == 1
    assert tracker.expected == 3 + 4
    assert tracker.received == 4
    assert tracker.lost == 3


def test_a_resync_across_two_chunks_is_seen():
    tracker = SequenceTracker()
    tracker.observe(np.array([10, 11]))
    tracker.observe(np.array([40_000]))
    assert tracker.resyncs == 1
    assert tracker.lost == 0


def test_the_extended_number_makes_a_gap_past_the_wrap_a_loss():
    """At 1080p60 a 70,000-packet outage is a quarter of a second, and wider
    than the sixteen-bit field. With the extended sequence number the gap is
    plain arithmetic rather than a jump nobody can classify (§spec:rtp)."""
    tracker = SequenceTracker()
    # 10, then 70,010: the low halves are 10 and 4474, the high halves 0 and 1.
    tracker.observe(np.array([10, 4474]), extended=np.array([0, 1]))
    assert tracker.resyncs == 0
    assert tracker.expected == 70_001
    assert tracker.lost == 69_999
    assert tracker.discontinuities == 1


def test_the_same_gap_without_the_extended_number_is_a_resync():
    """The sixteen-bit field cannot tell that gap from a restarted sender, so
    an RTP-only caller still gets RFC 3550's answer."""
    tracker = SequenceTracker()
    tracker.observe(np.array([10, 4474]))
    assert tracker.resyncs == 1
    assert tracker.lost == 0


def test_the_extended_number_carries_the_wrap_of_the_rtp_field():
    tracker = SequenceTracker()
    tracker.observe(np.array([65534, 65535, 0, 1]), extended=np.array([0, 0, 1, 1]))
    assert tracker.lost == 0
    assert tracker.discontinuities == 0
    assert tracker.resyncs == 0


def test_a_gap_across_two_chunks_is_counted_once_in_the_wide_space():
    tracker = SequenceTracker()
    tracker.observe(np.array([65535]), extended=np.array([0]))
    tracker.observe(np.array([9]), extended=np.array([1]))
    assert tracker.lost == 9
    assert tracker.discontinuities == 1


def test_reordering_stays_reordering_in_the_wide_space():
    tracker = SequenceTracker()
    tracker.observe(np.array([0, 65535, 1]), extended=np.array([1, 0, 1]))
    assert tracker.reordered == 1
    assert tracker.lost == 0


def test_a_restarted_sender_is_still_a_resync_in_the_wide_space():
    """Half the extended space is hours of packets, so nothing but a sender
    renumbering itself reaches it."""
    tracker = SequenceTracker()
    tracker.observe(np.array([10, 11]), extended=np.array([0, 0]))
    tracker.observe(np.array([0]), extended=np.array([0xC000]))
    assert tracker.resyncs == 1
    assert tracker.expected == 3
    assert tracker.lost == 0


def test_an_extended_number_per_packet_is_required():
    tracker = SequenceTracker()
    with pytest.raises(ValueError, match="one extended sequence number per packet"):
        tracker.observe(np.array([1, 2, 3]), extended=np.array([0, 0]))


def test_mixing_the_two_sequence_spaces_is_refused():
    """A chunk counted in 2^32 and the next in 2^16 would step between two
    unrelated numbers, so the mismatch is named rather than counted."""
    tracker = SequenceTracker()
    tracker.observe(np.array([10, 11]), extended=np.array([0, 0]))
    with pytest.raises(ValueError, match="one sequence space"):
        tracker.observe(np.array([12]))


def test_the_summary_names_every_number_it_carries():
    tracker = SequenceTracker()
    tracker.observe(np.array([10, 11, 15]))
    summary = tracker.summary()
    for word in ("packets", "lost", "reordered", "duplicated", "discontinuities"):
        assert word in summary


def test_a_two_dimensional_input_is_refused():
    """A chunk's sequence numbers are one per packet; a (packets, stride)
    view passed here by mistake would be silently ravelled."""
    tracker = SequenceTracker()
    with pytest.raises(ValueError, match="one number per packet"):
        tracker.observe(np.zeros((4, 2), dtype=np.int64))


def test_a_frame_ends_at_the_markers_rising_edge():
    """RFC 4175 sets the marker on the last packet of a frame, so the packet
    after a rising edge begins the next one (§spec:rtp)."""
    marker = np.array([False, False, True, False, False, True])
    assert frame_boundaries(marker).tolist() == [False, False, True, False, False, True]

    tracker = FrameTracker()
    assert tracker.observe(marker).tolist() == [0, 0, 0, 1, 1, 1]
    assert tracker.frames == 2


def test_a_frame_index_continues_across_chunks():
    tracker = FrameTracker()
    assert tracker.observe(np.array([False, True, False])).tolist() == [0, 0, 1]
    assert tracker.frames == 1
    # The second chunk resumes mid-frame rather than restarting at zero.
    assert tracker.observe(np.array([False, True, False])).tolist() == [1, 1, 2]
    assert tracker.frames == 2


def test_an_interlaced_frame_ends_the_count_twice():
    """ST 2110-20 section 6.1.2 marks the last packet of each field, so one
    interlaced frame counts two — half of what packets_per_frame() sizes."""
    tracker = FrameTracker()
    tracker.observe(np.array([False, False, True]))  # end of the first field
    tracker.observe(np.array([False, False, True]))  # end of the second
    assert tracker.frames == 2


def test_a_marker_on_the_very_first_packet_is_an_edge():
    tracker = FrameTracker()
    assert tracker.observe(np.array([True, False])).tolist() == [0, 1]
    assert tracker.frames == 1


def test_a_marker_split_across_the_chunk_boundary_is_one_edge():
    """The previous chunk's last marker is carried, so a frame that ends on a
    chunk boundary does not end twice."""
    tracker = FrameTracker()
    tracker.observe(np.array([False, True]))
    assert tracker.frames == 1
    # A repeat of the marked packet is not a second rising edge.
    assert tracker.observe(np.array([True, False])).tolist() == [1, 1]
    assert tracker.frames == 1


def test_a_repeated_final_packet_does_not_open_an_empty_frame():
    """Two markers in a row is one rising edge: a duplicated last packet, or
    the same packet arriving on both legs of a redundant pair."""
    assert frame_boundaries(np.array([True, True])).tolist() == [True, False]
    tracker = FrameTracker()
    tracker.observe(np.array([False, True, True, False]))
    assert tracker.frames == 1


def test_an_empty_chunk_leaves_the_frame_count_alone():
    tracker = FrameTracker()
    tracker.observe(np.array([False, True]))
    assert tracker.observe(np.array([], dtype=bool)).tolist() == []
    assert tracker.frames == 1


def test_a_chunk_with_no_marker_completes_no_frame():
    tracker = FrameTracker()
    assert tracker.observe(np.zeros(2000, dtype=bool)).tolist() == [0] * 2000
    assert tracker.frames == 0


def test_frame_boundaries_take_the_previous_marker():
    """The stateless form is what the tracker is built from, and what a
    caller holding its own state can use directly."""
    assert frame_boundaries(np.array([True, False]), previous=True).tolist() == [
        False,
        False,
    ]


def test_a_two_dimensional_marker_column_is_refused():
    tracker = FrameTracker()
    with pytest.raises(ValueError, match="one marker per packet"):
        tracker.observe(np.zeros((4, 2), dtype=bool))
