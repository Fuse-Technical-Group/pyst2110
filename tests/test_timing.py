"""The ST 2110-21 timing model (SPEC §spec:timing).

Limits and schedule constants are hand-computed from the formulas in SMPTE
ST 2110-21:2022 sections 6.2-6.4 and 7.1, cited at each case. The leaky
buckets are checked two ways: against senders whose behaviour the standard
describes in prose (ideal, bursting, early, late), and against a per-packet
reference loop written straight from sections 6.6.1 and 6.6.2 — the
vectorised passes must agree with it on jittered random senders
(§spec:testing).
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from pyst2110.geometry import choose_payload_size, packets_per_frame
from pyst2110.sdp import (
    SENDER_TYPE_NARROW,
    SENDER_TYPE_NARROW_LINEAR,
    SENDER_TYPE_WIDE,
    SdpVideo,
)
from pyst2110.timing import (
    Limits,
    Schedule,
    c_inst,
    frame_starts,
    measure,
    read_schedule,
    read_times,
    sender_limits,
    video_datum,
    vrx,
)
from pyst2110.transmit import max_payload_size

_GIGA = 1_000_000_000


def video(**overrides: Any) -> SdpVideo:
    """1080p50 YCbCr-4:2:2 10-bit, the format the hand computations use."""
    fields: dict[str, Any] = {
        "width": 1920,
        "height": 1080,
        "frame_rate": Fraction(50),
        "depth": 10,
        "sampling": "YCbCr-4:2:2",
    }
    return SdpVideo(**(fields | overrides))


def n_packets(fmt: SdpVideo) -> int:
    """Packets per frame at the largest payload under the format's UDP limit."""
    return packets_per_frame(fmt, choose_payload_size(fmt, max_payload_size(fmt)))


# 1920x1080 4:2:2 10-bit: 4800-octet lines, 1200-octet payloads under the
# Standard UDP Size Limit, so 4 packets a line and 4320 a frame.
_N_1080 = 4320
# 3840x2160: 9600-octet lines, the same 1200-octet payloads, 17280 a frame.
_N_2160 = 17280


def test_the_hand_computed_packet_counts_are_the_geometrys():
    assert n_packets(video()) == _N_1080
    assert n_packets(video(width=3840, height=2160)) == _N_2160


# --- Limits (section 7.1) --------------------------------------------------


def test_narrow_limits_for_1080p50():
    """Section 7.1.2 with N=4320, T_FRAME=1/50: VRX_FULL =
    MAX(INT(1500*8/1500), INT(4320/540)) = 8 and C_MAX =
    MAX(4, INT(4320/(43200 * (1080/1125) * (1/50)))) = MAX(4, 5) = 5."""
    limits = sender_limits(video(), _N_1080)
    assert limits.sender_type == SENDER_TYPE_NARROW
    assert limits.vrx_full == 8
    assert limits.cmax == 5
    assert limits.beta == Fraction(11, 10)
    # 6.6.1: T_DRAIN = (T_FRAME / N_PACKETS) * (1 / beta).
    assert limits.t_drain == Fraction(1, 50 * _N_1080) / Fraction(11, 10)


def test_narrow_linear_limits_for_1080p50():
    """Section 7.1.3 drops R_ACTIVE from the C_MAX divisor:
    MAX(4, INT(4320/864)) = 5, and VRX_FULL is Type N's."""
    limits = sender_limits(video(), _N_1080, sender_type=SENDER_TYPE_NARROW_LINEAR)
    assert limits.vrx_full == 8
    assert limits.cmax == 5


def test_wide_limits_for_1080p50():
    """Section 7.1.4: VRX_FULL = MAX(INT(1500*720/1500), INT(4320/6)) = 720
    and C_MAX = MAX(16, INT(4320/432)) = 16."""
    limits = sender_limits(video(), _N_1080, sender_type=SENDER_TYPE_WIDE)
    assert limits.vrx_full == 720
    assert limits.cmax == 16


def test_narrow_limits_for_2160p5994():
    """The fractional rate: T_FRAME = 1001/60000 and N = 17280 give
    VRX_FULL = INT(38.36) = 38 and C_MAX = INT(24.97) = 24."""
    fmt = video(width=3840, height=2160, frame_rate=Fraction(60000, 1001))
    limits = sender_limits(fmt, _N_2160)
    assert limits.vrx_full == 38
    assert limits.cmax == 24


def test_narrow_linear_limits_for_2160p5994():
    """INT(17280/(43200 * 1001/60000)) = INT(23.976) = 23: the R_ACTIVE
    divisor is exactly what separates this from Type N's 24."""
    fmt = video(width=3840, height=2160, frame_rate=Fraction(60000, 1001))
    limits = sender_limits(fmt, _N_2160, sender_type=SENDER_TYPE_NARROW_LINEAR)
    assert limits.cmax == 23


def test_wide_cmax_is_undefined_at_2160p5994_rates():
    """Section 7.1.4: "The CMAX definition above shall only apply to streams
    of less than 900000 packets/second", and 17280 * 60000/1001 is above it.
    The standard defines no bound there, so none is invented."""
    fmt = video(width=3840, height=2160, frame_rate=Fraction(60000, 1001))
    limits = sender_limits(fmt, _N_2160, sender_type=SENDER_TYPE_WIDE)
    assert limits.cmax is None
    assert limits.vrx_full == 3452  # INT(17280 * 200 / 1001)


def test_the_extended_udp_size_limit_scales_vrx_full_down():
    """Sections 7.1.2's MAXUDP "shall be assumed to be 1500 if the Standard
    UDP Size Limit is in use" and is the declared MAXUDP otherwise. At
    MAXUDP=8960 a 1080p50 frame is 1080 whole-line packets, so VRX_FULL =
    MAX(INT(12000/8960), INT(1080/540)) = 2 and C_MAX floors at 4."""
    fmt = video(max_udp=8960)
    count = n_packets(fmt)
    assert count == 1080
    limits = sender_limits(fmt, count)
    assert limits.vrx_full == 2
    assert limits.cmax == 4


def test_the_sender_type_defaults_to_the_formats_own():
    fmt = video(sender_type=SENDER_TYPE_WIDE)
    assert sender_limits(fmt, _N_1080).sender_type == SENDER_TYPE_WIDE
    resolved = sender_limits(fmt, _N_1080, sender_type=SENDER_TYPE_NARROW)
    assert resolved.sender_type == SENDER_TYPE_NARROW


def test_a_sender_type_the_standard_does_not_define_is_refused():
    with pytest.raises(ValueError, match="TP"):
        sender_limits(video(), _N_1080, sender_type="2110TPX")


# --- Read schedules (sections 6.2-6.4) -------------------------------------


def test_the_gapped_progressive_schedule_for_1080p50():
    """Section 6.3.2: R_ACTIVE = 1080/1125, T_RS = (T_FRAME * R_ACTIVE)/N
    = 1/225000 s, and TRO_DEFAULT = (43/1125) * T_FRAME at heights >= 1080."""
    schedule = read_schedule(video(), _N_1080)
    assert schedule.gapped is True
    assert schedule.r_active == Fraction(1080, 1125)
    assert schedule.t_frame == Fraction(1, 50)
    assert schedule.t_rs == Fraction(1, 225000)
    assert schedule.tr_offset == Fraction(43, 1125) / 50
    assert schedule.n_packets == _N_1080


def test_the_gapped_progressive_offset_below_1080_lines():
    """Section 6.3.2: TRO_DEFAULT = (28/750) * T_FRAME under 1080 lines."""
    fmt = video(width=1280, height=720)
    schedule = read_schedule(fmt, n_packets(fmt))
    assert schedule.tr_offset == Fraction(28, 750) / 50


def test_the_linear_schedule_spreads_over_the_whole_frame():
    """Section 6.4: T_RS = T_FRAME/N, and TRO_DEFAULT "as defined in the
    gapped model of section 6.3" for the same raster."""
    schedule = read_schedule(video(), _N_1080, sender_type=SENDER_TYPE_NARROW_LINEAR)
    assert schedule.gapped is False
    assert schedule.t_rs == Fraction(1, 50 * _N_1080)
    assert schedule.tr_offset == Fraction(43, 1125) / 50
    wide = read_schedule(video(), _N_1080, sender_type=SENDER_TYPE_WIDE)
    assert wide.t_rs == schedule.t_rs


def test_read_times_walk_the_gapped_progressive_frame():
    schedule = read_schedule(video(), _N_1080)
    times = read_times(schedule, 0, np.array([0, 1, _N_1080 - 1]))
    # TR_OFFSET = 43/56250 s floors to 764444 ns; j=4319 lands at
    # (43/56250 + 4319/225000) s = 0.01996 s exactly.
    assert times.tolist() == [764_444, 768_888, 19_960_000]
    assert times.dtype == np.int64


def test_the_datum_walks_frames_from_the_smpte_epoch():
    """Section 6.2: T_VD = N * T_FRAME + TR_OFFSET, origin at the SMPTE
    epoch — so frame 7 of a 50 Hz stream sits at 140 ms and change."""
    schedule = read_schedule(video(), _N_1080)
    assert video_datum(schedule, 0) == 764_444
    assert video_datum(schedule, 7) == 7 * 20_000_000 + 764_444


def test_a_signalled_tr_offset_replaces_the_default():
    """Section 6.2: receivers assume TRO_DEFAULT only "if this parameter is
    not present"."""
    schedule = read_schedule(video(tr_offset_us=100), _N_1080)
    assert schedule.tr_offset == Fraction(100, 1_000_000)
    override = read_schedule(video(tr_offset_us=100), _N_1080, tr_offset_us=250)
    assert override.tr_offset == Fraction(250, 1_000_000)


def test_the_gapped_interlaced_schedule_for_1080i25():
    """Section 6.3.3 and Table 1, 1125-line row: R_ACTIVE = HEIGHT/1125,
    TRO_DEFAULT = (INT((1125-1080)/2)/1125) * T_FRAME = (22/1125) * T_FRAME,
    T_LINE = T_FRAME/1125."""
    fmt = video(frame_rate=Fraction(25), interlaced=True)
    schedule = read_schedule(fmt, _N_1080)
    assert schedule.r_active == Fraction(1080, 1125)
    assert schedule.tr_offset == Fraction(22, 1125) / 25
    assert schedule.t_rs == Fraction(1, 112500)
    assert schedule.t_line == Fraction(1, 25 * 1125)


def test_interlaced_read_times_gap_between_the_fields():
    """Section 6.3.3: packet N/2 reads at T_FRAME/2 + T_LINE/2 past the
    datum — 20.8 ms exactly for 1080i25 — while packet N/2 - 1 ends the
    first field's ramp."""
    fmt = video(frame_rate=Fraction(25), interlaced=True)
    schedule = read_schedule(fmt, _N_1080)
    j = np.array([0, _N_1080 // 2 - 1, _N_1080 // 2])
    times = read_times(schedule, 0, j)
    assert times.tolist() == [782_222, 19_973_333, 20_800_000]


@pytest.mark.parametrize(
    ("height", "rate", "expected"),
    [
        # Table 1: 625-line systems, HEIGHT=576: (INT(49/2) + 2)/625 = 26/625.
        (576, Fraction(25), Fraction(26, 625) / 25),
        # 525-line systems, HEIGHT=487: (INT(38/2) + 1)/525 = 20/525.
        (487, Fraction(30000, 1001), Fraction(20, 525) * Fraction(1001, 30000)),
    ],
)
def test_the_standard_definition_interlaced_offsets_match_table_1(
    height: int, rate: Fraction, expected: Fraction
):
    fmt = video(width=720, height=height, frame_rate=rate, interlaced=True)
    schedule = read_schedule(fmt, n_packets(fmt))
    assert schedule.tr_offset == expected
    assert schedule.r_active == Fraction(height, 525 if height == 487 else 625)


def test_an_interlaced_raster_above_1125_lines_has_no_gapped_schedule():
    """Table 1 lists 525-, 625- and 1125-line interlaced systems and no
    other; a raster none of them holds has no row to read from."""
    fmt = video(width=3840, height=2160, frame_rate=Fraction(25), interlaced=True)
    with pytest.raises(ValueError, match=r"[Ii]nterlaced"):
        read_schedule(fmt, _N_2160)


# --- The senders the tests emit --------------------------------------------

# A frame index deep enough that the times are 2020s-era nanoseconds since
# the SMPTE epoch, which is what a bench capture carries: the arithmetic has
# to be exact there, not merely near zero.
_EPOCH_FRAME = 85_000_000_000


def ideal_times(schedule: Schedule, frames: int) -> np.ndarray:
    """Every packet emitted exactly at its read time, for whole frames."""
    j = np.arange(schedule.n_packets)
    return np.concatenate(
        [read_times(schedule, _EPOCH_FRAME + k, j) for k in range(frames)]
    )


def end_of_frame_markers(count: int, frames: int) -> np.ndarray:
    marker = np.zeros(count * frames, dtype=np.bool_)
    marker[count - 1 :: count] = True
    return marker


# --- The network compatibility model (section 6.6.1) -----------------------


def c_inst_reference(times: np.ndarray, t_drain: Fraction, origin: str) -> list[int]:
    """Section 6.6.1 as a per-packet loop: an infinite bucket, drained at
    every multiple of T_DRAIN "since the SMPTE Epoch", never below empty.
    Reported at the instant each packet has entered."""
    drain_ns = t_drain * _GIGA
    base = 0 if origin == "epoch" else int(times[0])
    level, drained_before = 0, None
    out = []
    for t in times.tolist():
        drains = (Fraction(t - base) / drain_ns).__floor__()
        if drained_before is not None:
            level = max(0, level - (drains - drained_before))
        drained_before = drains
        level += 1
        out.append(level)
    return out


def test_an_ideal_narrow_sender_never_queues_a_second_packet():
    """T_RS in the active period exceeds T_DRAIN by exactly beta, so every
    packet is drained before the next arrives."""
    schedule = read_schedule(video(), _N_1080)
    limits = sender_limits(video(), _N_1080)
    fullness = c_inst(ideal_times(schedule, 3), limits.t_drain)
    assert fullness.max() == 1
    assert fullness.dtype == np.int64


def test_a_burst_fills_the_bucket_by_its_own_size():
    limits = sender_limits(video(), _N_1080)
    times = np.full(100, 1_700_000_000 * _GIGA, dtype=np.int64)
    assert c_inst(times, limits.t_drain).tolist() == list(range(1, 101))


def test_the_two_drain_origins_the_field_uses():
    """Epoch-anchored is section 6.6.1's own grid; first-packet is what EBU
    LIST measures. With a 4 ns drain, packets at 10 and 13 straddle a grid
    instant at 12 but not one of their own making."""
    t_drain = Fraction(4, _GIGA)
    times = np.array([10, 13], dtype=np.int64)
    assert c_inst(times, t_drain, origin="epoch").tolist() == [1, 1]
    assert c_inst(times, t_drain, origin="first").tolist() == [1, 2]


@pytest.mark.parametrize("origin", ["epoch", "first"])
def test_c_inst_agrees_with_the_per_packet_reference(origin: str):
    rng = np.random.default_rng(2110)
    limits = sender_limits(video(), _N_1080)
    base = 1_700_000_000 * _GIGA
    steps = rng.integers(0, 9_000, size=5_000)
    times = base + np.cumsum(steps).astype(np.int64)
    computed = c_inst(times, limits.t_drain, origin=origin)
    assert computed.tolist() == c_inst_reference(times, limits.t_drain, origin)


def test_timestamps_out_of_order_are_refused():
    """Both buckets take emission instants in emission order; a capture that
    reordered them describes a different sender."""
    limits = sender_limits(video(), _N_1080)
    with pytest.raises(ValueError, match="order"):
        c_inst(np.array([10, 5], dtype=np.int64), limits.t_drain)


# --- The virtual receiver buffer model (section 6.6.2) ---------------------


def segment_read_times(schedule: Schedule, frame: int) -> list[Fraction]:
    """Every T_PRj of one frame, in seconds, straight from sections 6.3-6.4."""
    tvd = frame * schedule.t_frame + schedule.tr_offset
    times = []
    for j in range(schedule.n_packets):
        if schedule.gapped and schedule.interlaced:
            if j < Fraction(schedule.n_packets, 2):
                times.append(j * schedule.t_rs + tvd)
            else:
                assert schedule.t_line is not None
                times.append(
                    schedule.t_frame / 2
                    + schedule.t_line / 2
                    + (j - Fraction(schedule.n_packets, 2)) * schedule.t_rs
                    + tvd
                )
        else:
            times.append(j * schedule.t_rs + tvd)
    return times


def vrx_reference(
    times: np.ndarray, starts: np.ndarray, schedule: Schedule
) -> list[int]:
    """Section 6.6.2 as a per-packet loop, on the ideal datum: packets enter
    at emission, packet j leaves at T_PRj, and the level is read with each
    entering packet counted. Reads stop at the frame's last packet, and a
    read has happened once its truncated-nanosecond instant has passed —
    the grid ``read_times`` realises the schedule on."""
    out = []
    read_times_s: list[Fraction] = []
    entered = 0
    for t, start in zip(times.tolist(), starts.tolist(), strict=True):
        seconds = Fraction(t, _GIGA)
        if start:
            frame = round((seconds - schedule.tr_offset) / schedule.t_frame)
            read_times_s = segment_read_times(schedule, frame)
            entered = 0
        entered += 1
        drained = sum(1 for read in read_times_s if (read * _GIGA).__floor__() <= t)
        out.append(entered - drained)
    return out


def test_an_ideal_narrow_sender_holds_the_virtual_buffer_at_zero():
    schedule = read_schedule(video(), _N_1080)
    times = ideal_times(schedule, 3)
    starts = np.zeros(times.size, dtype=np.bool_)
    starts[::_N_1080] = True
    result = vrx(schedule, times, starts)
    assert result.per_packet.min() == 0
    assert result.per_packet.max() == 0
    assert result.datum_delta_ns.tolist() == [0, 0, 0]


def test_a_slightly_early_sender_keeps_one_packet_outstanding():
    schedule = read_schedule(video(), _N_1080)
    times = ideal_times(schedule, 2) - 1_000
    starts = np.zeros(times.size, dtype=np.bool_)
    starts[::_N_1080] = True
    result = vrx(schedule, times, starts)
    assert result.per_packet.max() == 1
    assert result.per_packet.min() == 1
    assert result.datum_delta_ns.tolist() == [-1_000, -1_000]


def test_a_whole_frame_burst_overflows_by_the_frame_size():
    """Every packet at the datum: one read instant has passed, so the
    buffer holds all that remain."""
    schedule = read_schedule(video(), _N_1080)
    times = np.full(_N_1080, video_datum(schedule, _EPOCH_FRAME), dtype=np.int64)
    starts = np.zeros(_N_1080, dtype=np.bool_)
    starts[0] = True
    result = vrx(schedule, times, starts)
    assert result.per_packet.max() == _N_1080 - 1


def test_a_late_sender_underflows_below_zero():
    """Section 6.6.2: packet j not in the bucket by T_PRj is an underflow;
    three read spacings late reads three packets short. 4800 packets a
    frame makes T_RS a whole 4000 ns, so "three spacings" is exact."""
    schedule = read_schedule(video(), 4800)
    times = ideal_times(schedule, 1) + 3 * 4000
    starts = np.zeros(times.size, dtype=np.bool_)
    starts[0] = True
    result = vrx(schedule, times, starts)
    assert result.per_packet.min() == -3


def test_an_interlaced_ideal_sender_holds_the_buffer_at_zero():
    fmt = video(frame_rate=Fraction(25), interlaced=True)
    schedule = read_schedule(fmt, _N_1080)
    times = ideal_times(schedule, 2)
    starts = np.zeros(times.size, dtype=np.bool_)
    starts[::_N_1080] = True
    result = vrx(schedule, times, starts)
    assert result.per_packet.max() == 0
    assert result.per_packet.min() == 0


@pytest.mark.parametrize("interlaced", [False, True])
def test_vrx_agrees_with_the_per_packet_reference(interlaced: bool):
    """A small frame — 270 packets — keeps the quadratic reference loop
    affordable; the vectorised pass has no such dependence on the count."""
    count = 270
    rng = np.random.default_rng(21102)
    fmt = video(frame_rate=Fraction(25), interlaced=True) if interlaced else video()
    schedule = read_schedule(fmt, count)
    jitter = rng.integers(-200_000, 200_000, size=3 * count)
    times = np.sort(ideal_times(schedule, 3) + jitter)
    starts = np.zeros(times.size, dtype=np.bool_)
    starts[::count] = True
    computed = vrx(schedule, times, starts)
    assert computed.per_packet.tolist() == vrx_reference(times, starts, schedule)


def test_the_first_packet_datum_forgives_a_constant_phase_error():
    """A sender 5 ms late against PTP but perfectly paced: the ideal datum
    reads it as a deep underflow, the first-packet datum as clean — which is
    the difference between measuring alignment and measuring pacing."""
    schedule = read_schedule(video(), _N_1080)
    times = ideal_times(schedule, 2) + 5_000_000
    starts = np.zeros(times.size, dtype=np.bool_)
    starts[::_N_1080] = True
    aligned = vrx(schedule, times, starts)
    assert aligned.per_packet.min() < -1_000
    paced = vrx(schedule, times, starts, datum="first-packet")
    assert paced.per_packet.min() == 0
    assert paced.per_packet.max() == 0


def test_the_rtp_datum_catches_a_timestamp_a_frame_ahead():
    """Section 6.6.2 evaluates the model "using the RTP Clock timebase": a
    sender stamping frame N+1 while transmitting at frame N's time is a
    frame early against its own claim, though its wire pacing looks ideal."""
    schedule = read_schedule(video(), _N_1080)
    times = ideal_times(schedule, 2)
    starts = np.zeros(times.size, dtype=np.bool_)
    starts[::_N_1080] = True
    ticks = 90_000 * schedule.t_frame
    honest = np.repeat(
        [int((_EPOCH_FRAME + k) * ticks) % (1 << 32) for k in range(2)], _N_1080
    ).astype(np.uint32)
    assert (
        vrx(
            schedule, times, starts, datum="rtp", rtp_timestamps=honest
        ).per_packet.max()
        == 0
    )
    ahead = np.repeat(
        [int((_EPOCH_FRAME + k + 1) * ticks) % (1 << 32) for k in range(2)], _N_1080
    ).astype(np.uint32)
    early = vrx(schedule, times, starts, datum="rtp", rtp_timestamps=ahead)
    assert early.per_packet.max() == _N_1080
    assert early.datum_delta_ns[0] < -19_000_000


def test_the_first_packet_must_start_a_frame():
    schedule = read_schedule(video(), _N_1080)
    times = ideal_times(schedule, 1)
    starts = np.zeros(times.size, dtype=np.bool_)
    with pytest.raises(ValueError, match="frame"):
        vrx(schedule, times, starts)


# --- Frame starts from the marker bit --------------------------------------


def test_a_frame_starts_after_a_marker():
    marker = np.array([False, False, True, False, True, False])
    assert frame_starts(marker).tolist() == [False] * 3 + [True, False, True]


def test_the_packet_before_the_chunk_can_end_a_frame():
    marker = np.array([False, True, False])
    assert frame_starts(marker, previous=True).tolist() == [True, False, True]


def test_interlaced_frames_start_on_the_first_field():
    """The F bit says which field a packet carries (ST 2110-20 section
    6.1.4), so a frame starts where a field does and F reads first."""
    marker = np.array([True, False, True, False, True, False, True])
    field = np.array([False, True, True, False, False, True, True])
    starts = frame_starts(marker, interlaced=True, field=field, previous=True)
    assert starts.tolist() == [True, False, False, True, False, False, False]


def test_interlaced_frames_without_the_f_bit_pair_the_fields():
    marker = np.array([False, True, False, True, False, True, False])
    starts = frame_starts(marker, interlaced=True, previous=True)
    assert starts.tolist() == [True, False, False, False, True, False, False]


# --- The measurement -------------------------------------------------------


def test_an_ideal_narrow_sender_measures_as_type_n():
    times = ideal_times(read_schedule(video(), _N_1080), 3)
    marker = end_of_frame_markers(_N_1080, 3)
    result = measure(video(), _N_1080, times, marker, previous_marker=True)
    assert result.profile == SENDER_TYPE_NARROW
    assert result.frames == 3
    assert result.packets == 3 * _N_1080
    assert result.c_inst_max == 1
    assert result.vrx_max == 0
    assert result.vrx_min == 0
    assert result.datum_delta_ns.tolist() == [0, 0, 0]
    # The histograms count every packet once.
    assert result.c_inst_histogram.sum() == result.packets
    assert result.vrx_histogram.sum() == result.packets
    assert result.vrx_histogram[0 - result.vrx_histogram_start] == result.packets


def test_a_sender_bursting_a_whole_frame_measures_as_non_compliant():
    schedule = read_schedule(video(), _N_1080)
    times = np.repeat(
        [video_datum(schedule, _EPOCH_FRAME + k) for k in range(2)], _N_1080
    ).astype(np.int64)
    marker = end_of_frame_markers(_N_1080, 2)
    result = measure(video(), _N_1080, times, marker, previous_marker=True)
    assert result.profile == ""
    assert result.c_inst_max > 16
    assert result.vrx_max > 720


def test_a_wide_sender_measures_as_wide_and_not_narrow():
    """Packets each 15 read spacings early: the virtual buffer rides at 15,
    past Type NL's 8 and inside Type W's 720, while C_INST stays at 1."""
    fmt = video(sender_type=SENDER_TYPE_NARROW_LINEAR)
    schedule = read_schedule(fmt, _N_1080)
    t_rs_ns = int(schedule.t_rs * _GIGA)
    times = ideal_times(schedule, 2) - 15 * t_rs_ns
    marker = end_of_frame_markers(_N_1080, 2)
    result = measure(fmt, _N_1080, times, marker, previous_marker=True)
    assert result.profile == SENDER_TYPE_WIDE
    assert result.vrx_max == 15
    assert result.c_inst_max == 1


def test_an_underflowing_sender_measures_as_non_compliant():
    schedule = read_schedule(video(), 4800)
    times = ideal_times(schedule, 1) + 3 * 4000
    marker = end_of_frame_markers(4800, 1)
    result = measure(video(), 4800, times, marker, previous_marker=True)
    assert result.profile == ""
    assert result.vrx_min == -3


def test_packets_before_the_first_frame_start_are_left_out():
    """Without the marker state before the capture, the first partial frame
    has no datum to measure against, as LIST also holds off."""
    times = ideal_times(read_schedule(video(), _N_1080), 3)
    marker = end_of_frame_markers(_N_1080, 3)
    result = measure(video(), _N_1080, times, marker)
    assert result.frames == 2
    assert result.packets == 2 * _N_1080


def test_a_declared_wide_sender_is_judged_against_wide_alone():
    fmt = video(sender_type=SENDER_TYPE_WIDE)
    schedule = read_schedule(fmt, _N_1080)
    times = ideal_times(schedule, 2)
    marker = end_of_frame_markers(_N_1080, 2)
    result = measure(fmt, _N_1080, times, marker, previous_marker=True)
    assert result.profile == SENDER_TYPE_WIDE


def test_the_declared_limits_ride_along_for_the_report():
    times = ideal_times(read_schedule(video(), _N_1080), 1)
    marker = end_of_frame_markers(_N_1080, 1)
    result = measure(video(), _N_1080, times, marker, previous_marker=True)
    assert result.limits == sender_limits(video(), _N_1080)
    assert isinstance(result.limits, Limits)
