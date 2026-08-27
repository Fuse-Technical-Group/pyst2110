"""The SMPTE ST 2110-21 timing model (SPEC §spec:timing).

Two leaky buckets decide whether a sender's pacing conforms: the Network
Compatibility Model of section 6.6.1 (an infinite bucket drained on a fixed
grid, fullness ``C_INST`` against ``C_MAX``) and the Virtual Receiver Buffer
Model of section 6.6.2 (a bucket of capacity ``VRX_FULL`` drained on the
Packet Read Schedule of sections 6.3 and 6.4). Both are computed here over
whole captures at once — one ``int64`` nanosecond timestamp per packet in,
one bucket level per packet out (§spec:interface-shape).

Nothing here paces a packet or timestamps one. Emission instants come from
whatever captured them; this module says what the standard makes of them
(§spec:scope-boundary).

Time is exact. Schedule constants are :class:`~fractions.Fraction` seconds,
instants are integer nanoseconds since the SMPTE Epoch (ST 2059-1, the PTP
epoch), and every division is a floor over integers — a 60000/1001 rate
never rounds until a result leaves as nanoseconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from pyst2110 import _chunk
from pyst2110.framing import frame_boundaries
from pyst2110.sdp import (
    RTP_CLOCK_RATE,
    SENDER_TYPE_NARROW,
    SENDER_TYPE_WIDE,
    SENDER_TYPES,
    STANDARD_UDP_SIZE_LIMIT,
    SdpVideo,
)

__all__ = [
    "Limits",
    "Measurement",
    "Schedule",
    "Vrx",
    "c_inst",
    "frame_starts",
    "measure",
    "read_schedule",
    "read_times",
    "sender_limits",
    "video_datum",
    "vrx",
]

_GIGA = 1_000_000_000
_MICROSECONDS_PER_SECOND = 1_000_000

# ST 2110-21 section 6.3.2, gapped PRS for progressive images: R_ACTIVE is
# 1080/1125 for every raster, and TRO_DEFAULT switches on the image height.
_PROGRESSIVE_ACTIVE = Fraction(1080, 1125)
_PROGRESSIVE_TALL = 1080
_TRO_TALL = Fraction(43, 1125)
_TRO_SHORT = Fraction(28, 750)

# Section 6.3.3, Table 1: the interlaced systems as (largest height, total
# lines, offset lines added to the halved blanking). R_ACTIVE = HEIGHT/total
# and TRO_DEFAULT = ((INT((total - HEIGHT)/2) + added)/total) * T_FRAME.
#
# The published PDF clips the added term out of the 525- and 625-line rows —
# the cell ends at "INT((total - HEIGHT)/2) +" — so those two values are
# reconstructed to reproduce ST 2110-21:2017's fixed ratios (20/525 at 487
# lines, 26/625 at 576) at the reference heights (SPEC §spec:timing).
_INTERLACED_SYSTEMS = ((487, 525, 1), (576, 625, 2), (1080, 1125, 0))
_FIELDS_PER_FRAME = 2

# Section 7.1: beta = 1.10 for every sender type, and the type parameters of
# 7.1.2 (Type N), 7.1.3 (Type NL) and 7.1.4 (Type W). The VRX_FULL numerators
# assume MAXUDP = 1500 under the Standard UDP Size Limit; a declared MAXUDP
# scales them (each clause's closing paragraph).
_BETA = Fraction(11, 10)
_ASSUMED_MAXUDP = 1500
_NARROW_VRX_OCTETS = 1500 * 8
_WIDE_VRX_OCTETS = 1500 * 720
_NARROW_VRX_DIVISOR = 27000
_WIDE_VRX_DIVISOR = 300
_NARROW_CMAX_FLOOR = 4
_WIDE_CMAX_FLOOR = 16
_NARROW_CMAX_DIVISOR = 43200
_WIDE_CMAX_DIVISOR = 21600
# Section 7.1.4: "The CMAX definition above shall only apply to streams of
# less than 900000 packets/second."
_WIDE_CMAX_PACKET_RATE = 900_000

# RFC 3550 section 5.1: the RTP timestamp wraps at thirty-two bits.
_TIMESTAMP_MODULUS = 1 << 32

# The exact integer passes multiply a nanosecond delta by a schedule
# denominator; past this the product would leave int64, so the capture is
# refused by name rather than wrapped in silence.
_INT64_LIMIT = 1 << 63


@dataclass(frozen=True)
class Schedule:
    """One stream's Packet Read Schedule, sections 6.2-6.4.

    All constants are exact :class:`~fractions.Fraction` seconds. ``gapped``
    says which of the two PRS shapes this is; ``t_line`` is Table 1's
    T_LINE and carried only by the gapped interlaced schedule, whose second
    field's reads sit ``t_frame/2 + t_line/2`` past the first's.
    """

    t_frame: Fraction
    n_packets: int
    gapped: bool
    interlaced: bool
    #: Section 6.3's ratio of active to total time; 1 for the linear PRS,
    #: whose reads span the whole frame.
    r_active: Fraction
    t_rs: Fraction
    tr_offset: Fraction
    t_line: Fraction | None = None


@dataclass(frozen=True)
class Limits:
    """One sender type's traffic-shape parameters, section 7.1."""

    sender_type: str
    #: Largest permitted C_INST — ``None`` where the standard defines no
    #: bound: Type W at 900000 packets/second or more (section 7.1.4).
    cmax: int | None
    vrx_full: int
    beta: Fraction
    #: Section 6.6.1's drain interval, in seconds.
    t_drain: Fraction


@dataclass(frozen=True)
class Vrx:
    """The Virtual Receiver Buffer level, per packet, and each frame's phase."""

    #: The bucket level at each packet's own emission instant, that packet
    #: counted. Negative is an underflow: the read schedule wanted a packet
    #: the sender had not yet emitted (section 6.6.2).
    per_packet: NDArray[np.int64]
    #: Per frame: the first packet's emission instant less the frame's datum,
    #: in nanoseconds. Positive is a late sender.
    datum_delta_ns: NDArray[np.int64]


@dataclass(frozen=True)
class Measurement:
    """What one capture says about its sender, against section 7.1's limits."""

    packets: int
    frames: int
    c_inst_max: int
    vrx_max: int
    vrx_min: int
    #: The sender type the measurement satisfies: the declared type where the
    #: peaks sit inside its limits, ``2110TPW`` where only Wide's hold them,
    #: and ``""`` where none does — or where the buffer underflowed.
    profile: str
    #: The declared type's limits, for the report beside the peaks.
    limits: Limits
    #: Packet counts by C_INST value, indexed by the value itself.
    c_inst_histogram: NDArray[np.int64]
    #: Packet counts by VRX level, indexed from ``vrx_histogram_start``.
    vrx_histogram: NDArray[np.int64]
    vrx_histogram_start: int
    datum_delta_ns: NDArray[np.int64]


def read_schedule(
    video: SdpVideo,
    n_packets: int,
    *,
    sender_type: str | None = None,
    tr_offset_us: int | None = None,
) -> Schedule:
    """The Packet Read Schedule for this format and sender type.

    ``n_packets`` is the packets per frame —
    :func:`pyst2110.geometry.packets_per_frame` for a sender built here, or
    the count a capture shows. ``sender_type`` defaults to the format's own
    ``TP``; Type N reads gapped (section 6.3) and Types NL and W linear
    (section 6.4). ``tr_offset_us`` overrides the format's ``TROFF``; with
    neither, TRO_DEFAULT is computed from the raster (sections 6.3.2, 6.3.3).

    The gapped PRS is defined only for rasters and rates from the ITU-R
    recommendations section 6.3.1 lists. That ancestry is not checkable from
    a width and a height, so it is not checked — except the one case with no
    Table 1 row at all, an interlaced raster above 1125 total lines, which
    raises.
    """
    kind = _sender_type(video, sender_type)
    if n_packets <= 0:
        raise ValueError(f"a frame of {n_packets} packets is not a stream")
    t_frame = 1 / video.frame_rate
    gapped = kind == SENDER_TYPE_NARROW

    # Section 6.4: the linear model's TRO_DEFAULT is the gapped model's for
    # the same raster, so both shapes need the gapped parameters.
    r_active, tro_default, _ = _gapped_parameters(video)
    override = video.tr_offset_us if tr_offset_us is None else tr_offset_us
    if override is None:
        tr_offset = tro_default * t_frame
    else:
        tr_offset = Fraction(override, _MICROSECONDS_PER_SECOND)
    if not 0 <= tr_offset < t_frame:
        raise ValueError(
            f"a TR_OFFSET of {tr_offset} s is not inside the {t_frame} s "
            f"frame period, which ST 2110-21 section 6.2 requires"
        )

    if gapped:
        t_rs = t_frame * r_active / n_packets
    else:
        r_active = Fraction(1)
        t_rs = t_frame / n_packets
    return Schedule(
        t_frame=t_frame,
        n_packets=n_packets,
        gapped=gapped,
        interlaced=video.interlaced,
        r_active=r_active,
        t_rs=t_rs,
        tr_offset=tr_offset,
        t_line=t_frame / _total_lines(video) if gapped and video.interlaced else None,
    )


def sender_limits(
    video: SdpVideo, n_packets: int, *, sender_type: str | None = None
) -> Limits:
    """Section 7.1's parameters for this format and sender type.

    MAXUDP in the VRX_FULL expressions is 1500 under the Standard UDP Size
    Limit and the declared ``MAXUDP`` otherwise, as each clause's closing
    paragraph directs — not the 1460-octet limit itself.
    """
    kind = _sender_type(video, sender_type)
    if n_packets <= 0:
        raise ValueError(f"a frame of {n_packets} packets is not a stream")
    t_frame = 1 / video.frame_rate
    maxudp = (
        _ASSUMED_MAXUDP if video.max_udp == STANDARD_UDP_SIZE_LIMIT else video.max_udp
    )

    cmax: int | None
    if kind == SENDER_TYPE_WIDE:
        # Section 7.1.4.
        vrx_full = max(
            _WIDE_VRX_OCTETS // maxudp,
            math.floor(n_packets / (_WIDE_VRX_DIVISOR * t_frame)),
        )
        if n_packets / t_frame < _WIDE_CMAX_PACKET_RATE:
            cmax = max(
                _WIDE_CMAX_FLOOR,
                math.floor(n_packets / (_WIDE_CMAX_DIVISOR * t_frame)),
            )
        else:
            cmax = None
    else:
        # Sections 7.1.2 and 7.1.3 share VRX_FULL; their C_MAX divisors
        # differ by exactly the gapped model's R_ACTIVE.
        vrx_full = max(
            _NARROW_VRX_OCTETS // maxudp,
            math.floor(n_packets / (_NARROW_VRX_DIVISOR * t_frame)),
        )
        divisor = _NARROW_CMAX_DIVISOR * t_frame
        if kind == SENDER_TYPE_NARROW:
            r_active, _, _ = _gapped_parameters(video)
            divisor *= r_active
        cmax = max(_NARROW_CMAX_FLOOR, math.floor(n_packets / divisor))

    # Section 6.6.1: T_DRAIN = (T_FRAME / N_PACKETS) * (1 / beta).
    return Limits(
        sender_type=kind,
        cmax=cmax,
        vrx_full=vrx_full,
        beta=_BETA,
        t_drain=t_frame / n_packets / _BETA,
    )


def video_datum(schedule: Schedule, frame_index: int) -> int:
    """T_VD for one frame: N * T_FRAME + TR_OFFSET, in nanoseconds.

    Section 6.2, with its origin at the SMPTE Epoch of ST 2059-1 — the PTP
    epoch, on the TAI timescale. Truncated to the nanosecond.
    """
    return math.floor((frame_index * schedule.t_frame + schedule.tr_offset) * _GIGA)


def read_times(
    schedule: Schedule, frame_index: int, j: NDArray[np.integer[Any]]
) -> NDArray[np.int64]:
    """T_PRj for packets ``j`` of one frame, in nanoseconds since the epoch.

    Sections 6.3.2 and 6.4: ``j * T_RS`` past the datum. Section 6.3.3 for
    the gapped interlaced schedule: the second field's packets sit a further
    ``T_FRAME/2 + T_LINE/2`` on, counted from ``j - N_PACKETS/2``. Each
    instant is truncated to the nanosecond, as :func:`video_datum` is, so an
    ideal sender's emission never lands before its own read.

    Array in, array out (§spec:interface-shape); a transmitter deriving a
    departure grid and the buffer model below share this one definition.
    """
    positions = np.asarray(j, dtype=np.int64)
    if positions.size and not (
        int(positions.min()) >= 0 and int(positions.max()) < schedule.n_packets
    ):
        raise ValueError(
            f"a packet index outside 0..{schedule.n_packets - 1} names no "
            f"read instant in this schedule"
        )
    datum_s = frame_index * schedule.t_frame + schedule.tr_offset
    out = np.empty(positions.shape, dtype=np.int64)
    for offset, first, count in _segments(schedule):
        inside = (positions >= first) & (positions < first + count)
        out[inside] = _floor_multiply(
            positions[inside] - first,
            schedule.t_rs * _GIGA,
            (datum_s + offset) * _GIGA,
        )
    return out


def c_inst(
    timestamps_ns: NDArray[np.integer[Any]],
    t_drain: Fraction,
    *,
    origin: Literal["epoch", "first"] = "epoch",
) -> NDArray[np.int64]:
    """The Network Compatibility Model's bucket level at each packet.

    Section 6.6.1: packets enter an infinite bucket at their emission
    instants — ``timestamps_ns``, nanoseconds since the SMPTE epoch, in
    emission order — and one packet leaves at every multiple of ``t_drain``
    since the epoch, if one is present. The level is read with the entering
    packet counted, so it is never below one; its peak against
    :attr:`Limits.cmax` is the compliance question.

    ``origin`` places the drain grid: ``"epoch"`` is the standard's — "the
    specific instant of the bucket draining the packet is N * TDRAIN seconds
    since the SMPTE Epoch" — and ``"first"`` anchors it to the first packet,
    which is what EBU LIST measures and the only choice where timestamps are
    not PTP-locked. The bucket is empty before the first packet.

    One vectorised pass: with ``d`` the drain-grid index at each packet and
    ``q = arange(n) - d``, the level is ``1 + q - running_minimum(q)`` — the
    reflected walk in closed form. O(n) time, no Python loop.
    """
    if origin not in ("epoch", "first"):
        raise ValueError(f"{origin!r} is not a drain origin: 'epoch' or 'first'")
    times = _emission_order(timestamps_ns)
    if times.size == 0:
        return np.zeros(0, dtype=np.int64)
    period = t_drain * _GIGA
    base = int(times[0])
    phase = Fraction(base) / period if origin == "epoch" else Fraction(0)
    drains = _floor_divide(times - base, period, phase)
    walk = np.arange(times.size, dtype=np.int64) - drains
    fullness: NDArray[np.int64] = 1 + walk - np.minimum.accumulate(walk)
    return fullness


def frame_starts(
    marker: NDArray[np.bool_],
    *,
    interlaced: bool = False,
    field: NDArray[np.bool_] | None = None,
    previous: bool = False,
) -> NDArray[np.bool_]:
    """Which packets begin a frame, from the RTP marker column.

    The packet after a marker's rising edge begins the next marked unit
    (§spec:rtp) — a frame progressive, a field interlaced. ``previous`` is
    the marker bit of the packet before this chunk; without it the first
    packet cannot be known to start anything and is not claimed to.

    Interlaced, a frame is two units. ``field`` — the F bit of
    :func:`pyst2110.payload.parse_payload_headers` — says which unit starts
    a frame: the one whose first packet reads first-field. Without it the
    units are paired in arrival order from the first unit start, which is
    wrong when the capture opens on a second field; pass the F bit where the
    stream carries one.
    """
    flags = _chunk.per_packet(marker, "marker", dtype=np.bool_)
    if flags.size == 0:
        return np.zeros(0, dtype=np.bool_)
    ends = frame_boundaries(flags, previous=previous)
    starts: NDArray[np.bool_] = np.concatenate(([previous], ends[:-1]))
    if not interlaced:
        return starts
    if field is not None:
        first = _chunk.per_packet(
            field, "field flag", count=int(flags.size), plural="field flags"
        )
        paired: NDArray[np.bool_] = starts & ~first.astype(np.bool_)
        return paired
    keep = np.flatnonzero(starts)[::_FIELDS_PER_FRAME]
    paired = np.zeros(flags.shape, dtype=np.bool_)
    paired[keep] = True
    return paired


def vrx(
    schedule: Schedule,
    timestamps_ns: NDArray[np.integer[Any]],
    starts: NDArray[np.bool_],
    *,
    datum: Literal["ideal", "rtp", "first-packet"] = "ideal",
    rtp_timestamps: NDArray[np.integer[Any]] | None = None,
) -> Vrx:
    """The Virtual Receiver Buffer level at each packet.

    Section 6.6.2: packet j enters the bucket at its emission instant and
    leaves at T_PRj; the level is read with the entering packet counted, so
    an ideal sender reads zero. Reads before a frame's datum and after its
    last packet do not drain — the schedule defines exactly ``n_packets``
    read instants per frame. A negative level is an underflow: the schedule
    read a packet the sender had not emitted.

    ``starts`` flags each frame's first packet (:func:`frame_starts`), and
    the first packet of the capture is required to be one — the model has no
    datum for a partial leading frame, so the caller trims to a frame start
    first, as :func:`measure` does.

    ``datum`` anchors each frame's T_VD:

    - ``"ideal"``: the nearest ``N * T_FRAME + TR_OFFSET`` to the frame's
      first packet — section 6.2's grid, assuming the sender is within half
      a frame of its slot.
    - ``"rtp"``: N recovered from each frame's RTP timestamp
      (``rtp_timestamps``, packet-aligned), which is the RTP Clock timebase
      the section evaluates the model on. This is the anchor that catches a
      sender misaligned to its own media clock by whole frames, at the cost
      of trusting a 90 kHz stamp quantised to 11 microseconds.
    - ``"first-packet"``: the first frame's first packet, advancing by
      T_FRAME per observed frame — EBU LIST's alternative, which measures
      pacing alone and forgives any constant phase error against PTP.
    """
    times = _emission_order(timestamps_ns)
    flags = _chunk.per_packet(
        starts, "frame start", count=int(times.size), plural="frame starts"
    ).astype(np.bool_)
    if times.size == 0:
        return Vrx(
            per_packet=np.zeros(0, dtype=np.int64),
            datum_delta_ns=np.zeros(0, dtype=np.int64),
        )
    if not flags[0]:
        raise ValueError(
            "the capture does not open on a frame start; trim to one first"
        )
    start_index = np.flatnonzero(flags)
    frame_of = np.cumsum(flags) - 1
    within = np.arange(times.size, dtype=np.int64) - start_index[frame_of]
    first_times = times[start_index]
    datums, remainders, den = _datums(
        schedule, first_times, datum, rtp_timestamps, start_index
    )

    # Read k of a segment has passed a packet at integer nanosecond t when
    # its truncated instant has: floor(T_VD + offset + k*T_RS) <= t, which is
    # T_VD + offset + k*T_RS < t + 1 exactly. The datum arrives split as its
    # floor and a remainder over ``den``, so the whole test scales onto one
    # integer grid. Deltas far outside a segment clip to answers the window's
    # edges already give — no reads before it, all of them after — which is
    # what keeps the products inside int64 at any input.
    deltas = times - datums[frame_of]
    part = remainders[frame_of]
    drained = np.zeros(times.size, dtype=np.int64)
    t_rs_ns = schedule.t_rs * _GIGA
    for offset, _, count in _segments(schedule):
        offset_ns = offset * _GIGA
        low = math.floor(offset_ns) - 1
        high = math.ceil(offset_ns + count * t_rs_ns) + 1
        scale = math.lcm(den, t_rs_ns.denominator, offset_ns.denominator)
        spacing = int(t_rs_ns * scale)
        if (max(abs(low), abs(high)) + 1) * scale >= _INT64_LIMIT // 2:
            raise ValueError("the frame span is too wide for exact 64-bit time")
        windowed = np.clip(deltas, low, high)
        room = (windowed + 1) * scale - part * (scale // den) - int(offset_ns * scale)
        read = -(-room // spacing)  # ceil: how many reads fit strictly below
        drained += np.clip(read, 0, count)
    return Vrx(
        per_packet=within + 1 - drained,
        datum_delta_ns=first_times - datums,
    )


def measure(
    video: SdpVideo,
    n_packets: int,
    timestamps_ns: NDArray[np.integer[Any]],
    marker: NDArray[np.bool_],
    *,
    sender_type: str | None = None,
    tr_offset_us: int | None = None,
    datum: Literal["ideal", "rtp", "first-packet"] = "ideal",
    origin: Literal["epoch", "first"] = "epoch",
    rtp_timestamps: NDArray[np.integer[Any]] | None = None,
    field: NDArray[np.bool_] | None = None,
    previous_marker: bool = False,
) -> Measurement:
    """Run both buckets over a capture and judge the sender against 7.1.

    ``timestamps_ns`` are emission instants in emission order and ``marker``
    the RTP marker column beside them; packets before the first frame start
    carry no datum and are left out of every number reported. The keyword
    arguments pass through to :func:`read_schedule`, :func:`c_inst`,
    :func:`frame_starts` and :func:`vrx`.

    The profile ladder is judged on the declared schedule, as EBU LIST
    judges it: an underflow is non-compliant outright; peaks inside the
    declared type's limits are that type; otherwise peaks inside Type W's
    are ``2110TPW`` — a claim about the numbers, not about Wide's own linear
    schedule, which a Type N declaration was not measured against.
    """
    schedule = read_schedule(
        video, n_packets, sender_type=sender_type, tr_offset_us=tr_offset_us
    )
    declared = sender_limits(video, n_packets, sender_type=sender_type)
    times = _emission_order(timestamps_ns)
    starts = frame_starts(
        marker, interlaced=video.interlaced, field=field, previous=previous_marker
    )
    first = np.flatnonzero(starts)
    if first.size == 0:
        raise ValueError(
            "no frame start in the capture: no marker edge, and no word on "
            "the packet before it (previous_marker)"
        )
    keep = slice(int(first[0]), None)
    times = times[keep]

    fullness = c_inst(times, declared.t_drain, origin=origin)
    buffer = vrx(
        schedule,
        times,
        starts[keep],
        datum=datum,
        rtp_timestamps=None if rtp_timestamps is None else rtp_timestamps[keep],
    )
    c_max = int(fullness.max())
    vrx_max = int(buffer.per_packet.max())
    vrx_min = int(buffer.per_packet.min())
    wide = sender_limits(video, n_packets, sender_type=SENDER_TYPE_WIDE)
    return Measurement(
        packets=int(times.size),
        frames=int(buffer.datum_delta_ns.size),
        c_inst_max=c_max,
        vrx_max=vrx_max,
        vrx_min=vrx_min,
        profile=_profile(declared, wide, c_max, vrx_max, vrx_min),
        limits=declared,
        c_inst_histogram=np.bincount(fullness).astype(np.int64),
        vrx_histogram=np.bincount(buffer.per_packet - vrx_min).astype(np.int64),
        vrx_histogram_start=vrx_min,
        datum_delta_ns=buffer.datum_delta_ns,
    )


def _profile(
    declared: Limits, wide: Limits, c_max: int, vrx_max: int, vrx_min: int
) -> str:
    """The best sender type the peaks satisfy, or the empty string."""
    if vrx_min < 0:
        return ""
    inside_cmax = declared.cmax is None or c_max <= declared.cmax
    if vrx_max <= declared.vrx_full and inside_cmax:
        return declared.sender_type
    if (
        declared.sender_type != SENDER_TYPE_WIDE
        and vrx_max <= wide.vrx_full
        and (wide.cmax is None or c_max <= wide.cmax)
    ):
        return SENDER_TYPE_WIDE
    return ""


def _sender_type(video: SdpVideo, override: str | None) -> str:
    kind = video.sender_type if override is None else override
    if kind not in SENDER_TYPES:
        raise ValueError(
            f"a TP of {kind!r} is not one of the {list(SENDER_TYPES)} sender "
            f"types ST 2110-21 section 7.1 defines"
        )
    return kind


def _gapped_parameters(video: SdpVideo) -> tuple[Fraction, Fraction, int]:
    """R_ACTIVE, TRO_DEFAULT as a fraction of T_FRAME, and the total lines.

    Sections 6.3.2 (progressive) and 6.3.3 with Table 1 (interlaced and
    PsF). PsF is not distinguishable from progressive in an SDP's bare
    ``interlace`` flag and is not modelled (§road:future).
    """
    if not video.interlaced:
        active = _PROGRESSIVE_ACTIVE
        tall = video.height >= _PROGRESSIVE_TALL
        return active, _TRO_TALL if tall else _TRO_SHORT, 0
    total = _total_lines(video)
    added = next(add for _, lines, add in _INTERLACED_SYSTEMS if lines == total)
    blanking = (total - video.height) // _FIELDS_PER_FRAME
    return (
        Fraction(video.height, total),
        Fraction(blanking + added, total),
        total,
    )


def _total_lines(video: SdpVideo) -> int:
    """Which Table 1 system holds this interlaced raster, by its height."""
    for top, lines, _ in _INTERLACED_SYSTEMS:
        if video.height <= top:
            return lines
    raise ValueError(
        f"no interlaced system in ST 2110-21 Table 1 holds {video.height} "
        f"active lines; the gapped interlaced schedule stops at 1125 total"
    )


def _segments(schedule: Schedule) -> list[tuple[Fraction, int, int]]:
    """The read instants relative to T_VD, as (offset, first j, count) runs.

    Each run's reads sit at ``offset + k * t_rs`` for ``k`` up to ``count``.
    One run for the linear and gapped progressive schedules; two for gapped
    interlaced, the second offset by ``T_FRAME/2 + T_LINE/2`` and indexed
    from ``N_PACKETS/2`` — kept exact, so an odd packet count keeps the
    half-step section 6.3.3's expression gives it.
    """
    if not (schedule.gapped and schedule.interlaced):
        return [(Fraction(0), 0, schedule.n_packets)]
    half = Fraction(schedule.n_packets, _FIELDS_PER_FRAME)
    first = math.ceil(half)
    if schedule.t_line is None:  # pragma: no cover - read_schedule sets it
        raise ValueError("a gapped interlaced schedule carries t_line")
    offset = (
        schedule.t_frame / _FIELDS_PER_FRAME
        + schedule.t_line / _FIELDS_PER_FRAME
        + (first - half) * schedule.t_rs
    )
    return [
        (Fraction(0), 0, first),
        (offset, first, schedule.n_packets - first),
    ]


def _datums(
    schedule: Schedule,
    first_times: NDArray[np.int64],
    datum: str,
    rtp_timestamps: NDArray[np.integer[Any]] | None,
    start_index: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], int]:
    """T_VD per frame on the anchor ``datum`` names, split for exactness.

    Returns the truncated nanosecond datums, the sub-nanosecond remainders
    as numerators, and their common denominator.
    """
    t_frame_ns = schedule.t_frame * _GIGA
    tr_offset_ns = schedule.tr_offset * _GIGA
    base = int(first_times[0])
    if datum == "first-packet":
        return _split_multiply(
            np.arange(first_times.size, dtype=np.int64), t_frame_ns, Fraction(base)
        )
    if datum == "ideal":
        # The nearest datum: floor((t - TR_OFFSET + T_FRAME/2) / T_FRAME).
        frames = _floor_divide(
            first_times - base,
            t_frame_ns,
            (base - tr_offset_ns + t_frame_ns / 2) / t_frame_ns,
        )
    elif datum == "rtp":
        if rtp_timestamps is None:
            raise ValueError("the rtp datum needs the rtp_timestamps column")
        stamps = _chunk.per_packet(rtp_timestamps, "RTP timestamp")[start_index]
        # The estimate from the emission time places the 13-hour wrap; the
        # stamp then places the frame inside it.
        ticks = _floor_divide(
            first_times - base,
            Fraction(_GIGA, RTP_CLOCK_RATE),
            (Fraction(base) - tr_offset_ns) * Fraction(RTP_CLOCK_RATE, _GIGA),
        )
        turns = (ticks - stamps + _TIMESTAMP_MODULUS // 2) // _TIMESTAMP_MODULUS
        unwrapped = stamps + turns * _TIMESTAMP_MODULUS
        per_frame = Fraction(RTP_CLOCK_RATE) * schedule.t_frame
        frames = (unwrapped * (2 * per_frame.denominator) + per_frame.numerator) // (
            2 * per_frame.numerator
        )
    else:
        raise ValueError(f"{datum!r} is not a datum: 'ideal', 'rtp' or 'first-packet'")
    anchor = int(frames[0])
    phase = anchor * schedule.t_frame + schedule.tr_offset
    return _split_multiply(frames - anchor, t_frame_ns, phase * _GIGA)


def _emission_order(timestamps_ns: NDArray[np.integer[Any]]) -> NDArray[np.int64]:
    """One int64 nanosecond instant per packet, non-decreasing.

    Both buckets read emission order off the array order; a capture that
    reordered packets describes a sender that does not exist, so it is
    refused rather than measured.
    """
    times = _chunk.per_packet(timestamps_ns, "timestamp")
    if times.size and np.any(np.diff(times) < 0):
        raise ValueError("timestamps are not in emission order")
    return times


def _floor_multiply(
    counts: NDArray[np.int64], period: Fraction, phase: Fraction
) -> NDArray[np.int64]:
    """``floor(phase + counts * period)`` elementwise, exact."""
    return _split_multiply(counts, period, phase)[0]


def _split_multiply(
    counts: NDArray[np.int64], period: Fraction, phase: Fraction
) -> tuple[NDArray[np.int64], NDArray[np.int64], int]:
    """``phase + counts * period`` as floors, remainders and denominator.

    ``phase`` may be epoch-sized: its whole part is split off as a Python
    integer, so only ``counts`` times the period's scaled numerator has to
    fit in int64 — which is checked, a capture too wide for exactness being
    refused by name rather than wrapped. The remainders come back as
    numerators over the returned denominator.
    """
    lcm = math.lcm(period.denominator, phase.denominator)
    scaled = period.numerator * (lcm // period.denominator)
    offset = phase.numerator * (lcm // phase.denominator)
    whole, part = divmod(offset, lcm)
    peak = int(np.abs(counts).max(initial=0)) * abs(scaled) + part
    if peak >= _INT64_LIMIT:
        raise ValueError("the frame span is too wide for exact 64-bit time")
    total = counts * scaled + part
    return whole + total // lcm, total % lcm, lcm


def _floor_divide(
    deltas: NDArray[np.int64], period: Fraction, phase: Fraction
) -> NDArray[np.int64]:
    """``floor(phase + deltas / period)`` elementwise, exact.

    The same split as :func:`_floor_multiply`, for the other direction:
    ``deltas`` are nanoseconds relative to a nearby base, so multiplying by
    the period's denominator stays inside int64 — checked, not assumed.
    """
    inverse = 1 / period
    lcm = math.lcm(phase.denominator, inverse.denominator)
    each = inverse.numerator * (lcm // inverse.denominator)
    offset = phase.numerator * (lcm // phase.denominator)
    whole, part = divmod(offset, lcm)
    peak = int(np.abs(deltas).max(initial=0)) * each + part
    if peak >= _INT64_LIMIT:
        raise ValueError("the capture is too long for exact 64-bit time")
    result: NDArray[np.int64] = whole + (deltas * each + part) // lcm
    return result
