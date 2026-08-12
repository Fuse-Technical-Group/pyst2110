"""The whole transmit path over one frame (SPEC §spec:transmit-headers).

The roadmap's verify criterion for the transmit slice: an SDP this library
wrote, read back into a format, geometry from that format, headers built for
it, and those headers fed through the receive parse to recover the frame
shape they were built for.

That round trip proves the writer and the reader agree with each other and
nothing about either agreeing with the wire — the hand-written octets in
``test_transmit.py`` and ``test_payload_header.py`` are what tie them to the
documents (§spec:testing). The other half of the criterion, transmitting to a
commercial receiver, needs a NIC this suite does not have.

The format is the consuming runtime's first target: 2160p24 YCbCr-4:2:2
10-bit, which is a 9600-octet line.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from pyst2110 import (
    FrameTracker,
    SdpFlow,
    SdpVideo,
    SequenceTracker,
    choose_payload_size,
    fits_raster,
    format_sdp,
    line_bytes,
    packets_per_frame,
    packets_per_line,
    parse_payload_headers,
    parse_rtp,
    parse_video_format,
    raster_offset,
)
from pyst2110.transmit import PACKET_HEADER_SIZE, FrameHeaders, max_payload_size

_FLOW = SdpFlow("239.100.0.1", 20000, "192.168.100.2")
_2160P24 = SdpVideo(
    width=3840,
    height=2160,
    frame_rate=Fraction(24),
    depth=10,
    sampling="YCbCr-4:2:2",
    colorimetry="BT2020",
)
_SSRC = 0x1234ABCD


def test_the_offer_this_library_writes_describes_the_format_it_was_given():
    """The head of the slice: a transmitter is configured by an offer it
    produces, so the offer has to survive its own reader (§spec:sdp)."""
    assert parse_video_format(format_sdp(_FLOW, _2160P24)) == _2160P24


def test_2160p24_geometry_is_what_the_raster_works_out_to():
    """3840 pixels of 4:2:2 10-bit is 1920 five-octet pgroups: a 9600-octet
    line. Of the 1432 octets the standard UDP limit leaves for sample data,
    1200 is the largest that divides the line — 240 pgroups, the next divisor
    up being 320 — so a frame is 8 packets a line and 17,280 packets.
    """
    video = parse_video_format(format_sdp(_FLOW, _2160P24))
    assert line_bytes(video) == 9600
    assert max_payload_size(video) == 1432
    payload_size = choose_payload_size(video, max_payload_size(video))
    assert payload_size == 1200
    assert payload_size + PACKET_HEADER_SIZE + 8 <= video.max_udp
    assert packets_per_line(video, payload_size) == 8
    assert packets_per_frame(video, payload_size) == 17_280
    # 83% of the datagram is sample data; the rest is what a divisor of the
    # line costs. The next size up would need a 1468-octet datagram.
    assert payload_size / video.max_udp > 0.82


def test_a_built_frame_parses_back_into_descriptors_that_tile_the_raster():
    """The verify criterion. A payload that tiles a line makes the frame one
    contiguous run, so exact starts prove no hole and no overlap at once."""
    video = parse_video_format(format_sdp(_FLOW, _2160P24))
    payload_size = choose_payload_size(video, max_payload_size(video))
    frame = FrameHeaders(video, payload_size, ssrc=_SSRC)
    block = frame.stamp(0)
    sizes = np.full(frame.packets, PACKET_HEADER_SIZE, dtype=np.int64)

    rtp = parse_rtp(block, sizes=sizes)
    payload = parse_payload_headers(block, rtp.payload_offset, sizes=sizes)

    assert not payload.overflowed.any()
    assert payload.segments.tolist() == [1] * frame.packets
    assert payload.field.tolist() == [False] * frame.packets
    assert (payload.length == payload_size).all()

    fits = fits_raster(video, payload.line, payload.offset)
    assert fits.all(), "a frame this library built places every descriptor"
    starts = raster_offset(video, payload.line, payload.offset)
    expected = np.arange(frame.packets, dtype=np.int64) * payload_size
    assert starts.tolist() == expected.tolist()
    assert int(payload.length.sum()) == video.height * line_bytes(video)


def test_the_transmit_offsets_agree_with_where_the_receive_parse_places_them():
    """The two sides of the boundary meet here: what the builder says a packet
    carries is where the parse says the same packet's payload belongs."""
    video = _2160P24
    payload_size = choose_payload_size(video, max_payload_size(video))
    frame = FrameHeaders(video, payload_size, ssrc=_SSRC)
    block = frame.stamp(0)
    sizes = np.full(frame.packets, PACKET_HEADER_SIZE, dtype=np.int64)

    rtp = parse_rtp(block, sizes=sizes)
    payload = parse_payload_headers(block, rtp.payload_offset, sizes=sizes)
    placed = raster_offset(video, payload.line, payload.offset)
    assert frame.frame_offset.tolist() == placed.tolist()


def test_the_rtp_fields_come_back_as_they_were_built():
    video = _2160P24
    frame = FrameHeaders(
        video,
        choose_payload_size(video, max_payload_size(video)),
        payload_type=112,
        ssrc=_SSRC,
    )
    block = frame.stamp(0)
    sizes = np.full(frame.packets, PACKET_HEADER_SIZE, dtype=np.int64)
    rtp = parse_rtp(block, sizes=sizes)

    assert (rtp.version == 2).all()
    assert (rtp.payload_type == 112).all()
    assert (rtp.ssrc == _SSRC).all()
    assert not rtp.padding.any()
    assert not rtp.extension.any()
    assert (rtp.csrc_count == 0).all()
    # The payload header sits directly after the fixed header, no CSRCs.
    assert (rtp.payload_offset == 12).all()
    assert rtp.marker.sum() == 1
    assert bool(rtp.marker[-1])


def test_three_frames_run_on_with_no_loss_and_three_frame_boundaries():
    """17,280 packets a frame wraps the 16-bit RTP field four times over three
    frames, which is what the extended sequence number is for (§spec:rtp)."""
    video = _2160P24
    payload_size = choose_payload_size(video, max_payload_size(video))
    frame = FrameHeaders(video, payload_size, ssrc=_SSRC, initial_sequence=65_000)

    sequences = SequenceTracker()
    frames = FrameTracker()
    sizes = np.full(frame.packets, PACKET_HEADER_SIZE, dtype=np.int64)
    stamps = []
    for index in range(3):
        block = frame.stamp(index)
        rtp = parse_rtp(block, sizes=sizes)
        payload = parse_payload_headers(block, rtp.payload_offset, sizes=sizes)
        sequences.observe(rtp.sequence, payload.extended_sequence)
        frames.observe(rtp.marker)
        stamps.append(int(rtp.timestamp[0]))

    assert sequences.received == 3 * 17_280
    assert sequences.lost == 0
    assert sequences.discontinuities == 0, "the 16-bit wrap read as a gap"
    assert sequences.resyncs == 0
    assert frames.frames == 3
    # 90 kHz over 24 frames a second is 3750 ticks a frame, exactly.
    assert stamps == [0, 3750, 7500]


def test_a_dropped_packet_leaves_a_hole_the_receive_path_can_see():
    """The headers this library builds carry enough for a receiver to detect
    what never arrived — the property the whole pairing exists for."""
    video = _2160P24
    payload_size = choose_payload_size(video, max_payload_size(video))
    frame = FrameHeaders(video, payload_size, ssrc=_SSRC)
    lossy = np.delete(frame.stamp(0), [11, 12], axis=0)
    sizes = np.full(lossy.shape[0], PACKET_HEADER_SIZE, dtype=np.int64)

    rtp = parse_rtp(lossy, sizes=sizes)
    payload = parse_payload_headers(lossy, rtp.payload_offset, sizes=sizes)
    sequences = SequenceTracker()
    sequences.observe(rtp.sequence, payload.extended_sequence)
    assert sequences.lost == 2

    starts = raster_offset(video, payload.line, payload.offset)
    covered = np.zeros(video.height * line_bytes(video) // payload_size, dtype=np.int64)
    covered[starts // payload_size] = 1
    assert int((covered == 0).sum()) == 2


def test_an_interlaced_frame_round_trips_with_its_fields_intact():
    """1080i25: two markers, two timestamps, and rows numbered within a field
    — which is the bound fits_raster applies there (§spec:geometry)."""
    video = SdpVideo(
        width=1920,
        height=1080,
        frame_rate=Fraction(25),
        depth=10,
        sampling="YCbCr-4:2:2",
        interlaced=True,
    )
    payload_size = choose_payload_size(video, max_payload_size(video))
    frame = FrameHeaders(video, payload_size, ssrc=_SSRC)
    block = frame.stamp(0)
    sizes = np.full(frame.packets, PACKET_HEADER_SIZE, dtype=np.int64)

    rtp = parse_rtp(block, sizes=sizes)
    payload = parse_payload_headers(block, rtp.payload_offset, sizes=sizes)

    assert rtp.marker.sum() == 2, "the marker ends each field, not the frame"
    assert payload.field.sum() == frame.packets // 2
    assert payload.line.max() == 539, "rows are numbered within the field"
    assert fits_raster(video, payload.line, payload.offset).all()
    # Section 6.1.3: one timestamp across a field, and the second field is a
    # half frame interval on — 90000 / 50 fields a second.
    assert sorted(set(rtp.timestamp.tolist())) == [0, 1800]


def test_an_odd_interlaced_height_places_every_descriptor_it_builds():
    """Section 6.1.5 gives the temporally first field the extra line, so a
    five-row interlaced frame is three rows then two. The bound the receive
    side applies has to admit the row the transmit side just built."""
    video = SdpVideo(
        width=4,
        height=5,
        frame_rate=Fraction(25),
        depth=10,
        sampling="YCbCr-4:2:2",
        interlaced=True,
    )
    frame = FrameHeaders(video, payload_size=5, ssrc=_SSRC)
    block = frame.stamp(0)
    sizes = np.full(frame.packets, PACKET_HEADER_SIZE, dtype=np.int64)

    rtp = parse_rtp(block, sizes=sizes)
    payload = parse_payload_headers(block, rtp.payload_offset, sizes=sizes)
    assert payload.line.size == 10
    assert fits_raster(video, payload.line, payload.offset).all()


def test_the_emitted_offer_and_the_built_frame_describe_one_flow():
    """End to end: the offer says General Packing Mode and 1200-octet packets
    fit under the limit it declares, which is what makes the pair consistent.
    """
    offer = format_sdp(_FLOW, _2160P24)
    video = parse_video_format(offer)
    payload_size = choose_payload_size(video, max_payload_size(video))
    frame = FrameHeaders(video, payload_size, ssrc=_SSRC)

    assert "PM=2110GPM" in offer
    assert "a=rtpmap:96 raw/90000" in offer
    # One SRD header a packet, so no packet declares a continuation and the
    # General Packing Mode the offer claims is the one being built.
    block = frame.stamp(0)
    assert ((block[:, 18] & 0x80) != 0).sum() == 0
    assert PACKET_HEADER_SIZE + payload_size + 8 <= video.max_udp
