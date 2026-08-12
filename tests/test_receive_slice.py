"""The whole receive path over one frame (§spec:rtp, §spec:payload-header).

An SDP offer in, and out the far end: how many packets the frame should be,
which frame each packet belongs to, whether any were lost, and where every
payload lands in the raster (§spec:geometry). The roadmap's verify criterion
for the receive slice, run against a frame built here rather than captured
off a commercial transmitter — a capture is the stronger evidence and needs
hardware this suite does not have (§spec:testing).

The packets are written field by field from the diagrams in RFC 3550
section 5.1 and SMPTE ST 2110-20 section 6.1.4 by the builder below. It is not
a writer of ours under test: it places each field with its own big-endian
conversion at its own octet, which is what makes it independent evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyst2110 import (
    FrameTracker,
    SdpVideo,
    SequenceTracker,
    byte_offset,
    choose_payload_size,
    fits_raster,
    line_bytes,
    packets_per_frame,
    packets_per_line,
    parse_payload_headers,
    parse_rtp,
    parse_video_format,
    pgroup,
)

# A 1080p59.94 YCbCr-4:2:2 10-bit offer, the format the roadmap names.
_OFFER = """\
v=0
o=- 1443716955 1443716955 IN IP4 192.168.100.2
s=SMPTE ST2110-20 narrow
t=0 0
m=video 20000 RTP/AVP 96
c=IN IP4 239.100.0.1/64
a=source-filter: incl IN IP4 239.100.0.1 192.168.100.2
a=rtpmap:96 raw/90000
a=fmtp:96 sampling=YCbCr-4:2:2; width=1920; height=1080; exactframerate=60000/1001; \
depth=10; TP=2110TPN
"""

_HEADER_SIZE = 20  # 12 of RTP, 2 of extended sequence, 6 of one SRD
_SSRC = 0x1234ABCD
_PAYLOAD_TYPE = 96
_MARKER = 0x80


def build_packet(
    sequence: int,
    timestamp: int,
    line: int,
    offset: int,
    length: int,
    marker: bool,
) -> bytearray:
    """One packet's headers, each field placed at its own octet."""
    header = bytearray(_HEADER_SIZE)
    header[0] = 0x80  # V=2, P=0, X=0, CC=0
    header[1] = (_MARKER if marker else 0x00) | _PAYLOAD_TYPE
    header[2:4] = (sequence % (1 << 16)).to_bytes(2, "big")  # RTP sequence
    header[4:8] = timestamp.to_bytes(4, "big")  # RTP timestamp
    header[8:12] = _SSRC.to_bytes(4, "big")  # SSRC
    header[12:14] = (sequence >> 16).to_bytes(2, "big")  # extended sequence
    header[14:16] = length.to_bytes(2, "big")  # SRD Length, octets
    header[16:18] = line.to_bytes(2, "big")  # F=0 in the top bit, then the row
    header[18:20] = offset.to_bytes(2, "big")  # C=0 in the top bit, then offset
    return header


def build_frame(
    payload_size: int, per_line: int, height: int, group: tuple[int, int], first: int
) -> np.ndarray:
    """A whole frame of packets, raster order, marker on the last."""
    group_bytes, group_pixels = group
    count = per_line * height
    rows = np.zeros((count, _HEADER_SIZE), dtype=np.uint8)
    for index in range(count):
        rows[index] = build_packet(
            sequence=first + index,
            timestamp=0x0AAAAAAA,
            line=index // per_line,
            offset=index % per_line * payload_size // group_bytes * group_pixels,
            length=payload_size,
            marker=index == count - 1,
        )
    return rows


@pytest.fixture(scope="module")
def frame() -> tuple[SdpVideo, int, np.ndarray, np.ndarray]:
    """The offer's frame, built once: format, payload size, packets, sizes.

    Numbered from 65,000, so the RTP field wraps inside the frame. Nothing
    below reads the packets other than through a parse, so one build serves
    every case.
    """
    video = parse_video_format(_OFFER)
    payload_size = choose_payload_size(video, video.max_udp)
    rows = build_frame(
        payload_size,
        packets_per_line(video, payload_size),
        video.height,
        pgroup(video),
        first=65_000,
    )
    return video, payload_size, rows, np.full(rows.shape[0], _HEADER_SIZE, np.int64)


def test_the_offer_predicts_the_frames_packet_count():
    video = parse_video_format(_OFFER)
    assert (video.width, video.height, video.depth) == (1920, 1080, 10)
    assert line_bytes(video) == 4800
    assert choose_payload_size(video, video.max_udp) == 1200
    assert packets_per_line(video, 1200) == 4
    assert packets_per_frame(video, 1200) == 4320


def test_a_frame_parses_into_descriptors_that_tile_the_raster_exactly_once(frame):
    """No overlap and no hole, which is the roadmap's verify criterion."""
    video, payload_size, rows, sizes = frame
    stride = line_bytes(video)
    assert rows.shape[0] == packets_per_frame(video, payload_size)

    rtp = parse_rtp(rows, sizes=sizes)
    payload = parse_payload_headers(rows, rtp.payload_offset, sizes=sizes)

    assert not payload.overflowed.any()
    assert payload.segments.tolist() == [1] * rows.shape[0]
    assert payload.field.tolist() == [False] * rows.shape[0]

    # Where every payload lands: the descriptors that name a place in the
    # raster, their row, then their offset turned into octets by the pgroup.
    fits = fits_raster(video, payload.line, payload.offset)
    assert fits.all(), "a conformant frame places every descriptor"
    starts = payload.line[fits] * stride + byte_offset(video, payload.offset[fits])
    ends = starts + payload.length[fits]

    coverage = np.zeros(video.height * stride + 1, dtype=np.int64)
    np.add.at(coverage, starts, 1)
    np.add.at(coverage, ends, -1)
    covered = np.cumsum(coverage)[:-1]
    assert covered.min() == 1, "a hole in the raster"
    assert covered.max() == 1, "an overlap in the raster"


def test_a_crafted_descriptor_is_flagged_before_it_places_anything():
    """One 20-octet datagram naming row 32767 at sample 32767 with a length of
    65535: every field is legal on the wire and every one of them is outside
    this flow. The gather it feeds is a device-side kernel, so the descriptor
    has to be refused before the arithmetic, not after (§spec:scope-boundary).
    """
    video = parse_video_format(_OFFER)
    stride = line_bytes(video)
    hostile = np.frombuffer(
        build_packet(
            sequence=0,
            timestamp=0,
            line=0x7FFF,
            offset=0x7FFF,
            length=0xFFFF,
            marker=False,
        ),
        dtype=np.uint8,
    ).reshape(1, _HEADER_SIZE)
    sizes = np.array([_HEADER_SIZE], dtype=np.int64)

    rtp = parse_rtp(hostile, sizes=sizes)
    payload = parse_payload_headers(hostile, rtp.payload_offset, sizes=sizes)
    assert payload.line.tolist() == [32767]
    assert payload.offset.tolist() == [32767]
    assert payload.length.tolist() == [65535]

    # Unmasked, the descriptor scales to 157 MB into a 5 MB frame.
    frame_bytes = video.height * stride
    unmasked = payload.line * stride + byte_offset(video, payload.offset)
    assert unmasked.tolist() == [157_363_515]
    assert unmasked[0] > frame_bytes

    # Masked, it places nothing at all.
    fits = fits_raster(video, payload.line, payload.offset)
    assert fits.tolist() == [False]
    starts = payload.line[fits] * stride + byte_offset(video, payload.offset[fits])
    assert starts.size == 0


def test_the_frame_and_sequence_trackers_agree_across_a_split_chunk(frame):
    """A frame arriving in two acquisitions is one frame and no loss."""
    _, _, rows, sizes = frame

    sequences = SequenceTracker()
    frames = FrameTracker()
    indices = []
    for chunk in (rows[:2000], rows[2000:]):
        headers = parse_rtp(chunk, sizes=sizes[: chunk.shape[0]])
        sequences.observe(headers.sequence)
        indices.append(frames.observe(headers.marker))

    assert sequences.received == 4320
    assert sequences.lost == 0
    assert sequences.discontinuities == 0, "the 16-bit wrap read as a gap"
    assert sequences.resyncs == 0
    # One frame, ended by the marker on its last packet.
    assert frames.frames == 1
    assert np.concatenate(indices).tolist() == [0] * 4320


def test_a_dropped_packet_shows_up_as_loss_and_a_hole(frame):
    """The two halves of the verify criterion move together: a missing packet
    is both a sequence gap and a gap in the raster."""
    video, payload_size, rows, sizes = frame
    lossy = np.delete(rows, [7, 8, 9], axis=0)
    sizes = sizes[: lossy.shape[0]]

    rtp = parse_rtp(lossy, sizes=sizes)
    sequences = SequenceTracker()
    sequences.observe(rtp.sequence)
    assert sequences.lost == 3
    assert sequences.discontinuities == 1

    payload = parse_payload_headers(lossy, rtp.payload_offset, sizes=sizes)
    stride = line_bytes(video)
    fits = fits_raster(video, payload.line, payload.offset)
    starts = payload.line[fits] * stride + byte_offset(video, payload.offset[fits])
    covered = np.zeros(video.height * stride, dtype=np.int64)
    for start, length in zip(
        starts.tolist(), payload.length[fits].tolist(), strict=True
    ):
        covered[start : start + length] += 1
    assert int((covered == 0).sum()) == 3 * payload_size
