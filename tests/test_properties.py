"""Properties over generated inputs, where the example tests pin chosen ones.

Hypothesis draws the inputs and, on a failure, shrinks them to the smallest
that still fails, so a report names a minimal packet or format. The
hand-written vectors in ``test_rtp.py``, ``test_payload_header.py`` and
``test_transmit.py`` tie the library to the documents (§spec:testing); these
check that the same rules hold across the input space.

The RTP parse is checked against a byte-by-byte reading of RFC 3550 written
here, on arbitrary bytes and sizes, where ``test_path_equality.py`` checks the
parse's two paths against each other. The frame and offer round trips are the
additional property of §spec:testing, as in ``test_rasters.py``, which walks a
fixed grid; these draw the raster, payload limit, frame index and optional
offer parameters that grid holds constant.
"""

from __future__ import annotations

import struct
from fractions import Fraction

import numpy as np
from hypothesis import given, reject
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pyst2110.division import (
    CONSTITUENT_COUNTS,
    DIVISIONS,
    reassemble,
    split,
    sub_image_size,
    to_constituent,
    to_essence,
)
from pyst2110.geometry import choose_payload_size, pgroup, pgroup_rows
from pyst2110.payload import parse_payload_headers
from pyst2110.rtp import FIXED_HEADER_SIZE, parse_rtp
from pyst2110.sdp import (
    RANGES,
    SENDER_TYPES,
    SdpFlow,
    SdpVideo,
    format_sdp,
    parse_sdp,
    parse_video_format,
)
from pyst2110.transmit import FrameHeaders

# RFC 3550 §5.1 and §5.3.1 sizes, restated so the reference parse below reads
# the wire from its own constants and not from the parser it checks.
_CSRC_OCTETS = 4
_EXTENSION_HEADER_OCTETS = 4
_EXTENSION_WORD_OCTETS = 4
_MAX_STRIDE = 96
_MAX_PACKETS = 24
# A conforming sender's first octet: version two, no padding, extension or
# CSRC list. Drawing it often steers chunks onto the parser's fast path.
_CONFORMING_FIRST_OCTET = 0x80


def _reference_rtp(packet: bytes, size: int) -> dict[str, int]:
    """One packet's fixed header, read byte by byte from RFC 3550 §5.1."""
    if size < FIXED_HEADER_SIZE:
        return dict.fromkeys(
            (
                "version",
                "padding",
                "extension",
                "csrc_count",
                "marker",
                "payload_type",
                "sequence",
                "timestamp",
                "ssrc",
            ),
            0,
        ) | {"payload_offset": size}
    first, second, sequence, timestamp, ssrc = struct.unpack_from(">BBHII", packet)
    csrc_count = first & 0x0F
    extension = bool(first & 0x10)
    offset = FIXED_HEADER_SIZE + _CSRC_OCTETS * csrc_count
    if extension and offset + _EXTENSION_HEADER_OCTETS <= size:
        (words,) = struct.unpack_from(">H", packet, offset + 2)
        offset += _EXTENSION_HEADER_OCTETS + _EXTENSION_WORD_OCTETS * words
    elif extension:
        offset += _EXTENSION_HEADER_OCTETS
    return {
        "version": first >> 6,
        "padding": int(bool(first & 0x20)),
        "extension": int(extension),
        "csrc_count": csrc_count,
        "marker": int(bool(second & 0x80)),
        "payload_type": second & 0x7F,
        "sequence": sequence,
        "timestamp": timestamp,
        "ssrc": ssrc,
        "payload_offset": min(offset, size),
    }


@st.composite
def _chunks(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray | None]:
    """A chunk of packets, half of them conforming, with or without sizes."""
    count = draw(st.integers(0, _MAX_PACKETS))
    stride = draw(st.integers(FIXED_HEADER_SIZE, _MAX_STRIDE))
    packets = draw(arrays(np.uint8, (count, stride)))
    if count and draw(st.booleans()):
        packets[:, 0] = _CONFORMING_FIRST_OCTET
    sizes = None
    if draw(st.booleans()):
        sizes = np.array(
            draw(st.lists(st.integers(0, stride), min_size=count, max_size=count)),
            dtype=np.int64,
        )
    return packets, sizes


@given(_chunks())
def test_parse_rtp_agrees_with_a_byte_by_byte_reading(
    chunk: tuple[np.ndarray, np.ndarray | None],
) -> None:
    packets, sizes = chunk
    parsed = parse_rtp(packets, sizes)
    stride = packets.shape[1]
    for row in range(packets.shape[0]):
        size = stride if sizes is None else int(sizes[row])
        expected = _reference_rtp(packets[row].tobytes(), size)
        got = {name: int(getattr(parsed, name)[row]) for name in expected}
        assert got == expected, f"row {row} of a {packets.shape} chunk"


@given(_chunks(), st.integers(1, 4))
def test_payload_headers_stay_inside_the_chunk_on_any_bytes(
    chunk: tuple[np.ndarray, np.ndarray | None], max_segments: int
) -> None:
    packets, sizes = chunk
    rtp = parse_rtp(packets, sizes)
    headers = parse_payload_headers(
        packets, rtp.payload_offset, sizes, max_segments=max_segments
    )
    count = packets.shape[0]
    assert (headers.segments >= 0).all()
    assert (headers.segments <= max_segments).all()
    assert int(headers.segments.sum()) == headers.packet.size
    if headers.packet.size:
        assert (np.diff(headers.packet) >= 0).all()
        assert headers.packet.min() >= 0
        assert headers.packet.max() < count
    # A withheld packet contributes no descriptors.
    withheld = headers.overflowed | headers.unreadable
    assert (headers.segments[withheld] == 0).all()


# ST 2110-20 formats small enough to build a frame per example: every
# sampling and depth with a pgroup, a width of whole pgroups, and a payload
# limit from one pgroup up to a few lines' worth.
_FORMATS = st.sampled_from(
    [
        ("YCbCr-4:2:2", 8),
        ("YCbCr-4:2:2", 10),
        ("YCbCr-4:2:2", 12),
        ("YCbCr-4:2:2", 16),
        ("YCbCr-4:4:4", 8),
        ("YCbCr-4:4:4", 10),
        ("RGB", 12),
        ("RGB", 16),
        ("YCbCr-4:2:0", 8),
        ("YCbCr-4:2:0", 10),
    ]
)


@st.composite
def _frames(draw: st.DrawFn) -> tuple[SdpVideo, int]:
    sampling, depth = draw(_FORMATS)
    probe = SdpVideo(
        width=1, height=2, frame_rate=Fraction(50), depth=depth, sampling=sampling
    )
    group_bytes, group_pixels = pgroup(probe)
    rows = pgroup_rows(probe)
    width = group_pixels * draw(st.integers(1, 12))
    height = rows * draw(st.integers(1, 8))
    interlaced = rows == 1 and draw(st.booleans())
    if interlaced:
        height = max(height, 2)
    video = SdpVideo(
        width=width,
        height=height,
        frame_rate=Fraction(draw(st.sampled_from([25, 30, 50, 60]))),
        depth=depth,
        sampling=sampling,
        interlaced=interlaced,
    )
    limit = draw(st.integers(group_bytes, group_bytes * 64))
    return video, choose_payload_size(video, limit)


@given(_frames(), st.integers(0, 3))
def test_a_built_frame_parses_back_to_the_raster_it_covers(
    frame: tuple[SdpVideo, int], frame_index: int
) -> None:
    video, payload_size = frame
    headers = FrameHeaders(video, payload_size)
    block = headers.stamp(frame_index).copy()
    rtp = parse_rtp(block)
    payload = parse_payload_headers(block, rtp.payload_offset)

    count = block.shape[0]
    assert (rtp.version == 2).all()
    # The marker ends a frame, or each field of an interlaced one (ST 2110-20
    # section 6.1.2), so it falls where the field changes and on the last.
    fields = payload.field.tolist()
    ends = [fields[i] != fields[i + 1] for i in range(count - 1)] + [True]
    assert rtp.marker.tolist() == ends
    # One SRD a packet, each the payload size, as General Packing Mode lays out.
    assert (payload.segments == 1).all()
    assert (payload.length == payload_size).all()
    assert not payload.overflowed.any()
    assert not payload.unreadable.any()
    # The thirty-two-bit counter, split across the two headers, runs on
    # without a gap from this frame's first packet.
    counter = (payload.extended_sequence << 16) | rtp.sequence.astype(np.int64)
    first = frame_index * count
    assert counter.tolist() == list(range(first, first + count))

    # Every row of every field is covered once, left to right, end to end.
    group_bytes, group_pixels = pgroup(video)
    step = payload_size // group_bytes * group_pixels
    rows_seen: dict[tuple[bool, int], list[int]] = {}
    for field, line, offset in zip(
        payload.field.tolist(),
        payload.line.tolist(),
        payload.offset_samples.tolist(),
        strict=True,
    ):
        rows_seen.setdefault((field, line), []).append(offset)
    for offsets in rows_seen.values():
        assert offsets == list(range(0, video.width, step))
    assert len(rows_seen) * len(range(0, video.width, step)) == count


_DIVISION_RASTERS = st.sampled_from([(16, 8), (32, 16), (64, 32), (128, 64)])


@given(
    st.sampled_from(DIVISIONS),
    st.sampled_from(CONSTITUENT_COUNTS),
    _DIVISION_RASTERS,
    st.integers(1, 3),
)
def test_reassembling_a_split_essence_returns_it(
    division: str, count: int, raster: tuple[int, int], channels: int
) -> None:
    width, height = raster
    essence = np.arange(width * height * channels, dtype=np.int64).reshape(
        height, width, channels
    )
    constituents = split(essence, division=division, count=count)
    assert len(constituents) == count
    sub_width, sub_height = sub_image_size(
        width, height, division=division, count=count
    )
    for constituent in constituents:
        assert constituent.shape == (sub_height, sub_width, channels)
    assert np.array_equal(reassemble(constituents, division=division), essence)


@given(
    st.sampled_from(DIVISIONS),
    st.sampled_from(CONSTITUENT_COUNTS),
    _DIVISION_RASTERS,
    st.data(),
)
def test_point_mapping_to_a_constituent_and_back_is_the_identity(
    division: str, count: int, raster: tuple[int, int], data: st.DataObject
) -> None:
    width, height = raster
    xs = np.array(data.draw(st.lists(st.integers(0, width - 1), min_size=1)))
    ys = np.array(
        data.draw(
            st.lists(st.integers(0, height - 1), min_size=xs.size, max_size=xs.size)
        )
    )
    raster_args = {"width": width, "height": height, "division": division}
    constituent, cx, cy = to_constituent(xs, ys, count=count, **raster_args)
    back_x, back_y = to_essence(constituent, cx, cy, count=count, **raster_args)
    assert np.array_equal(back_x, xs)
    assert np.array_equal(back_y, ys)


_SAMPLINGS = (
    "YCbCr-4:2:2",
    "YCbCr-4:4:4",
    "YCbCr-4:2:0",
    "RGB",
    "CLYCbCr-4:2:2",
    "ICtCp-4:2:2",
)
_IPV4 = st.tuples(*[st.integers(1, 254)] * 4).map(
    lambda octets: ".".join(map(str, octets))
)
_MULTICAST = st.tuples(
    st.integers(224, 239), *[st.integers(0, 255)] * 2, st.integers(1, 254)
).map(lambda octets: ".".join(map(str, octets)))


@st.composite
def _offers(draw: st.DrawFn) -> tuple[SdpFlow, SdpVideo]:
    flow = SdpFlow(
        destination_ip=draw(st.one_of(_MULTICAST, _IPV4)),
        destination_port=draw(st.integers(1, 65535)),
        source_ip=draw(st.one_of(st.just(""), _IPV4)),
    )
    video = SdpVideo(
        width=draw(st.integers(1, 7680)),
        height=draw(st.integers(1, 4320)),
        frame_rate=draw(
            st.sampled_from(
                [
                    Fraction(n, d)
                    for n, d in (
                        (24000, 1001),
                        (25, 1),
                        (30000, 1001),
                        (50, 1),
                        (60000, 1001),
                        (60, 1),
                        (120, 1),
                    )
                ]
            )
        ),
        depth=draw(st.sampled_from([8, 10, 12, 16])),
        sampling=draw(st.sampled_from(_SAMPLINGS)),
        colorimetry=draw(st.sampled_from(["BT709", "BT2020", "BT2100", "UNSPECIFIED"])),
        interlaced=draw(st.booleans()),
        max_udp=draw(st.sampled_from([1460, 8960])),
        sender_type=draw(st.sampled_from(SENDER_TYPES)),
        tr_offset_us=draw(st.one_of(st.none(), st.integers(0, 10_000))),
        cmax=draw(st.one_of(st.none(), st.integers(1, 64))),
        range=draw(st.sampled_from(RANGES)),
    )
    return flow, video


@given(_offers())
def test_an_offer_parses_back_to_the_flow_and_format_it_describes(
    offer: tuple[SdpFlow, SdpVideo],
) -> None:
    flow, video = offer
    try:
        text = format_sdp(flow, video)
    except ValueError:
        # A combination the emit refuses is not an offer; the example tests
        # pin which ones those are.
        reject()
    assert parse_sdp(text) == flow
    assert parse_video_format(text) == video
