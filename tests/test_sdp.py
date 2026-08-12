"""Reading a flow and a video format out of an SDP (SPEC §spec:sdp).

Vectors are written by hand from RFC 4566 and ST 2110-20, not round-tripped
through a writer of ours (§spec:testing).
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from pyst2110.sdp import parse_sdp, parse_video_format

_ST2110_20 = """\
v=0
o=- 1443716955 1443716955 IN IP4 192.168.100.2
s=SMPTE ST2110-20 narrow
t=0 0
m=video 20000 RTP/AVP 96
c=IN IP4 239.100.0.1/64
a=source-filter: incl IN IP4 239.100.0.1 192.168.100.2
a=rtpmap:96 raw/90000
a=fmtp:96 sampling=YCbCr-4:2:2; width=1920; height=1080; exactframerate=60000/1001
"""


def test_a_full_offer_yields_destination_and_source():
    flow = parse_sdp(_ST2110_20)
    assert flow.destination_ip == "239.100.0.1"
    assert flow.destination_port == 20000
    assert flow.source_ip == "192.168.100.2"


def test_the_connection_ttl_is_not_part_of_the_address():
    flow = parse_sdp("m=video 5004 RTP/AVP 96\nc=IN IP4 239.0.0.1/32\n")
    assert flow.destination_ip == "239.0.0.1"


def test_a_source_filter_without_a_space_after_the_colon_still_parses():
    text = (
        "m=video 5004 RTP/AVP 96\n"
        "c=IN IP4 239.0.0.1\n"
        "a=source-filter:incl IN IP4 239.0.0.1 10.0.0.9\n"
    )
    assert parse_sdp(text).source_ip == "10.0.0.9"


def test_an_excluding_source_filter_is_not_a_source():
    text = (
        "m=video 5004 RTP/AVP 96\n"
        "c=IN IP4 239.0.0.1\n"
        "a=source-filter: excl IN IP4 239.0.0.1 10.0.0.9\n"
    )
    assert parse_sdp(text).source_ip == ""


def test_no_source_filter_means_any_sender():
    flow = parse_sdp("m=video 5004 RTP/AVP 96\nc=IN IP4 239.0.0.1\n")
    assert flow.source_ip == ""


def test_a_missing_connection_line_is_named():
    with pytest.raises(ValueError, match="connection line"):
        parse_sdp("m=video 5004 RTP/AVP 96\n")


def test_a_missing_media_line_is_named():
    with pytest.raises(ValueError, match="m=video"):
        parse_sdp("c=IN IP4 239.0.0.1\n")


def test_a_non_numeric_port_is_no_port():
    with pytest.raises(ValueError, match="port 0"):
        parse_sdp("m=video none RTP/AVP 96\nc=IN IP4 239.0.0.1\n")


def test_a_port_with_a_stream_count_parses():
    """RFC 4566's port field is ``port ["/" integer]``; 2022-7 offers use it."""
    flow = parse_sdp("c=IN IP4 239.0.0.1\nm=video 20000/2 RTP/AVP 96\n")
    assert flow.destination_port == 20000


def test_the_video_section_wins_over_a_later_audio_section():
    """A 2110 SDP carries several media; a flow on the audio port sees nothing."""
    flow = parse_sdp(
        "c=IN IP4 239.0.0.1\n"
        "m=video 20000 RTP/AVP 96\n"
        "a=source-filter: incl IN IP4 239.0.0.1 192.0.2.5\n"
        "m=audio 20010 RTP/AVP 97\n"
        "c=IN IP4 239.0.0.2\n"
    )
    assert flow.destination_port == 20000
    assert flow.destination_ip == "239.0.0.1"
    assert flow.source_ip == "192.0.2.5"


def test_the_video_format_comes_off_the_fmtp_line():
    video = parse_video_format(_ST2110_20)
    assert video.width == 1920
    assert video.height == 1080
    assert video.frame_rate == Fraction(60000, 1001)
    assert video.sampling == "YCbCr-4:2:2"
    assert video.depth == 10
    assert video.interlaced is False


def test_a_bare_interlace_flag_is_read_by_its_presence():
    text = _ST2110_20.replace("exactframerate=60000/1001", "exactframerate=25; depth=8")
    assert parse_video_format(text).interlaced is False
    assert parse_video_format(text + "").depth == 8
    assert parse_video_format(text.rstrip() + "; interlace\n").interlaced is True


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("width=1920; ", "width"),
        ("height=1080; ", "height"),
        ("exactframerate=60000/1001", "exactframerate"),
    ],
)
def test_a_format_the_transmitter_needs_is_named_when_absent(
    missing: str, message: str
):
    with pytest.raises(ValueError, match=message):
        parse_video_format(_ST2110_20.replace(missing, ""))


def test_an_sdp_without_a_format_line_is_named():
    with pytest.raises(ValueError, match="a=fmtp"):
        parse_video_format("m=video 5004 RTP/AVP 96\nc=IN IP4 239.0.0.1\n")


def test_a_malformed_frame_rate_is_named():
    text = _ST2110_20.replace("exactframerate=60000/1001", "exactframerate=fast")
    with pytest.raises(ValueError, match="not an exactframerate"):
        parse_video_format(text)


def test_the_video_format_ignores_another_essence_fmtp():
    """An audio section before the video one must not describe the geometry."""
    text = (
        "c=IN IP4 239.0.0.1\n"
        "m=audio 20010 RTP/AVP 97\n"
        "a=fmtp:97 channel-order=SMPTE2110.(ST)\n"
        "m=video 20000 RTP/AVP 96\n"
        "a=fmtp:96 sampling=YCbCr-4:2:2; width=1920; height=1080; "
        "exactframerate=25; depth=10\n"
    )
    video = parse_video_format(text)
    assert video.width == 1920
    assert video.height == 1080
