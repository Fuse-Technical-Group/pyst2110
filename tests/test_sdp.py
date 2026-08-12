"""Reading and writing a flow and a video format as an SDP (SPEC §spec:sdp).

Vectors are written by hand from RFC 4566 and ST 2110-20, not round-tripped
through a writer of ours. The round trip is checked too, as the additional
property it is: it proves the pair agree with each other, and the fixed
vectors are what tie either of them to the documents (§spec:testing).
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import pytest

from pyst2110.sdp import (
    SdpFlow,
    SdpVideo,
    format_sdp,
    parse_sdp,
    parse_video_format,
)

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


def test_colorimetry_is_carried_through_as_the_token_the_sdp_said():
    text = _ST2110_20.rstrip() + "; colorimetry=BT2020\n"
    assert parse_video_format(text).colorimetry == "BT2020"


def test_an_absent_colorimetry_reads_as_unspecified():
    """ST 2110-20 section 7.5 has a value for 'not specified', so an SDP that
    omits a required parameter is recorded rather than invented over."""
    assert parse_video_format(_ST2110_20).colorimetry == "UNSPECIFIED"


def test_maxudp_is_the_senders_declared_limit_and_defaults_to_the_standard():
    """ST 2110-20 section 7.3: absent MAXUDP means the Standard UDP Size Limit."""
    assert parse_video_format(_ST2110_20).max_udp == 1460
    text = _ST2110_20.rstrip() + "; MAXUDP=8960\n"
    assert parse_video_format(text).max_udp == 8960


# --- Emitting -------------------------------------------------------------
#
# The vectors below are written from RFC 4566 section 5 (which lines, in which
# order) and SMPTE ST 2110-20 section 7 (which media type parameters, spelled
# how). They are not this library's parse run backwards.

_FLOW = SdpFlow("239.100.0.1", 20000, "192.168.100.2")


def video(**overrides: Any) -> SdpVideo:
    """The reference format, with whatever a case wants to differ."""
    fields: dict[str, Any] = {
        "width": 1920,
        "height": 1080,
        "frame_rate": Fraction(60000, 1001),
        "depth": 10,
        "sampling": "YCbCr-4:2:2",
        "colorimetry": "BT709",
    }
    return SdpVideo(**(fields | overrides))


_VIDEO = video()

# RFC 4566 section 5: v=, o=, s= are REQUIRED and "MUST appear in exactly the
# order given here", one time description follows, then the media section.
# ST 2110-20 section 7.1: media name "video", subtype "raw", and the rtpmap
# clause "shall indicate the 90 kHz RTP Clock rate".
_OFFER_LINES = [
    "v=0",
    "o=- 0 0 IN IP4 192.168.100.2",
    "s= ",  # RFC 4566 section 5.3: not empty; a single space where unnamed.
    "t=0 0",
    "m=video 20000 RTP/AVP 96",
    "c=IN IP4 239.100.0.1/64",
    "a=source-filter: incl IN IP4 239.100.0.1 192.168.100.2",
    "a=rtpmap:96 raw/90000",
    # ST 2110-20 section 7.2, in the order the standard lists them. Entries are
    # "separated by the semicolon character followed by whitespace" with "no
    # semicolon character after the last item" (section 7.1).
    "a=fmtp:96 sampling=YCbCr-4:2:2; width=1920; height=1080; "
    "exactframerate=60000/1001; depth=10; colorimetry=BT709; PM=2110GPM; "
    "SSN=ST2110-20:2017",
]


def test_an_emitted_offer_is_the_document_rfc_4566_and_st_2110_20_require():
    assert format_sdp(_FLOW, _VIDEO) == "".join(f"{line}\r\n" for line in _OFFER_LINES)


def test_records_end_with_crlf():
    """RFC 4566: "The sequence CRLF (0x0d0a) is used to end a record"."""
    text = format_sdp(_FLOW, _VIDEO)
    assert text.endswith("\r\n")
    assert text.count("\r\n") == len(_OFFER_LINES)
    assert "\n" not in text.replace("\r\n", "")


def test_every_required_media_type_parameter_is_present():
    """ST 2110-20 section 7.2 names eight, and a sender "shall include" them."""
    fmtp = _fmtp(format_sdp(_FLOW, _VIDEO))
    assert set(fmtp) == {
        "sampling",
        "width",
        "height",
        "exactframerate",
        "depth",
        "colorimetry",
        "PM",
        "SSN",
    }


def test_an_integer_frame_rate_is_a_single_decimal_number():
    """ST 2110-20 section 7.2 spells 25 as "25", not as "25/1"."""
    fmtp = _fmtp(format_sdp(_FLOW, video(frame_rate=Fraction(25))))
    assert fmtp["exactframerate"] == "25"


def test_a_non_integer_frame_rate_is_a_ratio_with_the_smallest_numerator():
    """Section 7.2 again: "utilizing the numerically smallest numerator"."""
    fmtp = _fmtp(format_sdp(_FLOW, video(frame_rate=Fraction(120000, 2002))))
    assert fmtp["exactframerate"] == "60000/1001"


def test_interlace_is_a_bare_flag_present_only_when_interlaced():
    """ST 2110-20 section 7.3: absent means progressive, so it is not written
    with a value and not written at all for progressive video."""
    assert "interlace" not in _fmtp(format_sdp(_FLOW, _VIDEO))
    assert _fmtp(format_sdp(_FLOW, video(interlaced=True)))["interlace"] == ""


def test_maxudp_is_written_only_when_it_is_not_the_standard_limit():
    """Section 7.3: "If absent, it indicates that the Standard UDP Size Limit
    is in use", so writing the default would say something it does not mean."""
    assert "MAXUDP" not in _fmtp(format_sdp(_FLOW, _VIDEO))
    assert _fmtp(format_sdp(_FLOW, video(max_udp=8960)))["MAXUDP"] == "8960"


def test_the_standard_number_follows_the_colorimetry():
    """Section 7.2: ST2110-20:2017 "unless the colorimetry value ALPHA or the
    TCS value ST2115LOGS3 are used", which need the 2022 revision."""
    assert _fmtp(format_sdp(_FLOW, _VIDEO))["SSN"] == "ST2110-20:2017"
    assert (
        _fmtp(format_sdp(_FLOW, video(colorimetry="ALPHA")))["SSN"] == "ST2110-20:2022"
    )


def test_a_unicast_destination_carries_no_ttl():
    """RFC 4566 section 5.7 requires a TTL for IPv4 *multicast* addresses."""
    flow = SdpFlow("192.0.2.10", 20000)
    assert "c=IN IP4 192.0.2.10\r\n" in format_sdp(flow, _VIDEO)


def test_an_ipv6_multicast_destination_carries_no_ttl():
    """RFC 4566 section 5.7: "the TTL value MUST NOT be present for IPv6"."""
    flow = SdpFlow("ff3e::8000:1", 20000)
    assert "c=IN IP6 ff3e::8000:1\r\n" in format_sdp(flow, _VIDEO)


def test_no_source_filter_means_the_line_is_absent_rather_than_empty():
    text = format_sdp(SdpFlow("239.100.0.1", 20000), _VIDEO)
    assert "source-filter" not in text
    # With no sender to name, the origin falls back to the unspecified address.
    assert "o=- 0 0 IN IP4 0.0.0.0\r\n" in text


def test_a_session_id_makes_the_origin_unique_when_a_caller_needs_it():
    text = format_sdp(_FLOW, _VIDEO, session_id=1443716955)
    assert "o=- 1443716955 1443716955 IN IP4 192.168.100.2\r\n" in text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("destination_ip", "239.100.0.1; DROP"),
        ("source_ip", "not-an-address"),
    ],
)
def test_an_address_that_is_not_one_is_refused(field: str, value: str):
    flow = SdpFlow(
        **(
            {"destination_ip": "239.100.0.1", "destination_port": 20000}
            | {field: value}
        )
    )
    with pytest.raises(ValueError, match="IP address"):
        format_sdp(flow, _VIDEO)


# Every character ``str.splitlines()`` starts a new line at. RFC 4566 ends a
# record with CRLF, but a reader that splits the document into lines — as
# parse_sdp does — begins a record at any of these.
_LINE_TERMINATORS = [
    "\n",
    "\r",
    "\v",
    "\f",
    "\x1c",
    "\x1d",
    "\x1e",
    "\x85",
    "\u2028",
    "\u2029",
]


def test_the_terminators_tested_are_the_ones_a_line_split_breaks_on():
    """The list is the property rather than a transcription: a character a
    split starts a line at is one a caller's string can forge a record with."""
    breaks = [one for one in _LINE_TERMINATORS if len(f"a{one}b".splitlines()) > 1]
    assert breaks == _LINE_TERMINATORS
    assert len(_LINE_TERMINATORS) == 10


@pytest.mark.parametrize("terminator", _LINE_TERMINATORS)
def test_no_line_terminator_in_a_session_name_can_forge_a_record(terminator: str):
    """A caller's string is data, not more SDP. Refusing CR and LF alone
    leaves eight characters that redirect the whole flow: a name carrying one
    emits an s= line that reads back as another connection and media section.
    """
    hostile = f"x{terminator}c=IN IP4 6.6.6.6{terminator}m=video 5004 RTP/AVP 96"
    with pytest.raises(ValueError, match="session name"):
        format_sdp(_FLOW, _VIDEO, session_name=hostile)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("destination_ip", "ff3e::8000:1%\r\nc=IN IP4 6.6.6.6\r\na=x"),
        # No '/' — an address parse refuses that one character and nothing
        # else of a scope, and a media line needs none of it to be read.
        ("source_ip", "fe80::1%\r\nc=IN IP4 6.6.6.6\r\nm=video 5004 x 96"),
    ],
)
def test_an_address_carrying_a_scope_identifier_is_refused(field: str, value: str):
    """CPython accepts any scope identifier free of '%', carriage returns and
    newlines included, so writing the caller's own string forges records with
    genuine CRLFs — past a strict RFC 4566 reader, not merely past a line
    split. A scope names a local interface and has no meaning in an offer.
    """
    flow = SdpFlow(
        **(
            {"destination_ip": "239.100.0.1", "destination_port": 20000}
            | {field: value}
        )
    )
    with pytest.raises(ValueError, match="scope"):
        format_sdp(flow, _VIDEO)


def test_the_address_written_is_the_one_that_was_parsed():
    """Validating one string and writing another is the gap an injection goes
    through, so what reaches the document is what the address parse made."""
    flow = SdpFlow("FF3E:0000:0000:0000:0000:0000:8000:0001", 20000, "2001:0DB8::0001")
    text = format_sdp(flow, _VIDEO)
    assert "c=IN IP6 ff3e::8000:1\r\n" in text
    assert "o=- 0 0 IN IP6 2001:db8::1\r\n" in text
    assert "a=source-filter: incl IN IP6 ff3e::8000:1 2001:db8::1\r\n" in text


def test_a_source_filter_over_two_families_names_neither():
    """RFC 4570 section 3 allows the wildcard address type, which a v4
    destination filtering a v6 source needs: naming either family describes
    the other address wrongly."""
    flow = SdpFlow("239.100.0.1", 20000, "2001:db8::1")
    assert "a=source-filter: incl IN * 239.100.0.1 2001:db8::1\r\n" in format_sdp(
        flow, _VIDEO
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        # A depth that forges fmtp parameters: reparsed, MAXUDP reads 9000 and
        # the packing mode reads Block, changing the sizing a peer computes.
        ("depth", "10; MAXUDP=9000; PM=2110BPM", "depth"),
        ("depth", "10\r\nc=IN IP4 6.6.6.6", "depth"),
        # ST 2110-20 section 7.4.2 lists the depths a sender may declare.
        ("depth", 9, "depth"),
        # The fmtp line is the document's last, so a c= forged here wins under
        # the last-wins reading a session-level connection line gets.
        ("max_udp", "9000\r\nc=IN IP4 6.6.6.6", "MAXUDP"),
        ("max_udp", 0, "MAXUDP"),
        ("width", "1920\r\nc=IN IP4 6.6.6.6", "width"),
        ("height", "1080\r\nc=IN IP4 6.6.6.6", "height"),
    ],
)
def test_a_format_value_that_is_not_the_integer_it_claims_is_refused(
    field: str, value: object, message: str
):
    """``SdpVideo`` is a plain dataclass, so an annotation of ``int`` is a
    promise and not a check: a string reaches the fmtp line verbatim."""
    with pytest.raises(ValueError, match=message):
        format_sdp(_FLOW, video(**{field: value}))


@pytest.mark.parametrize("session_id", ["1 1 IN IP4 6.6.6.6\r\nc=IN IP4 6.6.6.6", -1])
def test_a_session_id_that_is_not_a_whole_number_is_refused(session_id: object):
    """It fills the o= line's identifier and its version. A string forges an
    o= and a c= near the top of the document; a negative writes a malformed
    origin with no error at all."""
    with pytest.raises(ValueError, match="session id"):
        format_sdp(_FLOW, _VIDEO, session_id=session_id)


@pytest.mark.parametrize("sampling", ["", "YCbCr 4:2:2"])
def test_a_sampling_that_is_not_one_token_is_refused(sampling: str):
    """ST 2110-20 section 7.1: "no whitespace within the name or value"."""
    with pytest.raises(ValueError, match="sampling"):
        format_sdp(_FLOW, video(sampling=sampling))


@pytest.mark.parametrize(("field", "value"), [("width", 0), ("height", 32768)])
def test_a_raster_outside_what_st_2110_permits_is_refused(field: str, value: int):
    """Section 7.2: width and height are "integers between 1 and 32767"."""
    with pytest.raises(ValueError, match=field):
        format_sdp(_FLOW, video(**{field: value}))


def test_a_port_outside_the_udp_range_is_refused():
    with pytest.raises(ValueError, match="port"):
        format_sdp(SdpFlow("239.100.0.1", 70000), _VIDEO)


def test_a_payload_type_outside_the_seven_bit_field_is_refused():
    """RFC 3550 section 5.1 gives the payload type seven bits."""
    with pytest.raises(ValueError, match="payload type"):
        format_sdp(_FLOW, _VIDEO, payload_type=200)


@pytest.mark.parametrize(
    "flow",
    [
        SdpFlow("239.100.0.1", 20000, "192.168.100.2"),
        SdpFlow("239.100.0.1", 20000),
        SdpFlow("192.0.2.10", 5004, "192.0.2.1"),
    ],
)
def test_an_emitted_offer_parses_back_to_the_flow_it_described(flow: SdpFlow):
    """The additional property, not the evidence: reader and writer agree."""
    assert parse_sdp(format_sdp(flow, _VIDEO)) == flow


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"frame_rate": Fraction(24)},
        {"interlaced": True},
        {"max_udp": 8960},
        {"depth": 12, "sampling": "RGB", "colorimetry": "BT2020"},
        {"width": 3840, "height": 2160, "frame_rate": Fraction(24)},
    ],
)
def test_an_emitted_offer_parses_back_to_the_format_it_described(overrides: dict):
    described = video(**overrides)
    assert parse_video_format(format_sdp(_FLOW, described)) == described


def _fmtp(text: str) -> dict[str, str]:
    """The fmtp line's parameters, split here rather than by the parse."""
    line = next(one for one in text.split("\r\n") if one.startswith("a=fmtp:"))
    _, _, rest = line.partition(" ")
    fields = (field.strip().partition("=") for field in rest.split(";"))
    return {name: value for name, _, value in fields}
