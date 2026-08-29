"""Reading and writing a flow and a video format as an SDP (SPEC §spec:sdp).

Vectors are written by hand from RFC 4566, RFC 4570, ST 2110-10, ST 2110-20
and ST 2110-21, not round-tripped through a writer of ours. The round trip is
checked too, as the additional property it is: it proves the pair agree with
each other, and the fixed vectors are what tie either of them to the
documents (§spec:testing).
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import pytest

from pyst2110.sdp import (
    SdpFlow,
    SdpVideo,
    format_dup_sdp,
    format_sdp,
    parse_dup_sdp,
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


def test_an_earlier_audio_section_does_not_lend_the_video_its_connection():
    """The mirror of the case above, with the audio section first. Only what
    precedes the first ``m=`` is session level (RFC 4566 section 5), so an
    audio block's own ``c=`` is that block's and never the video's — a flow
    built from it would carry the audio group and receive nothing."""
    flow = parse_sdp(
        "c=IN IP4 239.0.0.1\n"
        "m=audio 20010 RTP/AVP 97\n"
        "c=IN IP4 239.0.0.2\n"
        "m=video 20000 RTP/AVP 96\n"
    )
    assert flow.destination_ip == "239.0.0.1"
    assert flow.destination_port == 20000


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


@pytest.mark.parametrize(
    ("declared", "forged", "message"),
    [
        ("width=1920", "width=40000", "width"),
        ("width=1920", "width=0", "width"),
        ("height=1080", "height=40000", "height"),
        ("height=1080", "height=0", "height"),
    ],
)
def test_a_raster_the_standard_does_not_permit_is_refused_on_the_way_in(
    declared: str, forged: str, message: str
):
    """Section 7.2 bounds both at 1-32767, and the emitter refuses them there.
    A parse that let one through would hand the transmit path a width whose
    sample offsets run past the SRD Offset field into the C bit."""
    with pytest.raises(ValueError, match=message):
        parse_video_format(_ST2110_20.replace(declared, forged))


def test_a_declared_udp_size_no_datagram_could_have_is_refused():
    """A UDP length field is sixteen bits, so a MAXUDP above 65535 describes
    a datagram that cannot exist."""
    with pytest.raises(ValueError, match="MAXUDP"):
        parse_video_format(_ST2110_20.rstrip() + "; MAXUDP=200000\n")


def test_an_sdp_without_a_format_line_is_named():
    with pytest.raises(ValueError, match="a=fmtp"):
        parse_video_format("m=video 5004 RTP/AVP 96\nc=IN IP4 239.0.0.1\n")


def test_a_malformed_frame_rate_is_named():
    text = _ST2110_20.replace("exactframerate=60000/1001", "exactframerate=fast")
    with pytest.raises(ValueError, match="not an exactframerate"):
        parse_video_format(text)


@pytest.mark.parametrize("rate", ["0", "0/1", "-50", "-30000/1001", "90001"])
def test_a_frame_rate_no_sender_could_have_is_refused_on_the_way_in(rate: str):
    """A rate at or below zero is not a rate, and one above the 90 kHz RTP
    Clock of ST 2110-20 section 6.1.3 gives successive frames the same media
    timestamp. Unbounded, a zero divided the parse's own TROFF check and every
    consumer of ``frame_rate`` after it, and a negative one reached
    :func:`pyst2110.timing.sender_limits` as a negative T_DRAIN."""
    text = _ST2110_20.replace("exactframerate=60000/1001", f"exactframerate={rate}")
    with pytest.raises(ValueError, match="exactframerate"):
        parse_video_format(text)


def test_a_zero_frame_rate_is_refused_before_troff_divides_by_it():
    """The TROFF bound is a frame period, so the rate is checked first."""
    text = _ST2110_20.replace("exactframerate=60000/1001", "exactframerate=0; TROFF=10")
    with pytest.raises(ValueError, match="exactframerate"):
        parse_video_format(text)


def test_the_rtp_clock_rate_itself_is_a_frame_rate_the_parse_accepts():
    text = _ST2110_20.replace("exactframerate=60000/1001", "exactframerate=90000")
    assert parse_video_format(text).frame_rate == Fraction(90_000)


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
    # ST 2110-20 section 7.2, in the order the standard lists them, then the
    # one ST 2110-21 section 8.1 adds. Entries are "separated by the semicolon
    # character followed by whitespace" with "no semicolon character after the
    # last item" (2110-20 section 7.1).
    "a=fmtp:96 sampling=YCbCr-4:2:2; width=1920; height=1080; "
    "exactframerate=60000/1001; depth=10; colorimetry=BT709; PM=2110GPM; "
    "SSN=ST2110-20:2017; TP=2110TPN",
]

# SMPTE ST 2110-20 section 7.2, "Required Media Type Parameters": the eight a
# sender "shall include ... in the a=fmtp clause of the SDP for all streams
# conforming to this standard".
_ST2110_20_REQUIRED = frozenset(
    {
        "sampling",
        "depth",
        "width",
        "height",
        "exactframerate",
        "colorimetry",
        "PM",
        "SSN",
    }
)

# SMPTE ST 2110-21 section 8.1, "Required Parameters": the *additional*
# parameter a sender "shall include in the a=fmtp clause of the SDP for all
# video RTP streams conforming to this standard". It lives in -21 rather than
# -20, so a conformance review of -20 alone does not see it.
_ST2110_21_REQUIRED = frozenset({"TP"})


def test_an_emitted_offer_is_the_document_rfc_4566_and_st_2110_20_require():
    assert format_sdp(_FLOW, _VIDEO) == "".join(f"{line}\r\n" for line in _OFFER_LINES)


def test_records_end_with_crlf():
    """RFC 4566: "The sequence CRLF (0x0d0a) is used to end a record"."""
    text = format_sdp(_FLOW, _VIDEO)
    assert text.endswith("\r\n")
    assert text.count("\r\n") == len(_OFFER_LINES)
    assert "\n" not in text.replace("\r\n", "")


def test_every_required_media_type_parameter_is_present():
    """A sender's required set is not one document's. ST 2110-20 section 7.2
    names eight and ST 2110-21 section 8.1 adds TP, so an offer carrying only
    the first eight is a stream a conformant receiver has grounds to refuse.
    """
    fmtp = _fmtp(format_sdp(_FLOW, _VIDEO))
    assert set(fmtp) == _ST2110_20_REQUIRED | _ST2110_21_REQUIRED


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"interlaced": True},
        {"max_udp": 8960},
        {"colorimetry": "ALPHA"},
        {"interlaced": True, "max_udp": 8960, "depth": 12, "sampling": "RGB"},
    ],
)
def test_no_optional_parameter_displaces_a_required_one(overrides: dict):
    """The set above is what an offer must carry whatever else it carries. The
    section 7.3 parameters are written by their difference from a default, and
    a required one must not be able to go missing behind that."""
    fmtp = _fmtp(format_sdp(_FLOW, video(**overrides)))
    assert set(fmtp) >= _ST2110_20_REQUIRED | _ST2110_21_REQUIRED


def test_the_sender_signals_its_traffic_shape_because_st_2110_21_requires_it():
    """Section 8.1: TP "signals the type of the sender as defined in section
    7.1", and section 7.1.2 fixes Type N's token as 2110TPN."""
    assert _fmtp(format_sdp(_FLOW, _VIDEO))["TP"] == "2110TPN"


@pytest.mark.parametrize(
    "sender_type",
    # ST 2110-21 section 7.1.1: a sender "shall conform to one or more of the
    # types defined in clauses 7.1.2, 7.1.3, or 7.1.4". Each of those clauses
    # ends "shall signal compliance with a Media Type Parameter TP of value
    # <token>" — Narrow, Narrow Linear and Wide, spelled as the standard
    # prints them.
    ["2110TPN", "2110TPNL", "2110TPW"],
)
def test_every_sender_type_the_standard_defines_can_be_signalled(sender_type: str):
    fmtp = _fmtp(format_sdp(_FLOW, _VIDEO, sender_type=sender_type))
    assert fmtp["TP"] == sender_type


@pytest.mark.parametrize(
    "sender_type",
    [
        "2110TPX",  # not a type
        "",
        "narrow",  # the prose name, not the token
        "2110TPn",  # the tokens are printed uppercase
        # A forged parameter: reparsed, the packing mode reads Block and the
        # sizing a peer computes changes.
        "2110TPN; PM=2110BPM",
        "2110TPN\r\nc=IN IP4 6.6.6.6",
    ],
)
def test_a_sender_type_st_2110_21_does_not_define_is_refused(sender_type: str):
    """A TP is a claim about pacing a receiver provisions its buffer from, so
    an unrecognised one is refused rather than passed through."""
    with pytest.raises(ValueError, match="TP"):
        format_sdp(_FLOW, _VIDEO, sender_type=sender_type)


def test_the_sender_type_follows_the_parameters_of_the_document_that_owns_it():
    """ST 2110-21 section 8.1 calls TP "additional" to 2110-20's set, so it is
    written after them — and before the section 7.3 parameters, which are
    written only by their difference from a default."""
    fmtp = list(_fmtp(format_sdp(_FLOW, video(interlaced=True, max_udp=8960))))
    assert fmtp.index("SSN") < fmtp.index("TP") < fmtp.index("interlace")
    assert fmtp.index("TP") < fmtp.index("MAXUDP")


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
    text = format_sdp(flow, _VIDEO, any_source=True)
    assert "c=IN IP6 ff3e::8000:1\r\n" in text


def test_a_multicast_offer_that_names_no_sender_is_refused():
    """ST 2110-10 section 8.4: a sender "should indicate the source address
    information for streams within the SDP in order to support source-specific
    multicast sessions by use of an inclusive source filter line". A transmit
    offer has a sender and knows its address, so leaving the line out is
    almost always a caller that forgot rather than one that meant it — and a
    transmit SDK refuses the offer without it."""
    with pytest.raises(ValueError, match="any_source"):
        format_sdp(SdpFlow("239.100.0.1", 20000), _VIDEO)


def test_an_any_source_multicast_offer_may_still_omit_the_filter():
    """RFC 4570 section 3.1: "the default behavior when a source-filter
    attribute is not provided in a session description is that all traffic
    sent to the specified <connection-address> value should be accepted (i.e.,
    from any source address)". That is a real session, so it stays reachable
    — as a statement rather than as the default."""
    text = format_sdp(SdpFlow("239.100.0.1", 20000), _VIDEO, any_source=True)
    assert "source-filter" not in text
    # With no sender to name, the origin falls back to the unspecified address.
    assert "o=- 0 0 IN IP4 0.0.0.0\r\n" in text


def test_naming_a_sender_and_offering_the_group_to_any_source_contradict():
    """The two say opposite things about who may send, so an offer that says
    both says nothing this library will guess between."""
    with pytest.raises(ValueError, match="any_source"):
        format_sdp(_FLOW, _VIDEO, any_source=True)


def test_a_unicast_offer_needs_no_source_filter():
    """RFC 4570's filter selects among the senders to a group, and ST 2110-10
    section 8.4 asks for the line "in order to support source-specific
    multicast sessions". A unicast flow has no group and no such choice."""
    assert "source-filter" not in format_sdp(SdpFlow("192.0.2.10", 20000), _VIDEO)


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


@pytest.mark.parametrize(
    "flow",
    [
        SdpFlow("239.100.0.1", 20000, "2001:db8::1"),
        SdpFlow("ff3e::8000:1", 20000, "192.168.100.2"),
    ],
)
def test_a_source_filter_spanning_two_address_families_is_refused(flow: SdpFlow):
    """RFC 4570 section 3 gives the filter one <address-types> for both
    addresses, and section 3.1 bounds the wildcard that would cover two:
    "When the <addrtype> value is the '*' wildcard, the <dest-address> MUST be
    either an FQDN or '*' (i.e., it MUST NOT be an IPv4 or IPv6 address)".
    This library writes IP literals, so the mixed pair has no legal spelling —
    and no packet either, a v6 sender reaching no v4 group."""
    with pytest.raises(ValueError, match="address famil"):
        format_sdp(flow, _VIDEO)


def test_a_source_filter_names_the_one_family_both_addresses_share():
    """RFC 4570 section 3: <address-types> "identifies the address family, and
    for the purpose of this document may be either <addrtype> value 'IP4' or
    'IP6'". ST 2110-10 section 8.4's example is the IP4 form."""
    assert "a=source-filter: incl IN IP4 239.100.0.1 192.168.100.2\r\n" in format_sdp(
        _FLOW, _VIDEO
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
    ("flow", "any_source"),
    [
        (SdpFlow("239.100.0.1", 20000, "192.168.100.2"), False),
        (SdpFlow("239.100.0.1", 20000), True),
        (SdpFlow("192.0.2.10", 5004, "192.0.2.1"), False),
        (SdpFlow("192.0.2.10", 5004), False),
    ],
)
def test_an_emitted_offer_parses_back_to_the_flow_it_described(
    flow: SdpFlow, any_source: bool
):
    """The additional property, not the evidence: reader and writer agree."""
    assert parse_sdp(format_sdp(flow, _VIDEO, any_source=any_source)) == flow


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


# --- ST 2110-21 section 8 parameters ---------------------------------------
#
# Section 8.1 requires TP; section 8.2 permits TROFF, "a positive integer
# number of microseconds", and CMAX, "an integer number". Both optional
# parameters carry meaning by their absence — the type's default — so both
# are read as absent and written only when set (§spec:sdp, §spec:timing).


def test_the_sender_type_is_read_off_tp():
    text = _ST2110_20.rstrip() + "; TP=2110TPW\n"
    assert parse_video_format(text).sender_type == "2110TPW"


def test_an_offer_without_tp_reads_as_narrow():
    """Section 8.1 makes TP mandatory, so an offer without it is one 2110-21
    does not describe. Narrow is the strictest type and the emit default, so
    an undeclared sender is measured against the demanding schedule rather
    than passed through with no type at all."""
    assert parse_video_format(_ST2110_20).sender_type == "2110TPN"


def test_a_sender_type_the_standard_does_not_define_is_refused_on_the_way_in():
    with pytest.raises(ValueError, match="TP"):
        parse_video_format(_ST2110_20.rstrip() + "; TP=2110TPX\n")


def test_troff_is_read_in_microseconds_and_absent_means_the_default():
    assert parse_video_format(_ST2110_20).tr_offset_us is None
    text = _ST2110_20.rstrip() + "; TROFF=640\n"
    assert parse_video_format(text).tr_offset_us == 640


@pytest.mark.parametrize("value", ["²", "٣", "①"])
def test_a_troff_of_digits_int_would_not_read_the_same_way_is_refused(value: str):
    """``str.isdigit()`` is true of superscripts, of every script's decimal
    digits and of circled numbers. The superscript then dies inside ``int()``,
    naming the text rather than the parameter; the Arabic-Indic digit is worse
    — ``int()`` reads it, so TROFF=٣ was silently accepted as 3."""
    with pytest.raises(ValueError, match="TROFF"):
        parse_video_format(_ST2110_20.rstrip() + f"; TROFF={value}\n")


def test_a_media_port_of_non_ascii_digits_is_not_a_port():
    """The same idiom in the ``m=`` line: RFC 4566's port is a decimal
    integer, and ٥٠٠٤ is not the port 5004."""
    text = "m=video ٥٠٠٤ RTP/AVP 96\nc=IN IP4 239.0.0.1\n"
    with pytest.raises(ValueError, match="port"):
        parse_sdp(text)


@pytest.mark.parametrize("value", ["-1", "1.5", "640us", "16684"])
def test_a_troff_outside_a_frame_or_not_a_whole_number_is_refused(value: str):
    """Section 6.2: TR_OFFSET is the difference between the most recent
    integer multiple of T_FRAME and T_VD, non-negative — so it lies inside one
    frame period, which at 60000/1001 is 16683 microseconds and change."""
    with pytest.raises(ValueError, match="TROFF"):
        parse_video_format(_ST2110_20.rstrip() + f"; TROFF={value}\n")


def test_cmax_is_read_as_the_senders_claim_and_absent_means_the_types():
    assert parse_video_format(_ST2110_20).cmax is None
    assert parse_video_format(_ST2110_20.rstrip() + "; CMAX=5\n").cmax == 5


def test_a_cmax_that_is_not_a_positive_whole_number_is_refused():
    with pytest.raises(ValueError, match="CMAX"):
        parse_video_format(_ST2110_20.rstrip() + "; CMAX=0\n")


def test_the_emitted_sender_type_is_the_formats_unless_the_caller_overrides():
    assert _fmtp(format_sdp(_FLOW, video(sender_type="2110TPW")))["TP"] == "2110TPW"
    text = format_sdp(_FLOW, video(sender_type="2110TPW"), sender_type="2110TPNL")
    assert _fmtp(text)["TP"] == "2110TPNL"


def test_troff_and_cmax_are_written_only_when_set():
    assert "TROFF" not in _fmtp(format_sdp(_FLOW, _VIDEO))
    assert "CMAX" not in _fmtp(format_sdp(_FLOW, _VIDEO))
    fmtp = _fmtp(format_sdp(_FLOW, video(tr_offset_us=640, cmax=5)))
    assert fmtp["TROFF"] == "640"
    assert fmtp["CMAX"] == "5"
    assert list(fmtp).index("TP") < list(fmtp).index("TROFF") < list(fmtp).index("CMAX")


@pytest.mark.parametrize(
    "overrides",
    [
        {"tr_offset_us": "640; PM=2110BPM"},
        {"tr_offset_us": -1},
        {"tr_offset_us": 16684},
        {"cmax": 0},
        {"cmax": True},
    ],
)
def test_an_optional_2110_21_parameter_is_validated_before_it_is_written(
    overrides: dict,
):
    with pytest.raises(ValueError, match=r"TROFF|CMAX"):
        format_sdp(_FLOW, video(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"sender_type": "2110TPNL"},
        {"sender_type": "2110TPW", "tr_offset_us": 640, "cmax": 12},
    ],
)
def test_the_2110_21_parameters_parse_back_to_the_format_that_wrote_them(
    overrides: dict,
):
    described = video(**overrides)
    assert parse_video_format(format_sdp(_FLOW, described)) == described


# --- ST 2022-7 duplication --------------------------------------------------
#
# The vector below is RFC 7104 section 8.2's own example, "Separate
# Destination Addresses", transcribed line for line: a session-level
# a=group:DUP naming the a=mid: tags of two m=video blocks, each block
# carrying its own c= and a=source-filter. Its payload is MP2T rather than raw
# video, and the RFC's own spellings are kept — no space after the
# source-filter colon, a /127 scope — because a vector edited towards what
# this library writes is the writer again (§spec:testing).

_RFC_7104_DUP = """\
v=0
o=ali 1122334455 1122334466 IN IP4 dup.example.com
s=DUP Grouping Semantics
t=0 0
a=group:DUP S1a S1b
m=video 30000 RTP/AVP 100
c=IN IP4 233.252.0.1/127
a=source-filter:incl IN IP4 233.252.0.1 198.51.100.1
a=rtpmap:100 MP2T/90000
a=mid:S1a
m=video 30000 RTP/AVP 101
c=IN IP4 233.252.0.2/127
a=source-filter:incl IN IP4 233.252.0.2 198.51.100.1
a=rtpmap:101 MP2T/90000
a=mid:S1b
"""


def test_both_legs_of_a_duplicated_offer_are_read():
    """Each leg is its own block's connection and source filter."""
    assert parse_dup_sdp(_RFC_7104_DUP) == (
        SdpFlow("233.252.0.1", 30000, "198.51.100.1"),
        SdpFlow("233.252.0.2", 30000, "198.51.100.1"),
    )


def test_the_legs_come_back_in_the_order_the_group_names_them():
    """That order decides which leg is which everywhere downstream, so it is
    the group and not the document order that is read (§spec:redundancy)."""
    text = _RFC_7104_DUP.replace("a=group:DUP S1a S1b", "a=group:DUP S1b S1a")
    first, second = parse_dup_sdp(text)
    assert first.destination_ip == "233.252.0.2"
    assert second.destination_ip == "233.252.0.1"


def test_two_tags_over_one_media_block_are_refused():
    """The failure the refusal exists for: a sender that emitted one leg where
    two were meant sends unprotected essence and reports success."""
    text = _RFC_7104_DUP.partition("m=video 30000 RTP/AVP 101")[0]
    with pytest.raises(ValueError, match="DUP"):
        parse_dup_sdp(text)


def test_one_tag_over_two_media_blocks_is_refused():
    """Rivermax's rmx_output_media_set_sdp states the rule from the other
    side: the number of identification tags "has to correspond to the number
    of m=video blocks"."""
    text = _RFC_7104_DUP.replace("a=group:DUP S1a S1b", "a=group:DUP S1a")
    with pytest.raises(ValueError, match="DUP"):
        parse_dup_sdp(text)


def test_a_tag_naming_no_media_block_is_refused():
    """The counts agree here and the tags still do not name the blocks."""
    text = _RFC_7104_DUP.replace("a=mid:S1b", "a=mid:S2b")
    with pytest.raises(ValueError, match="S1b"):
        parse_dup_sdp(text)


def test_a_group_naming_one_block_twice_is_refused():
    """Two tags and two blocks, and one leg described twice — one path, not
    two, and the receiver joins the same socket from both."""
    text = _RFC_7104_DUP.replace("a=group:DUP S1a S1b", "a=group:DUP S1a S1a")
    with pytest.raises(ValueError, match="S1a"):
        parse_dup_sdp(text)


def test_two_media_blocks_carrying_one_tag_are_refused():
    text = _RFC_7104_DUP.replace("a=mid:S1b", "a=mid:S1a")
    with pytest.raises(ValueError, match="S1a"):
        parse_dup_sdp(text)


def test_two_distinct_blocks_naming_one_socket_are_refused():
    """The two cases above catch a document that names one block twice; this
    is the one where both blocks are real and describe the same socket. The
    writer already refuses it, and the reader is the side facing a document
    it did not write: a receiver provisioned from such a pair joins one group
    twice while its operator is told the essence is protected."""
    text = _RFC_7104_DUP.replace(
        "c=IN IP4 233.252.0.2/127", "c=IN IP4 233.252.0.1/127"
    ).replace(
        "a=source-filter:incl IN IP4 233.252.0.2 198.51.100.1",
        "a=source-filter:incl IN IP4 233.252.0.1 198.51.100.1",
    )
    with pytest.raises(ValueError, match="one path described twice"):
        parse_dup_sdp(text)


def test_two_legs_onto_one_group_from_different_senders_are_a_pair():
    """Whole-flow equality is the test, not the address: source-specific
    multicast puts two paths onto one group, and that is a real pair."""
    text = _RFC_7104_DUP.replace(
        "c=IN IP4 233.252.0.2/127", "c=IN IP4 233.252.0.1/127"
    ).replace(
        "a=source-filter:incl IN IP4 233.252.0.2 198.51.100.1",
        "a=source-filter:incl IN IP4 233.252.0.1 198.51.100.2",
    )
    first, second = parse_dup_sdp(text)
    assert first.destination_ip == second.destination_ip
    assert (first.source_ip, second.source_ip) == ("198.51.100.1", "198.51.100.2")


def test_more_legs_than_a_pair_are_refused():
    """A pair is what this library models — pyst2110.redundancy reconstructs
    two legs — and what Rivermax carries, RMX_MAX_DUP_STREAMS being two. A
    third leg is one nothing downstream of the parse can take."""
    text = _RFC_7104_DUP.replace("a=group:DUP S1a S1b", "a=group:DUP S1a S1b S1c")
    third = (
        "m=video 30000 RTP/AVP 102\n"
        "c=IN IP4 233.252.0.3/127\n"
        "a=source-filter:incl IN IP4 233.252.0.3 198.51.100.1\n"
        "a=mid:S1c\n"
    )
    with pytest.raises(ValueError, match="two"):
        parse_dup_sdp(text + third)


@pytest.mark.parametrize(
    ("declared", "written", "message"),
    [
        ("c=IN IP4 233.252.0.2/127\n", "", "connection"),
        ("m=video 30000 RTP/AVP 101", "m=video 0 RTP/AVP 101", "port 0"),
    ],
)
def test_a_leg_that_describes_no_flow_is_named_by_its_tag(
    declared: str, written: str, message: str
):
    """Each leg is held to what a single-leg offer is held to, and the tag is
    what says which of the two the caller has to fix."""
    with pytest.raises(ValueError, match=f"S1b.*{message}"):
        parse_dup_sdp(_RFC_7104_DUP.replace(declared, written))


def test_a_leg_without_its_own_connection_takes_the_sessions():
    """RFC 4566 section 5.7 scopes a session-level c= to every media section
    supplying none, which is how RFC 7104 section 8.1's one group address
    reaches both legs."""
    text = _RFC_7104_DUP.replace("c=IN IP4 233.252.0.2/127\n", "").replace(
        "t=0 0", "c=IN IP4 233.252.0.9\nt=0 0"
    )
    assert parse_dup_sdp(text)[1].destination_ip == "233.252.0.9"


def test_a_single_leg_offer_is_not_a_redundant_pair():
    with pytest.raises(ValueError, match="a=group:DUP"):
        parse_dup_sdp(_ST2110_20)


def test_a_redundant_offer_is_not_read_as_one_leg():
    """parse_sdp returns one flow, so handed a pair it would return whichever
    block came first — and the caller would join, or send, one leg of two."""
    with pytest.raises(ValueError, match="parse_dup_sdp"):
        parse_sdp(_RFC_7104_DUP)


def test_a_port_count_does_not_make_an_offer_redundant():
    """RFC 4566's port field is ``port ["/" integer]``, which an ST 2022-7
    offer may carry. a=group:DUP is what makes a pair, and this document is
    one leg however its port is spelled."""
    text = "c=IN IP4 239.0.0.1\nm=video 20000/2 RTP/AVP 96\n"
    assert parse_sdp(text) == SdpFlow("239.0.0.1", 20000)
    with pytest.raises(ValueError, match="a=group:DUP"):
        parse_dup_sdp(text)


def test_a_grouping_that_is_not_duplication_leaves_a_single_leg_offer_alone():
    """RFC 5888 carries other semantics on a=group:, and only DUP means a
    redundant pair (RFC 7104 section 5)."""
    text = "a=group:LS 1 2\nc=IN IP4 239.0.0.1\nm=video 20000 RTP/AVP 96\n"
    assert parse_sdp(text).destination_port == 20000


# The same document in the direction this library writes. The group line is a
# session-level attribute, so RFC 4566 section 5 puts it after t= and before
# the first media section; each block carries its own connection, source
# filter and identification tag, and the two blocks describe one essence, so
# the format line is the same line twice.
_SECOND_LEG = SdpFlow("239.100.0.2", 20000, "192.168.101.2")

_DUP_OFFER_LINES = [
    "v=0",
    "o=- 0 0 IN IP4 192.168.100.2",
    "s= ",
    "t=0 0",
    "a=group:DUP 1 2",
    "m=video 20000 RTP/AVP 96",
    "c=IN IP4 239.100.0.1/64",
    "a=source-filter: incl IN IP4 239.100.0.1 192.168.100.2",
    "a=mid:1",
    "a=rtpmap:96 raw/90000",
    _OFFER_LINES[-1],
    "m=video 20000 RTP/AVP 96",
    "c=IN IP4 239.100.0.2/64",
    "a=source-filter: incl IN IP4 239.100.0.2 192.168.101.2",
    "a=mid:2",
    "a=rtpmap:96 raw/90000",
    _OFFER_LINES[-1],
]


def test_an_emitted_dup_offer_is_the_document_rfc_7104_requires():
    assert format_dup_sdp(_FLOW, _SECOND_LEG, _VIDEO) == "".join(
        f"{line}\r\n" for line in _DUP_OFFER_LINES
    )


def test_the_dup_tags_are_as_many_as_the_video_blocks_and_name_them():
    """The rule RFC 7104 and Rivermax both state, asserted over the document
    this library writes rather than over the one it reads."""
    lines = format_dup_sdp(_FLOW, _SECOND_LEG, _VIDEO).split("\r\n")
    tags = next(one for one in lines if one.startswith("a=group:DUP")).split()[1:]
    assert len(tags) == len([one for one in lines if one.startswith("m=video ")]) == 2
    assert [f"a=mid:{tag}" for tag in tags] == [
        one for one in lines if one.startswith("a=mid:")
    ]


def test_one_flow_written_twice_is_not_a_redundant_pair():
    """Two blocks naming one socket describe one path, and an offer that says
    otherwise claims a protection the sender has not got."""
    with pytest.raises(ValueError, match="one leg"):
        format_dup_sdp(_FLOW, _FLOW, _VIDEO)


def test_two_legs_may_share_a_group_where_their_senders_differ():
    """Source-specific multicast: one group, two sources, two paths — RFC
    7104 section 8.1's case written as section 8.2's two blocks."""
    second = SdpFlow("239.100.0.1", 20000, "192.168.101.2")
    assert parse_dup_sdp(format_dup_sdp(_FLOW, second, _VIDEO)) == (_FLOW, second)


def test_every_leg_is_validated_as_a_single_leg_offer_is():
    """One set of rules for a flow: a multicast leg naming no sender is
    refused wherever it appears."""
    with pytest.raises(ValueError, match="any_source"):
        format_dup_sdp(_FLOW, SdpFlow("239.100.0.2", 20000), _VIDEO)


def test_the_video_format_of_a_duplicated_offer_is_the_one_both_legs_carry():
    """Two paths, one essence, so the format parse is the one a caller
    already had and reads the first block's fmtp line."""
    assert parse_video_format(format_dup_sdp(_FLOW, _SECOND_LEG, _VIDEO)) == _VIDEO


@pytest.mark.parametrize(
    ("first", "second", "any_source"),
    [
        (_FLOW, _SECOND_LEG, False),
        (SdpFlow("239.100.0.1", 20000), SdpFlow("239.100.0.2", 20000), True),
        (SdpFlow("192.0.2.10", 5004), SdpFlow("192.0.2.11", 5004), False),
        (SdpFlow("ff3e::8000:1", 20000), SdpFlow("ff3e::8000:2", 20000), True),
    ],
)
def test_an_emitted_dup_offer_parses_back_to_the_two_legs_it_described(
    first: SdpFlow, second: SdpFlow, any_source: bool
):
    """The additional property, not the evidence: reader and writer agree."""
    text = format_dup_sdp(first, second, _VIDEO, any_source=any_source)
    assert parse_dup_sdp(text) == (first, second)
