"""Reading an RP 2110-23 group as one offer (SPEC §spec:multi-stream).

The reference documents are SMPTE RP 2110-23:2019's own Annex B and Annex E
examples, transcribed by hand. The PDF wraps each fmtp line across three and
spaces every record with a blank line, and prints the SSN value between
typographic quotes; the transcription joins each fmtp line, drops the blank
lines and the quotes, and changes nothing else — the annexes' missing ``t=``
line and ``TP`` parameter included.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from pyst2110.division import MULTI_2SI, MULTI_SD
from pyst2110.sdp import (
    SdpConformanceWarning,
    SdpFlow,
    parse_dup_sdp,
    parse_group_sdp,
    parse_sdp,
    parse_video_format,
)

_ANNEX_FMTP = (
    "a=fmtp:112 sampling=YCbCr-4:2:2; width=1920; height=1080; "
    "exactframerate=60000/1001; depth=10; TCS=SDR; colorimetry=BT709; "
    "PM=2110GPM; SSN=ST2110-20:2017"
)


def _annex_block(tag: str, address: str) -> str:
    return (
        "m=video 30000 RTP/AVP 112\n"
        f"c=IN IP4 {address}\n"
        "a=rtpmap:112 raw/90000\n"
        f"{_ANNEX_FMTP}\n"
        f"a=mid:{tag}\n"
    )


_ANNEX_HEADER = "v=0\no=- 0 0 IN IP4 10.200.8.2\ns=Video 1\n"


def _annex(semantics: str, count: int) -> str:
    tags = [str(number) for number in range(1, count + 1)]
    return (
        _ANNEX_HEADER
        + f"a=group:{semantics} {' '.join(tags)}\n"
        + "".join(
            _annex_block(tag, f"239.252.0.{index}") for index, tag in enumerate(tags)
        )
    )


# RP 2110-23 Annex B: a 2160p59.94 UHD signal sent as four 2SI streams.
_ANNEX_B = _annex("MULTI-2SI", 4)
# Annex E: a 4320p59.94 signal sent as sixteen 2SI streams.
_ANNEX_E = _annex("MULTI-2SI", 16)


def test_annex_b_reads_as_four_2si_constituents_of_a_2160_line_essence():
    group = parse_group_sdp(_ANNEX_B)
    assert group.division == MULTI_2SI
    assert group.flows == tuple(
        SdpFlow(f"239.252.0.{index}", 30000) for index in range(4)
    )
    assert (group.video.width, group.video.height) == (1920, 1080)
    assert group.video.frame_rate == Fraction(60000, 1001)
    assert group.video.sampling == "YCbCr-4:2:2"
    assert group.video.colorimetry == "BT709"
    assert (group.essence_width, group.essence_height) == (3840, 2160)


def test_annex_e_reads_as_sixteen_constituents_of_a_4320_line_essence():
    group = parse_group_sdp(_ANNEX_E)
    assert len(group.flows) == 16
    assert group.flows[15] == SdpFlow("239.252.0.15", 30000)
    assert (group.essence_width, group.essence_height) == (7680, 4320)


def test_square_division_reads_as_its_own_division():
    assert parse_group_sdp(_annex("MULTI-SD", 4)).division == MULTI_SD


def test_constituents_come_back_in_group_order_not_document_order():
    blocks = [_annex_block(str(n), f"239.252.0.{n}") for n in (1, 2, 3, 4)]
    text = _ANNEX_HEADER + "a=group:MULTI-2SI 1 2 3 4\n" + "".join(blocks[::-1])
    assert [flow.destination_ip for flow in parse_group_sdp(text).flows] == [
        "239.252.0.1",
        "239.252.0.2",
        "239.252.0.3",
        "239.252.0.4",
    ]


def test_tags_that_are_not_numbers_take_their_number_from_their_position():
    tags = ["Q1", "Q2", "Q3", "Q4"]
    text = (
        _ANNEX_HEADER
        + f"a=group:MULTI-2SI {' '.join(tags)}\n"
        + "".join(_annex_block(tag, f"239.252.0.{n}") for n, tag in enumerate(tags))
    )
    assert parse_group_sdp(text).flows[3].destination_ip == "239.252.0.3"


def test_a_numbered_tag_out_of_its_position_is_refused():
    """The RP writes constituent n's tag as n; a document numbering a block
    one thing and placing it another says two things about which sub-image
    the flow carries."""
    text = _ANNEX_B.replace("MULTI-2SI 1 2 3 4", "MULTI-2SI 2 1 3 4")
    with pytest.raises(ValueError, match="position"):
        parse_group_sdp(text)


@pytest.mark.parametrize("count", [2, 3, 8])
def test_a_group_of_a_count_the_rp_does_not_define_is_refused(count: int):
    with pytest.raises(ValueError, match="4 or 16"):
        parse_group_sdp(_annex("MULTI-2SI", count))


@pytest.mark.parametrize(
    ("old", "new", "named"),
    [
        ("width=1920", "width=1280", "width"),
        ("height=1080", "height=720", "height"),
        ("depth=10", "depth=12", "depth"),
        ("sampling=YCbCr-4:2:2", "sampling=RGB", "sampling"),
        ("exactframerate=60000/1001", "exactframerate=50", "frame rate"),
        ("colorimetry=BT709", "colorimetry=BT2020", "colorimetry"),
        ("TCS=SDR", "TCS=PQ", "TCS"),
    ],
)
def test_constituents_of_unequal_format_are_refused(old: str, new: str, named: str):
    """The division assumes equal sub-images; an essence assembled from
    unequal ones has no defined raster."""
    blocks = _ANNEX_B.split("m=video")
    blocks[3] = blocks[3].replace(old, new)
    with pytest.raises(ValueError, match=named):
        parse_group_sdp("m=video".join(blocks))


def test_interlaced_constituents_are_refused():
    text = _ANNEX_B.replace("SSN=ST2110-20:2017", "SSN=ST2110-20:2017; interlace")
    with pytest.raises(ValueError, match="interlace"):
        parse_group_sdp(text)


def test_a_sub_image_two_sample_interleave_cannot_have_come_from_is_refused():
    """An odd sub-image width is a split of no essence: interleave deals
    whole pixel pairs."""
    text = _ANNEX_B.replace("width=1920", "width=1919")
    with pytest.raises(ValueError, match="divide"):
        parse_group_sdp(text)


def test_the_temporal_division_is_refused_by_name():
    """Annex A's six-phase 720p300 document: phases carry differing RTP
    timestamps, a frame schedule this library has no model of."""
    with pytest.raises(ValueError, match="PHASED"):
        parse_group_sdp(_annex("PHASED", 6))


# RP 2110-23 Annex D, reduced to its session lines: ST 2022-7 over a group is
# a primary and a secondary group line and a DUP line per constituent.
_ANNEX_D = (
    _ANNEX_HEADER
    + "a=group:MULTI-SD 1P 2P 3P 4P\n"
    + "a=group:MULTI-SD 1S 2S 3S 4S\n"
    + "".join(f"a=group:DUP {n}P {n}S\n" for n in range(1, 5))
    + "".join(_annex_block(f"{n}P", f"239.252.0.{n - 1}") for n in range(1, 5))
    + "".join(_annex_block(f"{n}S", f"239.252.1.{n - 1}") for n in range(1, 5))
)


def test_a_redundant_group_is_refused_by_name():
    with pytest.raises(ValueError, match="2022-7"):
        parse_group_sdp(_ANNEX_D)


def test_the_pair_parse_refuses_a_redundant_group_too():
    """Half of that document is a pair and half a group, and neither parse
    reads the whole."""
    with pytest.raises(ValueError, match="RP 2110-23"):
        parse_dup_sdp(_ANNEX_D)


@pytest.mark.parametrize("parse", [parse_sdp, parse_video_format])
def test_the_single_flow_parses_refuse_a_group_naming_the_group_parse(parse):
    """Read as one flow, a four-way offer is a quarter of an essence joined
    with success reported."""
    with pytest.raises(ValueError, match="parse_group_sdp"):
        parse(_ANNEX_B)


def test_a_single_flow_offer_is_not_a_group():
    single = _ANNEX_HEADER + _annex_block("1", "239.252.0.0")
    with pytest.raises(ValueError, match="parse_sdp"):
        parse_group_sdp(single)


def test_a_tag_naming_no_media_block_is_refused():
    text = _ANNEX_B.replace("a=mid:4", "a=mid:5")
    with pytest.raises(ValueError, match="'4'"):
        parse_group_sdp(text)


def test_a_media_block_the_group_does_not_name_is_refused():
    text = _ANNEX_B + _annex_block("5", "239.252.0.9")
    with pytest.raises(ValueError, match="5 'm=video' blocks"):
        parse_group_sdp(text)


def test_two_constituents_naming_one_address_port_and_sender_are_refused():
    """No receiver can tell the two flows apart."""
    text = _ANNEX_B.replace("239.252.0.1\n", "239.252.0.0\n")
    with pytest.raises(ValueError, match="cannot tell"):
        parse_group_sdp(text)


def test_constituents_sharing_a_multicast_address_on_distinct_ports_warn():
    """Section 5.6 binds a sender to a distinct multicast address each; two
    flows on distinct ports are still two flows, so the read goes on."""
    blocks = _ANNEX_B.split("m=video")
    blocks[2] = (
        blocks[2].replace(" 30000 ", " 30002 ").replace("239.252.0.1", "239.252.0.0")
    )
    with pytest.warns(SdpConformanceWarning, match=r"5\.6"):
        group = parse_group_sdp("m=video".join(blocks))
    assert group.flows[1] == SdpFlow("239.252.0.0", 30002)


def test_constituents_of_differing_troff_warn_and_read_the_first():
    """Section 5.5 binds a sender to one TRoffset for every constituent."""
    blocks = _ANNEX_B.split("m=video")
    blocks[1] = blocks[1].replace("SSN=ST2110-20:2017", "SSN=ST2110-20:2017; TROFF=40")
    with pytest.warns(SdpConformanceWarning, match="TROFF"):
        group = parse_group_sdp("m=video".join(blocks))
    assert group.video.tr_offset_us == 40


def test_a_constituent_without_its_own_fmtp_line_is_named_by_its_tag():
    text = _ANNEX_B.replace(_ANNEX_FMTP + "\na=mid:3", "a=mid:3")
    with pytest.raises(ValueError, match="'3'"):
        parse_group_sdp(text)
