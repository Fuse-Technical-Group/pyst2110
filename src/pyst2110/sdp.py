"""SDP: the little of RFC 4566 that names a flow and its video format.

Enough to read the addresses out of an ST 2110 offer, and enough of
ST 2110-20's ``a=fmtp:`` to size the frame a transmitter is about to pace
(SPEC §spec:sdp). Both directions: a receiver reads an SDP it is handed, and
a transmitter is configured by one it produces.

An offer carries the required media type parameters of two documents:
ST 2110-20 section 7.2 and the ``TP`` of ST 2110-21 section 8.1.

A redundant pair is one offer and not two. RFC 7104 groups the two legs of
ST 2022-7 with a session-level ``a=group:DUP``, and :func:`parse_dup_sdp` and
:func:`format_dup_sdp` are that document both ways (§spec:redundancy).

So is one essence divided over several flows. SMPTE RP 2110-23 groups four or
sixteen constituent flows with ``a=group:MULTI-2SI`` or ``a=group:MULTI-SD``,
and :func:`parse_group_sdp` and :func:`format_group_sdp` are that document
both ways (§spec:multi-stream).

Colorimetry is carried through, not interpreted: what a consumer does with
``BT2020`` is its own concern, and this records which token the SDP said.
"""

from __future__ import annotations

import ipaddress
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields
from fractions import Fraction

from pyst2110 import _layout
from pyst2110.division import CONSTITUENT_COUNTS, MULTI_2SI, MULTI_SD, sub_image_size

__all__ = [
    "RTP_CLOCK_RATE",
    "SENDER_TYPES",
    "SENDER_TYPE_NARROW",
    "SENDER_TYPE_NARROW_LINEAR",
    "SENDER_TYPE_WIDE",
    "STANDARD_UDP_SIZE_LIMIT",
    "SdpConformanceWarning",
    "SdpFlow",
    "SdpGroup",
    "SdpVideo",
    "format_dup_sdp",
    "format_group_sdp",
    "format_sdp",
    "parse_dup_sdp",
    "parse_group_sdp",
    "parse_sdp",
    "parse_video_format",
]

#: The RTP clock rate ST 2110-20 section 6.1.3 fixes for video: "The RTP Clock
#: rate for streams compliant to this standard shall be 90 kHz."
RTP_CLOCK_RATE = 90_000

#: The Standard UDP Size Limit of SMPTE ST 2110-10: what a 1500-octet MTU
#: leaves after a 40-octet IPv6 header, counting the 8-octet UDP header within
#: itself. ST 2110-20 section 7.3 makes an absent ``MAXUDP`` mean exactly this,
#: so it is the default and never written out.
STANDARD_UDP_SIZE_LIMIT = 1460

# ST 2110-20 section 6.3.2. The geometry this library computes tiles a line
# with equal packets and sets no Line Continuation bit, which is the General
# Packing Mode; the Block Packing Mode's 180-octet rule is a different sizing
# and would be a false claim here (§spec:geometry).
_PACKING_MODE = "2110GPM"

# ST 2110-20 section 7.2: the 2017 revision unless the colorimetry value ALPHA
# or the TCS value ST2115LOGS3 is used. TCS is not modelled, so colorimetry is
# the whole of the test this library can make.
_STANDARD_NUMBER = "ST2110-20:2017"
_STANDARD_NUMBER_2022 = "ST2110-20:2022"
_ALPHA = "ALPHA"

# ST 2110-20 section 7.5 has a token for a colorimetry nobody stated, so an SDP
# that omits the required parameter is recorded rather than guessed over.
_UNSPECIFIED = "UNSPECIFIED"

#: A Narrow sender (SMPTE ST 2110-21 section 7.1.2): the gapped packet read
#: schedule, sending nothing during the interval a blanked raster would take.
SENDER_TYPE_NARROW = "2110TPN"

#: A Narrow Linear sender (section 7.1.3): the same buffer model over the
#: linear packet read schedule, spread evenly across the whole frame.
SENDER_TYPE_NARROW_LINEAR = "2110TPNL"

#: A Wide sender (section 7.1.4): the linear schedule under a buffer model
#: some ninety times looser, for senders whose pacing software decides.
SENDER_TYPE_WIDE = "2110TPW"

#: Every value ``TP`` may take. Section 7.1.1: a sender "shall conform to one
#: or more of the types defined in clauses 7.1.2, 7.1.3, or 7.1.4", and each
#: of those clauses fixes the token its type signals.
SENDER_TYPES = (SENDER_TYPE_NARROW, SENDER_TYPE_NARROW_LINEAR, SENDER_TYPE_WIDE)

# ST 2110-20 section 7.2: width and height are "integers between 1 and 32767
# inclusive" — which is also all the SRD Row Number and Offset fields hold.
_MAX_RASTER = _layout.VALUE_MASK
# A UDP port and a UDP datagram's length are both sixteen-bit fields, so no
# port or MAXUDP above this describes one that can exist.
_MAX_PORT = _layout.U16_MODULUS - 1
_MAX_UDP_SIZE = _layout.U16_MODULUS - 1
# RFC 3550 section 5.1 gives the payload type seven bits.
_MAX_PAYLOAD_TYPE = _layout.PAYLOAD_TYPE_MASK
_DEFAULT_PAYLOAD_TYPE = _layout.DYNAMIC_PAYLOAD_TYPE
_MAX_TTL = 255
# ST 2110-21 section 8.2's TROFF is expressed in microseconds; its CMAX is
# "an integer number", bounded here only by the field a count could fill.
_MICROSECONDS_PER_SECOND = 1_000_000
_MAX_CMAX = _layout.U32_MODULUS - 1
# RFC 4566 section 5.2 wants the session id "based on a 64-bit NTP timestamp".
_MAX_SESSION_ID = (1 << 64) - 1
# ST 2110-20 section 7.4.2 lists the depths a sender may declare. 16f is
# half-float and not modelled here (§road:future).
_DEPTHS = (8, 10, 12, 16)

# RFC 7104 section 5: the grouping semantics that mean one stream sent twice,
# carried on the ``a=group:`` attribute RFC 5888 defines. Other semantics —
# LS, FID — ride the same attribute and mean other things.
_DUP = "DUP"

# ST 2022-7 sends one essence by two paths, and two is what this library
# models throughout: pyst2110.redundancy reconstructs a pair, and Rivermax
# caps RMX_MAX_DUP_STREAMS at two. A group naming three legs describes a
# document nothing downstream of the parse can take.
_DUP_LEGS = 2

# RP 2110-23 section 5.3: the three grouping semantics it defines on RFC
# 5888's attribute, each meaning one essence divided over several flows.
# PHASED divides time, and is recognised only to be refused by name.
_PHASED = "PHASED"
_RP_GROUPS = (_PHASED, MULTI_2SI, MULTI_SD)

# ST 2110-20 section 7.3: an absent TCS means SDR, so two constituents that
# differ only in whether they spell it out carry the same transfer.
_DEFAULT_TCS = "SDR"

# RFC 4175 section 7: a payload format's name goes in ``a=rtpmap`` as the
# encoding name, and ST 2110-20's is ``raw``. RFC 4855 section 3: encoding
# names "are case-insensitive".
_RAW_ENCODING = "raw"


class SdpConformanceWarning(UserWarning):
    """An offer was read that a conforming sender would not have written.

    Strict in what is sent, tolerant in what is read: every offer this
    library writes carries what the standards require of a sender, and an
    offer that omits something is read the one way it can be — and this
    warning says what was missing and what was assumed. A consumer routes it
    into its own logging, or makes it an error, with the ``warnings`` module
    (§spec:sdp).
    """


# Every character ``str.splitlines()`` starts a new line at. RFC 4566 ends a
# record with CRLF alone, but a reader that splits the document into lines —
# as :func:`parse_sdp` does — begins a record at any of these, so a caller's
# string carrying one declares a record of its own.
_LINE_TERMINATORS = frozenset("\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029")


@dataclass(frozen=True)
class SdpFlow:
    """Where an SDP says the packets are, and who is allowed to send them."""

    destination_ip: str
    destination_port: int
    source_ip: str = ""


@dataclass(frozen=True)
class SdpVideo:
    """The ST 2110-20 format parameters that decide how a frame is packed."""

    width: int
    height: int
    frame_rate: Fraction
    depth: int
    sampling: str
    #: The token the SDP said, uninterpreted. ``UNSPECIFIED`` where the SDP
    #: named none, which ST 2110-20 section 7.5 defines as that very case.
    colorimetry: str = _UNSPECIFIED
    interlaced: bool = False
    #: The largest UDP payload the sender will emit, from ``MAXUDP``.
    max_udp: int = STANDARD_UDP_SIZE_LIMIT
    #: The sender's ST 2110-21 type, from ``TP``: one of :data:`SENDER_TYPES`.
    #: It describes the pacing of whatever puts the packets on the wire, so
    #: the caller that owns the pacer owns the value (SPEC §spec:sdp). Narrow
    #: where an offer names none, that being the strictest type and the one
    #: a hardware pacer fed by this library's headers is.
    sender_type: str = SENDER_TYPE_NARROW
    #: ``TROFF``, ST 2110-21 section 8.2: the sender's TR_OFFSET in whole
    #: microseconds where it differs from the type's default; ``None`` is the
    #: default, which :func:`pyst2110.timing.read_schedule` computes.
    tr_offset_us: int | None = None
    #: ``CMAX``, section 8.2: the largest C_INST the sender claims to emit;
    #: ``None`` leaves the claim at the type's own limit. Carried, not
    #: enforced — a sender's claim is what a measurement is compared with.
    cmax: int | None = None

    @property
    def frame_interval_ns(self) -> int:
        """Nanoseconds between successive frames, rounded to the nearest."""
        return round(Fraction(1_000_000_000) / self.frame_rate)


@dataclass(frozen=True)
class SdpGroup:
    """One essence carried as an RP 2110-23 group of constituent flows.

    ``flows`` are the constituents in group order, which is constituent order:
    the first carries constituent 1 of :mod:`pyst2110.division`'s numbering.
    ``video`` is the one format every constituent shares, and so describes a
    sub-image; the essence raster is derived, because no line of the document
    carries it.
    """

    #: The RP's grouping semantics: :data:`~pyst2110.division.MULTI_2SI` or
    #: :data:`~pyst2110.division.MULTI_SD`.
    division: str
    flows: tuple[SdpFlow, ...]
    video: SdpVideo

    @property
    def essence_width(self) -> int:
        """Pixels across the essence: two sub-images for four constituents,
        four for sixteen."""
        return self.video.width * math.isqrt(len(self.flows))

    @property
    def essence_height(self) -> int:
        """Rows down the essence, by the same count as :attr:`essence_width`."""
        return self.video.height * math.isqrt(len(self.flows))


def parse_sdp(text: str) -> SdpFlow:
    """Extract the receive flow from an SDP.

    Reads the connection address (``c=``), the media port (``m=``) and the
    source filter (``a=source-filter``). Raises ``ValueError`` naming what is
    missing, because an SDP that cannot describe a flow is the caller's
    problem to fix rather than something to guess at.

    One flow, so a document RFC 7104 grouped into a redundant pair is refused
    rather than answered with whichever leg came first: a caller reading a
    destination out of a two-leg offer this way joins — or sends — one leg of
    two and reports success (§spec:redundancy). :func:`parse_dup_sdp` reads
    that document. Nothing else changes what an offer without the group means,
    the ``20000/2`` port count RFC 4566 permits included.
    """
    session, media = _sections(text)
    _refuse_division(session)
    if _dup_tags(session) is not None:
        raise ValueError(
            "the SDP groups its media into an RFC 7104 redundant pair with a "
            "session-level 'a=group:DUP' line, which is two flows and not one: "
            "parse_dup_sdp reads both legs"
        )

    videos = _video_sections(session, media)
    if not videos:
        raise ValueError("the SDP has no 'm=video <port> ...' media section")

    # The first video section, not last-wins over the document: a 2110 SDP
    # carrying video then audio would otherwise yield the audio port, and a
    # flow attached to it receives nothing.
    video = videos[0]
    if not video.connection:
        raise ValueError("the SDP has no 'c=IN IP4 <address>' connection line")
    if not video.port:
        raise ValueError("the SDP's video section declares port 0, which disables it")
    return SdpFlow(video.connection, video.port, video.source)


def parse_dup_sdp(text: str) -> tuple[SdpFlow, SdpFlow]:
    """Extract both legs of an ST 2022-7 redundant pair from one offer.

    RFC 7104 groups the legs with a session-level ``a=group:DUP`` naming the
    ``a=mid:`` tags of two ``m=video`` blocks, each block carrying its own
    ``c=`` and ``a=source-filter``. **The legs come back in the order the
    group names them**, not in the order the blocks appear: that order is what
    decides which leg is which everywhere downstream (§spec:redundancy).

    A document whose tags and media blocks disagree is refused. Read as a
    single-leg offer with an oddity it would put one leg on the wire where two
    were meant, which is unprotected essence sent and success reported — the
    silent failure a refusal here is for. Rivermax's
    ``rmx_output_media_set_sdp`` states the same rule from the other side: the
    count of ``DUP`` tags has to correspond to the count of ``m=video``
    blocks.

    Raises ``ValueError`` naming what disagrees, and for an offer carrying no
    group at all — that is a single-leg document, which :func:`parse_sdp`
    reads.
    """
    session, media = _sections(text)
    tags = _dup_tags(session)
    if tags is None:
        raise ValueError(
            "the SDP has no session-level 'a=group:DUP <tag> <tag>' line, so "
            "it describes one leg and not a redundant pair: parse_sdp reads it"
        )

    if _division_groups(session):
        raise ValueError(
            "the SDP carries ST 2022-7 over an RP 2110-23 group (section 5.4), "
            "a primary and a secondary group of constituents paired by 'DUP' "
            "lines, which neither the pair parse nor parse_group_sdp reads"
        )

    sections = _tagged_sections(_DUP, tags, _video_sections(session, media))
    if len(tags) != _DUP_LEGS:
        raise ValueError(
            f"the SDP's 'a=group:DUP' names {len(tags)} legs; ST 2022-7 as "
            f"this library models it is two, which is also all Rivermax's "
            f"RMX_MAX_DUP_STREAMS carries"
        )
    legs = [_flow(section, tag) for section, tag in zip(sections, tags, strict=True)]
    first, second = legs

    # The reader refuses what the writer refuses. Two blocks that name one
    # socket are one path described twice, and a receiver provisioned from
    # them joins the same group twice while its operator is told the essence
    # is protected — so disrupting the one path blacks it. Whole-flow
    # equality is the test rather than the address alone, because two legs
    # onto one group from different senders are a source-specific pair and
    # genuinely two paths (§spec:redundancy).
    if first == second:
        raise ValueError(
            f"both of the SDP's legs name {first.destination_ip} port "
            f"{first.destination_port} and the same sender, which is one path "
            f"described twice rather than the two ST 2022-7 protects with"
        )
    return first, second


def parse_group_sdp(text: str) -> SdpGroup:
    """Extracts an RP 2110-23 group: its division, its constituent flows in
    group order, and the one sub-image format they share.

    RFC 5888's session-level ``a=group:`` names the division — ``MULTI-2SI``
    or ``MULTI-SD`` — and the ``a=mid:`` tag of four or sixteen ``m=video``
    blocks, each with its own ``c=``, source filter and fmtp line. **Position
    in the group line is the constituent number**, whatever order the blocks
    appear in, and a decimal tag naming another position is refused: the RP
    numbers its tags as it numbers the constituents, and a document saying
    two things about which sub-image a flow carries is one a receiver
    assembles wrong (§spec:multi-stream).

    Raises ``ValueError`` naming the cause for constituents of unequal
    format, which have no defined essence raster; for interlaced ones, the RP
    defining no field structure under division; for two constituents naming
    one address, port and sender, which no receiver can tell apart; for the
    temporal ``PHASED`` division and for ST 2022-7 over a group, neither of
    which is modelled; and for a document carrying no group, which
    :func:`parse_sdp` reads.

    What RP 2110-23 binds a sender to and a receiver can still read is taken
    with an :class:`SdpConformanceWarning`: constituents of differing
    ``TROFF``, against section 5.5, read as the first constituent's; and
    constituents sharing a multicast address, against section 5.6.
    """
    session, media = _sections(text)
    groups = _division_groups(session)
    if not groups:
        raise ValueError(
            f"the SDP has no session-level 'a=group:{MULTI_2SI}' or "
            f"'a=group:{MULTI_SD}' line, so it describes no RP 2110-23 group: "
            f"parse_sdp reads a single flow"
        )
    division, tags = groups[0]
    if division == _PHASED:
        raise ValueError(
            "the SDP's 'a=group:PHASED' divides an essence in time, each phase "
            "carrying its own RTP timestamps, and that frame schedule is not "
            f"modelled here; parse_group_sdp reads {MULTI_2SI} and {MULTI_SD}"
        )
    if len(groups) > 1 or _dup_tags(session) is not None:
        raise ValueError(
            "the SDP carries ST 2022-7 over an RP 2110-23 group (section 5.4), "
            "a primary and a secondary group of constituents paired by 'DUP' "
            "lines, which is not read here"
        )
    if len(tags) not in CONSTITUENT_COUNTS:
        raise ValueError(
            f"the SDP's 'a=group:{division}' names {len(tags)} constituents; "
            f"RP 2110-23 divides an essence into 4 or 16"
        )
    for position, tag in enumerate(tags, start=1):
        if _decimal(tag) and int(tag) != position:
            raise ValueError(
                f"the SDP's 'a=group:{division}' puts tag {tag!r} in position "
                f"{position}; RP 2110-23 numbers each constituent's tag by its "
                f"position, so the document names two sub-images for one flow"
            )

    sections = _tagged_sections(division, tags, _video_sections(session, media))
    flows = tuple(
        _flow(section, tag) for section, tag in zip(sections, tags, strict=True)
    )
    _check_constituent_flows(flows, tags)

    formats = []
    for section, tag in zip(sections, tags, strict=True):
        try:
            formats.append(_video_format(list(section.lines), stacklevel=4))
        except ValueError as exc:
            raise ValueError(f"the SDP's {tag!r} constituent: {exc}") from exc
    video = _shared_format(formats, tags)
    sub_image_size(
        video.width * math.isqrt(len(tags)),
        video.height * math.isqrt(len(tags)),
        division=division,
        count=len(tags),
    )
    return SdpGroup(division, flows, video)


def _check_constituent_flows(flows: Sequence[SdpFlow], tags: Sequence[str]) -> None:
    """Refuse constituents no receiver can tell apart, and warn of ones
    sharing a multicast address, which section 5.6 forbids a sender."""
    seen: dict[SdpFlow, str] = {}
    groups: dict[str, str] = {}
    for flow, tag in zip(flows, tags, strict=True):
        if flow in seen:
            raise ValueError(
                f"the SDP's {seen[flow]!r} and {tag!r} constituents both name "
                f"{flow.destination_ip} port {flow.destination_port} and the "
                f"same sender, so a receiver cannot tell their packets apart"
            )
        seen[flow] = tag
        address = ipaddress.ip_address(flow.destination_ip)
        if not address.is_multicast:
            continue
        if flow.destination_ip in groups:
            warnings.warn(
                f"the SDP's {groups[flow.destination_ip]!r} and {tag!r} "
                f"constituents share the multicast address "
                f"{flow.destination_ip}; RP 2110-23 section 5.6 requires a "
                f"distinct one for each",
                SdpConformanceWarning,
                stacklevel=3,
            )
        groups.setdefault(flow.destination_ip, tag)


def _shared_format(
    formats: Sequence[tuple[SdpVideo, dict[str, str]]], tags: Sequence[str]
) -> SdpVideo:
    """The one format every constituent declares, refusing a group of unequal
    ones and an interlaced one; a differing ``TROFF`` warns and reads as the
    first constituent's (RP 2110-23 section 5.5)."""
    first, first_parameters = formats[0]
    if first.interlaced:
        raise ValueError(
            "the SDP's constituents are interlaced, and RP 2110-23 defines no "
            "field structure under division"
        )
    for (video, parameters), tag in zip(formats[1:], tags[1:], strict=True):
        for name in (one.name for one in fields(SdpVideo)):
            if name == "tr_offset_us":
                continue
            if getattr(video, name) != getattr(first, name):
                raise ValueError(
                    f"the SDP's {tag!r} constituent declares a {_label(name)} of "
                    f"{getattr(video, name)} where the first declares "
                    f"{getattr(first, name)}; a division assembles one essence "
                    f"only from equal sub-images"
                )
        tcs = parameters.get("TCS", _DEFAULT_TCS)
        first_tcs = first_parameters.get("TCS", _DEFAULT_TCS)
        if tcs != first_tcs:
            raise ValueError(
                f"the SDP's {tag!r} constituent declares a TCS of {tcs} where "
                f"the first declares {first_tcs}; a division assembles one "
                f"essence only from equal sub-images"
            )
    if any(video.tr_offset_us != first.tr_offset_us for video, _ in formats):
        warnings.warn(
            "the SDP's constituents declare differing TROFF values; RP 2110-23 "
            "section 5.5 requires one for every constituent, and the first "
            "constituent's is read",
            SdpConformanceWarning,
            stacklevel=3,
        )
    return first


def _label(name: str) -> str:
    """An SdpVideo field as the error names it."""
    return name.replace("_", " ")


def _division_groups(session: list[str]) -> list[tuple[str, tuple[str, ...]]]:
    """Every session-level ``a=group:`` carrying one of RP 2110-23's
    semantics, as its semantics and tags, in document order."""
    groups = []
    for line in session:
        if not line.startswith("a=group:"):
            continue
        words = line[len("a=group:") :].split()
        if words[:1] and words[0] in _RP_GROUPS:
            groups.append((words[0], tuple(words[1:])))
    return groups


def _refuse_division(session: list[str]) -> None:
    """Refuse an RP 2110-23 group where one flow is asked for."""
    groups = _division_groups(session)
    if groups:
        raise ValueError(
            f"the SDP groups its media with a session-level 'a=group:"
            f"{groups[0][0]}' line, RP 2110-23's division of one essence over "
            f"several flows, so one flow would be part of it: parse_group_sdp "
            f"reads the group"
        )


def _tagged_sections(
    semantics: str, tags: Sequence[str], sections: Sequence[_MediaSection]
) -> list[_MediaSection]:
    """The media sections a group line's tags name, in the group's order.

    A document whose tags and media blocks disagree is refused: read anyway,
    it is one a receiver joins part of while reporting the whole.
    """
    if len(tags) != len(sections):
        raise ValueError(
            f"the SDP's 'a=group:{semantics}' names {len(tags)} tags over "
            f"{len(sections)} 'm=video' blocks, and RFC 5888 groups one block "
            f"per tag: a document that says two things about how many flows "
            f"it has is one a receiver joins part of"
        )
    if len(set(tags)) != len(tags):
        raise ValueError(
            f"the SDP's 'a=group:{semantics}' names {' '.join(tags)}, which "
            f"names one block twice"
        )
    blocks: dict[str, _MediaSection] = {}
    for section in sections:
        if section.mid in blocks:
            raise ValueError(
                f"two of the SDP's 'm=video' blocks carry 'a=mid:"
                f"{section.mid}', so a tag naming that block names both"
            )
        blocks[section.mid] = section
    missing = [tag for tag in tags if tag not in blocks]
    if missing:
        raise ValueError(
            f"the SDP's 'a=group:{semantics}' names {missing[0]!r}, which no "
            f"'m=video' block carries an 'a=mid:' for"
        )
    return [blocks[tag] for tag in tags]


def _flow(section: _MediaSection, tag: str) -> SdpFlow:
    """One leg's flow, refusing a block that does not describe one."""
    if not section.connection:
        raise ValueError(f"the SDP's {tag!r} block has no connection address")
    if not section.port:
        raise ValueError(f"the SDP's {tag!r} block declares port 0, which disables it")
    return SdpFlow(section.connection, section.port, section.source)


@dataclass(frozen=True)
class _MediaSection:
    """One ``m=video`` block, with whatever it inherits from the session."""

    port: int
    connection: str
    source: str
    mid: str
    #: The block's own lines, from its ``m=`` line on.
    lines: tuple[str, ...] = field(default=(), compare=False)


def _sections(text: str) -> tuple[list[str], list[list[str]]]:
    """A document's session-level lines and its media sections.

    RFC 4566 section 5: everything before the first ``m=`` is session level,
    and every line after one belongs to that media section until the next.
    """
    session: list[str] = []
    media: list[list[str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("m="):
            media.append([line])
        elif media:
            media[-1].append(line)
        else:
            session.append(line)
    return session, media


def _video_sections(session: list[str], media: list[list[str]]) -> list[_MediaSection]:
    """Every ``m=video`` block, in document order.

    A media section's own attributes override the session-level ones "for the
    respective media" (RFC 4566 section 5.7), so a block without a ``c=`` of
    its own takes the session's — which is how RFC 7104's first example is
    written, one group address for both legs.
    """
    connection = _attribute(session, "c=", _connection_address)
    source = _attribute(session, "a=source-filter", _source_address)
    return [
        _MediaSection(
            port=_media_port(block[0]),
            connection=_attribute(block, "c=", _connection_address) or connection,
            source=_attribute(block, "a=source-filter", _source_address) or source,
            mid=_attribute(block, "a=mid:", _mid),
            lines=tuple(block),
        )
        for block in media
        if _is_video(block[0])
    ]


def _is_video(line: str) -> bool:
    """``m=video 20000 RTP/AVP 96`` — whether this media line opens video.

    One definition, because every reader here scopes itself to the video
    section and a media type read three ways is a media type two of them can
    get wrong.
    """
    return line[len("m=") :].split()[:1] == ["video"]


def _attribute(lines: list[str], prefix: str, read: Callable[[str], str]) -> str:
    """The last line with this prefix, read — or the empty string for none."""
    values = [read(line) for line in lines if line.startswith(prefix)]
    return values[-1] if values else ""


def _mid(line: str) -> str:
    """``a=mid:S1a`` — RFC 5888's identification tag for a media section."""
    return line[len("a=mid:") :].strip()


def _dup_tags(session: list[str]) -> tuple[str, ...] | None:
    """The tags an ``a=group:DUP`` names, or ``None`` where none does.

    RFC 5888 makes ``a=group:`` a session-level attribute carrying semantics
    and then identification tags, and RFC 7104 section 5 defines ``DUP`` as
    the semantics for duplication. Another semantics on the same attribute —
    ``LS``, ``FID`` — is somebody else's grouping and not this one.
    """
    for line in session:
        if not line.startswith("a=group:"):
            continue
        fields = line[len("a=group:") :].split()
        if fields[:1] == [_DUP]:
            return tuple(fields[1:])
    return None


def _connection_address(line: str) -> str:
    """``c=IN IP4 239.1.1.1/64`` — the multicast group, minus its TTL."""
    fields = line[len("c=") :].split()
    if len(fields) < 3:
        return ""
    return fields[2].split("/", 1)[0]


def _media_port(line: str) -> int:
    """``m=video 20000 RTP/AVP 96`` — the port, ignoring the rest.

    RFC 4566's port field is ``port ["/" integer]``: ST 2022-7 offers and
    hierarchical encodings carry a count after a slash. Taking the field
    whole would read ``20000/2`` as non-numeric and report the line missing.
    """
    fields = line[len("m=") :].split()
    if len(fields) < 2:
        return 0
    port = fields[1].split("/", 1)[0]
    if not _decimal(port):
        return 0
    return int(port)


def parse_video_format(text: str) -> SdpVideo:
    """Extract the ST 2110-20 video format from an SDP's ``a=fmtp:`` line.

    Raises ``ValueError`` naming what is missing. A transmitter needs the
    width, height and rate to know how many packets a frame is and when the
    next one departs; guessing any of them would put the wrong number of
    packets on the wire at the wrong time.

    Also raises where the ``a=rtpmap`` for the payload type the fmtp line
    describes names an encoding other than ``raw``, or ``raw`` at a clock
    other than 90 kHz: those parameters describe a raster only under
    ST 2110-20's encoding. An offer mapping no encoding is read as raw at
    90 kHz, with a :class:`SdpConformanceWarning`.

    One flow's format, so a document dividing one essence over several flows
    with an RP 2110-23 group is refused: each constituent's fmtp line
    describes a sub-image, and read alone it sizes a quarter of the picture
    (§spec:multi-stream). :func:`parse_group_sdp` reads that document.
    """
    session, media = _sections(text)
    _refuse_division(session)
    # The first video section's: a multi-essence SDP carries an a=fmtp: per
    # essence, and reading the audio one would describe a frame geometry that
    # does not exist.
    block = next((block for block in media if _is_video(block[0])), [])
    # Past _video_format and _require_raw, to the caller's line.
    return _video_format(block, stacklevel=4)[0]


def _video_format(
    block: list[str], *, stacklevel: int
) -> tuple[SdpVideo, dict[str, str]]:
    """One ``m=video`` block's format, and the fmtp parameters it came from.

    ``stacklevel`` is the frame a conformance warning names, counted from
    :func:`_require_raw`.
    """
    parameters = _format_parameters(block)
    if not parameters:
        raise ValueError("the SDP has no 'a=fmtp:<payload> ...' format line")
    _require_raw(block, stacklevel=stacklevel)

    missing = [name for name in ("width", "height") if name not in parameters]
    if missing:
        raise ValueError(f"the SDP's fmtp line has no {' or '.join(missing)}")
    if "exactframerate" not in parameters:
        raise ValueError("the SDP's fmtp line has no exactframerate")

    # Bounded here as well as on the way out: a width the emitter refuses is
    # one the transmit path cannot put an SRD Offset in either, and the two
    # sides of one parameter are worth no two different answers.
    width = int(parameters["width"])
    height = int(parameters["height"])
    # Section 7.3: absent means the Standard UDP Size Limit is in use.
    max_udp = int(parameters.get("MAXUDP", STANDARD_UDP_SIZE_LIMIT))
    _integer(width, "width", 1, _MAX_RASTER)
    _integer(height, "height", 1, _MAX_RASTER)
    _integer(max_udp, "MAXUDP", 1, _MAX_UDP_SIZE)
    frame_rate = _frame_rate(parameters["exactframerate"])

    # ST 2110-21 section 8: TP is required and TROFF and CMAX optional, each
    # optional one meaning its default by its absence. An offer without TP is
    # one section 8.1 does not permit; Narrow is what it is read as, for the
    # reasons SdpVideo.sender_type gives.
    sender_type = parameters.get("TP", SENDER_TYPE_NARROW)
    validate_sender_type(sender_type)
    tr_offset_us = _optional_integer(parameters, "TROFF")
    if tr_offset_us is not None:
        _tr_offset(tr_offset_us, frame_rate)
    cmax = _optional_integer(parameters, "CMAX")
    if cmax is not None:
        _integer(cmax, "CMAX", 1, _MAX_CMAX)

    video = SdpVideo(
        width=width,
        height=height,
        frame_rate=frame_rate,
        depth=int(parameters.get("depth", 10)),
        sampling=parameters.get("sampling", ""),
        colorimetry=parameters.get("colorimetry", _UNSPECIFIED),
        # A flag with no value: SMPTE ST 2110-20 writes bare "interlace".
        interlaced="interlace" in parameters,
        max_udp=max_udp,
        sender_type=sender_type,
        tr_offset_us=tr_offset_us,
        cmax=cmax,
    )
    return video, parameters


def _optional_integer(parameters: dict[str, str], name: str) -> int | None:
    """A media type parameter's value as a whole number, or ``None`` if absent.

    ``int()`` accepts more than a parameter may carry — a sign, surrounding
    whitespace, underscores — so the digits are checked first, and the error
    names the parameter rather than the text ``int()`` choked on.
    """
    text = parameters.get(name)
    if text is None:
        return None
    if not _decimal(text):
        raise ValueError(f"a {name} of {text!r} is not a whole number")
    return int(text)


def _decimal(text: str) -> bool:
    """Whether ``text`` is the ASCII decimal spelling of a whole number.

    ``str.isdigit()`` alone is not that test, and fails it in both directions.
    It is true of superscripts, which ``int()`` then refuses — so a gate
    written to name its parameter let ``int()`` choke instead. And it is true
    of every script's decimal digits, which ``int()`` accepts — so ``TROFF=٣``
    was read as 3, a value the document does not say.
    """
    return text.isascii() and text.isdigit()


def _require_raw(block: list[str], *, stacklevel: int) -> None:
    """Refuse a video offer whose payload is not ST 2110-20's ``raw``.

    A compressed flow's fmtp line carries a width, a height and a depth too
    — ST 2110-22 flows name the picture they code — so nothing in the format
    parameters tells a raster from a codestream. The rtpmap does.
    """
    mapping = _video_payload_mapping(block)
    if mapping is None:
        warnings.warn(
            "the SDP maps no encoding to its video payload type with an "
            f"a=rtpmap line; ST 2110-20 section 7.1 requires "
            f"{_RAW_ENCODING}/{RTP_CLOCK_RATE}, and it is read as that",
            SdpConformanceWarning,
            stacklevel=stacklevel,
        )
        return
    encoding, clock = mapping
    if encoding.lower() != _RAW_ENCODING:
        raise ValueError(
            f"the SDP's video is {encoding!r}, not ST 2110-20's {_RAW_ENCODING!r}: "
            "its fmtp line describes a coded payload, not a raster"
        )
    if not clock.isdigit() or int(clock) != RTP_CLOCK_RATE:
        raise ValueError(
            f"the SDP's rtpmap gives raw a {clock!r} clock; ST 2110-20 section "
            f"7.1 requires {RTP_CLOCK_RATE}"
        )


def _video_payload_mapping(block: list[str]) -> tuple[str, str] | None:
    """The encoding name and clock rate a video block's ``a=rtpmap`` gives
    the payload type its fmtp line names, or ``None`` where it maps none."""
    fmtp = next((line for line in block if line.startswith("a=fmtp:")), None)
    if fmtp is None:
        return None
    words = fmtp[len("a=fmtp:") :].split(maxsplit=1)
    if not words:
        return None
    prefix = f"a=rtpmap:{words[0]} "
    rtpmap = next((line for line in block if line.startswith(prefix)), None)
    if rtpmap is None:
        return None
    # <encoding name>/<clock rate>[/<encoding parameters>], RFC 4566 section 6.
    encoding, _, rest = rtpmap[len(prefix) :].strip().partition("/")
    return encoding, rest.partition("/")[0]


def _format_parameters(block: list[str]) -> dict[str, str]:
    """``a=fmtp:96 width=1920; height=1080; interlace`` as a mapping.

    Bare flags map to the empty string, which is what makes ``interlace``
    detectable by presence. The block's first fmtp line is read.
    """
    fmtp = next((line for line in block if line.startswith("a=fmtp:")), None)
    if fmtp is None:
        return {}
    _, _, rest = fmtp.partition(" ")
    parameters: dict[str, str] = {}
    for entry in rest.split(";"):
        name, _, value = entry.strip().partition("=")
        if name:
            parameters[name] = value.strip()
    return parameters


def _frame_rate(text: str) -> Fraction:
    """``50``, or ``30000/1001`` for the fractional rates."""
    numerator, separator, denominator = text.partition("/")
    try:
        if separator:
            rate = Fraction(int(numerator), int(denominator))
        else:
            rate = Fraction(int(numerator))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"not an exactframerate: {text}") from exc
    validate_frame_rate(rate)
    return rate


def validate_frame_rate(rate: Fraction) -> None:
    """Refuse an ``exactframerate`` no sender could be running at.

    Bounded on both sides, and by the same helper both ways: the parse and the
    emit describe one flow, and two copies of one bound are two chances to
    disagree about it — which is what this was, the emit refusing a rate at or
    below zero while the parse admitted one and divided by it.

    A rate at or below zero is not a rate. Above, the bound is the 90 kHz RTP
    Clock ST 2110-20 section 6.1.3 fixes for video: a frame period shorter
    than one tick of it gives successive frames the same media timestamp, so
    the timebase the stream is carried on cannot express the rate. It is also
    what keeps the frame period from dividing to nothing in the arithmetic
    downstream of a parse (:mod:`pyst2110.timing`).
    """
    if not 0 < rate <= RTP_CLOCK_RATE:
        raise ValueError(
            f"an exactframerate of {rate} is outside 0-{RTP_CLOCK_RATE}, the "
            f"RTP Clock rate ST 2110-20 section 6.1.3 fixes for video"
        )


def _source_address(line: str) -> str:
    """``a=source-filter: incl IN IP4 <destination> <source>``.

    The colon may or may not be followed by a space, and several sources may
    be listed; the first is taken, since a flow filters on one sender.
    """
    _, _, rest = line.partition(":")
    fields = rest.split()
    if len(fields) < 5 or fields[0] != "incl":
        return ""
    return fields[4]


def format_sdp(
    flow: SdpFlow,
    video: SdpVideo,
    *,
    payload_type: int = _DEFAULT_PAYLOAD_TYPE,
    sender_type: str | None = None,
    any_source: bool = False,
    session_name: str = " ",
    session_id: int = 0,
    ttl: int = 64,
) -> str:
    """Write the SDP offer that describes this flow and format.

    The document RFC 4566 section 5 requires, in the order it requires, with
    the media type parameters of ST 2110-20 section 7.2 and ST 2110-21
    section 8.1 on the ``a=fmtp:`` line and the 90 kHz clock of ST 2110-20
    section 7.1 on the ``a=rtpmap:`` one. Records end with CRLF, as RFC 4566
    specifies.

    ``sender_type`` is the ``TP`` parameter, one of :data:`SENDER_TYPES`. It
    describes the pacing of whatever puts the packets on the wire, which is
    not this library (§spec:scope-boundary), so the caller that owns the pacer
    owns the value: ``video.sender_type`` by default, and this argument where
    a caller sets one on the way out. The default is documented in SPEC
    §spec:sdp. The optional ``TROFF`` and ``CMAX`` of ST 2110-21 section 8.2
    are written from ``video`` where it carries them, and absent otherwise —
    absence being what means the default.

    A **multicast** offer says who may send. ``SdpFlow.source_ip`` names the
    sender and writes the ``a=source-filter`` line ST 2110-10 section 8.4 asks
    a sender for; ``any_source=True`` offers the group to any sender and
    writes no such line, which RFC 4570 defines as accepting all of them.
    Naming neither raises, and naming both raises: one of the two is a choice,
    and a multicast offer that made it by accident is refused outright by at
    least one transmit SDK. A unicast destination has no group to filter and
    needs neither.

    ``session_id`` fills both the session identifier and its version in the
    ``o=`` line. It defaults to zero, which is repeatable rather than unique:
    RFC 4566 wants the tuple globally unique, so a caller offering several
    sessions passes its own. ``session_name`` defaults to the single space
    section 5.3 prescribes where a session has no meaningful name.

    **Pass a name if a sender is going to read this back.** The default is
    what the RFC prescribes, and at least one transmit SDK refuses it: NVIDIA
    Rivermax (1.90.18) logs ``'x=<token>' format not found`` and fails stream
    creation with ``RMX_INVALID_PARAM_MIX``, while accepting the same document
    with any non-blank name — measured at 1080p60 and 2160p24, so it is the
    name and not the format. The default stays as it is because this library
    writes what the standards say and a receiver-facing offer is not wrong for
    being unnamed; a caller driving a sender knows what its session is called
    and is the one that should say so.

    Every caller-supplied value is validated before it is written, and an
    address is written in the form the address parse made of it rather than
    the form it arrived in. An SDP is line-structured, so a value carrying a
    line terminator would otherwise declare records of its own, and a
    malformed address describes a flow that cannot be joined. Raises
    ``ValueError`` naming the field.

    What is not written: the ``a=ts-refclk:`` and ``a=mediaclk:`` attributes
    of ST 2110-10. They name a PTP domain and a media clock this library does
    not model, and inventing either would describe a synchronisation the
    sender has not actually got. A caller that owns the clock appends them.
    """
    return _offer(
        (flow,),
        video,
        semantics=None,
        payload_type=payload_type,
        sender_type=sender_type,
        any_source=any_source,
        session_name=session_name,
        session_id=session_id,
        ttl=ttl,
    )


def format_dup_sdp(
    first: SdpFlow,
    second: SdpFlow,
    video: SdpVideo,
    *,
    payload_type: int = _DEFAULT_PAYLOAD_TYPE,
    sender_type: str | None = None,
    any_source: bool = False,
    session_name: str = " ",
    session_id: int = 0,
    ttl: int = 64,
) -> str:
    """Write the one offer that describes both legs of an ST 2022-7 pair.

    :func:`format_sdp`'s document with RFC 7104's grouping over it: a
    session-level ``a=group:DUP`` naming an ``a=mid:`` tag per leg, and a
    ``m=video`` block per leg carrying its own ``c=`` and ``a=source-filter``.
    Both blocks carry the same format, two paths being one essence, and the
    ``o=`` line names the first leg's sender. Every keyword means what it does
    for a single-leg offer and applies to both legs.

    The legs are written in the order they are passed, and :func:`parse_dup_sdp`
    reads them back in that order (§spec:redundancy).

    Two legs that name one destination, port and sender are one path written
    twice, and are refused: an offer claiming a protection the sender has not
    got is the same silent failure the parse refuses in the other direction.
    Legs sharing a group but not a sender are a pair — source-specific
    multicast is two paths onto one address.
    """
    if first == second:
        raise ValueError(
            f"both legs name {first.destination_ip} port "
            f"{first.destination_port} and the same sender, which is one "
            f"leg written twice rather than the two paths ST 2022-7 protects "
            f"with"
        )
    return _offer(
        (first, second),
        video,
        semantics=_DUP,
        payload_type=payload_type,
        sender_type=sender_type,
        any_source=any_source,
        session_name=session_name,
        session_id=session_id,
        ttl=ttl,
    )


def format_group_sdp(
    flows: Sequence[SdpFlow],
    video: SdpVideo,
    *,
    division: str = MULTI_2SI,
    payload_type: int = _DEFAULT_PAYLOAD_TYPE,
    sender_type: str | None = None,
    any_source: bool = False,
    session_name: str = " ",
    session_id: int = 0,
    ttl: int = 64,
) -> str:
    """Writes the one offer that describes an RP 2110-23 group.

    :func:`format_sdp`'s document with RFC 5888's grouping over it: a
    session-level ``a=group:`` naming the division and the tags 1 to N, and a
    ``m=video`` block per constituent carrying its own ``c=`` and
    ``a=source-filter`` and ending in its ``a=mid:``, as RP 2110-23 section
    5.3 lays the document out. ``flows`` are the constituents in constituent
    order, four or sixteen of them, and :func:`parse_group_sdp` reads them
    back in that order. Every keyword means what it does for a single-flow
    offer and applies to every constituent.

    ``video`` is each constituent's format, so it describes a sub-image: a
    four-way group of a 7680x4320 essence passes 3840x2160. Every block
    carries it, which writes the one ``TROFF`` RP 2110-23 section 5.5
    requires of every constituent; the equal RTP timestamps it also requires
    follow from stamping each constituent's headers from one format and one
    frame index. ``division`` is ``MULTI-2SI`` unless the caller names
    ``MULTI-SD``, which the RP deprecates for new designs.

    Raises ``ValueError`` for a count other than 4 or 16, for ``PHASED``, for
    an interlaced format, for a sub-image the division cannot have cut from
    an essence, for two constituents on one multicast address, which section
    5.6 forbids, and for two that name one address, port and sender.
    """
    if division == _PHASED:
        raise ValueError(
            "PHASED divides an essence in time, each phase carrying its own RTP "
            f"timestamps, and only {MULTI_2SI} and {MULTI_SD} are written here"
        )
    count = len(flows)
    if count not in CONSTITUENT_COUNTS:
        raise ValueError(
            f"a group of {count} constituents is not one RP 2110-23 defines; "
            f"it divides an essence into 4 or 16"
        )
    if video.interlaced:
        raise ValueError(
            "an interlaced format cannot be divided: RP 2110-23 defines no "
            "field structure under division"
        )
    side = math.isqrt(count)
    sub_image_size(
        video.width * side, video.height * side, division=division, count=count
    )
    seen: set[SdpFlow] = set()
    multicast: set[str] = set()
    for flow in flows:
        if flow in seen:
            raise ValueError(
                f"two constituents name {flow.destination_ip} port "
                f"{flow.destination_port} and the same sender, so a receiver "
                f"cannot tell their packets apart"
            )
        seen.add(flow)
        destination = _address(flow.destination_ip, "destination")
        if destination.is_multicast and str(destination) in multicast:
            raise ValueError(
                f"two constituents share the multicast address {destination}; "
                f"RP 2110-23 section 5.6 requires a distinct one for each"
            )
        multicast.add(str(destination))
    return _offer(
        flows,
        video,
        semantics=division,
        payload_type=payload_type,
        sender_type=sender_type,
        any_source=any_source,
        session_name=session_name,
        session_id=session_id,
        ttl=ttl,
    )


def _offer(
    flows: Sequence[SdpFlow],
    video: SdpVideo,
    *,
    semantics: str | None,
    payload_type: int,
    sender_type: str | None,
    any_source: bool,
    session_name: str,
    session_id: int,
    ttl: int,
) -> str:
    """The offer one flow, a pair or a group shares, validated before a line
    is written.

    One writer for all three, because a single-leg offer, one leg of a pair
    and one constituent of a group are the same media section under the same
    rules: two copies of them are two chances to disagree. ``semantics`` is
    the ``a=group:`` grouping the flows, or ``None`` for a single flow.
    """
    destinations = [_address(flow.destination_ip, "destination") for flow in flows]
    origins = [
        _address(flow.source_ip, "source") if flow.source_ip else None for flow in flows
    ]
    for flow in flows:
        _integer(flow.destination_port, "port", 1, _MAX_PORT)
    validate_payload_type(payload_type)
    _integer(ttl, "TTL", 0, _MAX_TTL)
    _integer(session_id, "session id", 0, _MAX_SESSION_ID)
    if _LINE_TERMINATORS.intersection(session_name):
        raise ValueError("a session name cannot carry a line terminator")
    for destination, origin in zip(destinations, origins, strict=True):
        if origin is not None and origin.version != destination.version:
            raise ValueError(
                f"the sender {origin} and the destination {destination} are in "
                f"different address families, which no flow is: RFC 4570 permits "
                f"the '*' address type only where the destination is an FQDN, so "
                f"a filter over both has no spelling"
            )
        if origin is not None and any_source:
            raise ValueError(
                f"the flow names {origin} as its sender and any_source offers the "
                f"group to every sender; the offer cannot say both"
            )
        if origin is None and destination.is_multicast and not any_source:
            raise ValueError(
                f"a multicast offer for {destination} names no sender: set "
                f"SdpFlow.source_ip to the sender's address, which ST 2110-10 "
                f"section 8.4 asks a sender to signal, or pass any_source=True to "
                f"offer the group to any sender"
            )
    parameters = _media_type_parameters(video, sender_type)
    origin = origins[0]
    host = str(origin) if origin else _unspecified_host(destinations[0])
    lines = [
        "v=0",
        f"o=- {session_id} {session_id} IN {_addrtype(origin or destinations[0])} "
        f"{host}",
        f"s={session_name}",
        "t=0 0",
    ]
    # RFC 5888 leaves a tag's value to the document, and RFC 7104's and
    # RP 2110-23's examples both number them; RP 2110-23 section 5.3 makes the
    # number the constituent's.
    tags = tuple(str(n) for n in range(1, len(flows) + 1)) if semantics else ("",)
    if semantics:
        lines.append(f"a=group:{semantics} {' '.join(tags)}")
    # RP 2110-23 section 5.3 ends each constituent's section with its a=mid:.
    # A pair keeps the place it has always been written in.
    mid_last = semantics in _RP_GROUPS
    for flow, destination, leg_origin, tag in zip(
        flows, destinations, origins, tags, strict=True
    ):
        # RFC 4566 section 5.7: an IPv4 multicast address "MUST also have a
        # time to live (TTL) value present", and for IPv6 it "MUST NOT be
        # present".
        scope = (
            f"/{ttl}" if destination.version == 4 and destination.is_multicast else ""
        )
        lines.append(f"m=video {flow.destination_port} RTP/AVP {payload_type}")
        lines.append(f"c=IN {_addrtype(destination)} {destination}{scope}")
        if leg_origin is not None:
            lines.append(
                f"a=source-filter: incl IN {_addrtype(destination)} "
                f"{destination} {leg_origin}"
            )
        if tag and not mid_last:
            lines.append(f"a=mid:{tag}")
        lines.append(f"a=rtpmap:{payload_type} raw/{RTP_CLOCK_RATE}")
        lines.append(f"a=fmtp:{payload_type} {parameters}")
        if tag and mid_last:
            lines.append(f"a=mid:{tag}")
    return "".join(f"{line}\r\n" for line in lines)


def _media_type_parameters(video: SdpVideo, sender_type: str | None) -> str:
    """The ``a=fmtp:`` parameters for a format, in ST 2110-20 section 7 order.

    Section 7.1 fixes the punctuation: entries "separated by the semicolon
    (';') character followed by whitespace", with "no semicolon character
    after the last item". It fixes no order, so the order here is the two
    standards' own: the eight ST 2110-20 section 7.2 requires, then the ``TP``
    ST 2110-21 section 8.1 calls "additional", then section 7.3's.

    The parameters of section 7.3 are written only where they differ from
    their defaults, because that is what their absence is defined to mean —
    writing ``MAXUDP=1460`` claims a limit was negotiated when it was not.
    """
    _token(video.sampling, "sampling")
    _token(video.colorimetry, "colorimetry")
    _integer(video.width, "width", 1, _MAX_RASTER)
    _integer(video.height, "height", 1, _MAX_RASTER)
    _integer(video.depth, "depth", _DEPTHS[0], _DEPTHS[-1])
    if video.depth not in _DEPTHS:
        raise ValueError(
            f"a depth of {video.depth} is not one of the {list(_DEPTHS)} bits "
            f"ST 2110-20 section 7.4.2 lists"
        )
    _integer(video.max_udp, "MAXUDP", 1, _MAX_UDP_SIZE)
    validate_frame_rate(video.frame_rate)
    if sender_type is None:
        sender_type = video.sender_type
    validate_sender_type(sender_type)
    if video.tr_offset_us is not None:
        _tr_offset(video.tr_offset_us, video.frame_rate)
    if video.cmax is not None:
        _integer(video.cmax, "CMAX", 1, _MAX_CMAX)

    parameters = [
        f"sampling={video.sampling}",
        f"width={video.width}",
        f"height={video.height}",
        f"exactframerate={_exact_frame_rate(video.frame_rate)}",
        f"depth={video.depth}",
        f"colorimetry={video.colorimetry}",
        f"PM={_PACKING_MODE}",
        f"SSN={_standard_number(video.colorimetry)}",
        f"TP={sender_type}",
    ]
    # ST 2110-21 section 8.2's optional parameters, beside the section 8.1 one
    # that shares their document, and each only where it differs from the
    # default its absence means.
    if video.tr_offset_us is not None:
        parameters.append(f"TROFF={video.tr_offset_us}")
    if video.cmax is not None:
        parameters.append(f"CMAX={video.cmax}")
    if video.interlaced:
        parameters.append("interlace")
    if video.max_udp != STANDARD_UDP_SIZE_LIMIT:
        parameters.append(f"MAXUDP={video.max_udp}")
    return "; ".join(parameters)


def _standard_number(colorimetry: str) -> str:
    """Which revision of ST 2110-20 a sender signals (section 7.2)."""
    return _STANDARD_NUMBER_2022 if colorimetry == _ALPHA else _STANDARD_NUMBER


def _exact_frame_rate(rate: Fraction) -> str:
    """``25``, or ``30000/1001`` — ST 2110-20 section 7.2's two spellings.

    A ``Fraction`` is already in lowest terms, which is the standard's
    "numerically smallest numerator value possible".
    """
    if rate.denominator == 1:
        return str(rate.numerator)
    return f"{rate.numerator}/{rate.denominator}"


def validate_sender_type(value: str) -> None:
    """Refuse a ``TP`` that is not one ST 2110-21 section 7.1 defines.

    A TP is a claim about pacing a receiver provisions its buffer from, and
    the same claim on the way in and on the way out (§spec:sdp).

    Package-internal, and shared with :mod:`pyst2110.timing`, which resolves
    the same token against a caller's override: two copies of one bound are
    two chances to disagree about it.
    """
    if value not in SENDER_TYPES:
        raise ValueError(
            f"a TP of {value!r} is not one of the {list(SENDER_TYPES)} "
            f"sender types ST 2110-21 section 7.1 defines"
        )


def _tr_offset(value: object, frame_rate: Fraction) -> None:
    """Refuse a ``TROFF`` that is not a whole number of microseconds in a frame.

    ST 2110-21 section 8.2 makes the value "a positive integer number of
    microseconds", and section 6.2 makes TR_OFFSET the difference between the
    most recent frame boundary and T_VD — so it lies inside one frame period,
    which is the bound.
    """
    period_us = _MICROSECONDS_PER_SECOND / frame_rate
    last_inside = math.ceil(period_us) - 1
    _integer(value, "TROFF", 0, last_inside)


def _token(value: str, name: str) -> None:
    """Refuse a media type parameter value that is not a single token.

    ST 2110-20 section 7.1 allows "no whitespace within the name or value", so
    this is the standard's own rule — and it is also what stops a value from
    forging a parameter, or a line, of its own.
    """
    if not value or any(character.isspace() for character in value):
        raise ValueError(f"{name}={value!r} is not a single ST 2110-20 token")


def validate_payload_type(value: int) -> None:
    """Refuse an RTP payload type outside its seven bits (RFC 3550 §5.1).

    Package-internal, and shared with :mod:`pyst2110.transmit`: the offer and
    the headers describe one flow, and two copies of one bound are two
    chances to disagree about it.
    """
    _integer(value, "payload type", 0, _MAX_PAYLOAD_TYPE)


def _integer(value: object, name: str, low: int, high: int) -> None:
    """Refuse a value that is not a whole number inside the range named.

    :class:`SdpFlow` and :class:`SdpVideo` are plain dataclasses, so an
    annotation of ``int`` is a promise and not a check: a string reaches the
    document verbatim, and one carrying a semicolon or a line terminator
    forges a media type parameter or a record of its own.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"a {name} of {value!r} is not a whole number")
    if not low <= value <= high:
        raise ValueError(f"a {name} of {value} is outside the range {low}-{high}")


def _address(text: str, role: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Parse an address, so that what is written is one and only one.

    The parsed object is what the document carries, never the caller's own
    string. CPython accepts an IPv6 scope identifier made of any characters
    but ``%`` — carriage returns and newlines among them — so writing the
    input back would forge records with real CRLFs, past a strict RFC 4566
    reader and not merely past a line split. A scope names a local interface
    and means nothing to a peer, so it is refused rather than stripped.
    """
    try:
        address = ipaddress.ip_address(text)
    except ValueError as exc:
        raise ValueError(f"the {role} {text!r} is not an IP address") from exc
    if isinstance(address, ipaddress.IPv6Address) and address.scope_id is not None:
        raise ValueError(f"the {role} {text!r} carries a scope identifier")
    return address


def _addrtype(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str:
    """RFC 4566's ``<addrtype>``, which the address family decides."""
    return "IP4" if address.version == 4 else "IP6"


def _unspecified_host(
    destination: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> str:
    """The ``o=`` address where no source filter names a sender.

    RFC 4566 wants a unicast address there. With no sender declared there is
    none to give, so the unspecified address of the destination's own family
    says so rather than naming a host that is not the origin.
    """
    return "0.0.0.0" if destination.version == 4 else "::"  # noqa: S104
