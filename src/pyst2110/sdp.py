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

Colorimetry is carried through, not interpreted: what a consumer does with
``BT2020`` is its own concern, and this records which token the SDP said.
"""

from __future__ import annotations

import ipaddress
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction

from pyst2110 import _layout

__all__ = [
    "RTP_CLOCK_RATE",
    "SENDER_TYPES",
    "SENDER_TYPE_NARROW",
    "SENDER_TYPE_NARROW_LINEAR",
    "SENDER_TYPE_WIDE",
    "STANDARD_UDP_SIZE_LIMIT",
    "SdpFlow",
    "SdpVideo",
    "format_dup_sdp",
    "format_sdp",
    "parse_dup_sdp",
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

# The identification tags a written offer gives its two blocks. RFC 5888
# leaves the value a token of the document's own choosing, and RFC 5888's own
# examples number them; nothing outside the document reads them.
_DUP_TAGS = ("1", "2")

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

    sections = _video_sections(session, media)
    if len(tags) != len(sections):
        raise ValueError(
            f"the SDP's 'a=group:DUP' names {len(tags)} tags over "
            f"{len(sections)} 'm=video' blocks, and RFC 7104 groups one block "
            f"per tag: a document that says two things about how many legs it "
            f"has is one a sender puts one leg of on the wire"
        )
    if len(tags) != _DUP_LEGS:
        raise ValueError(
            f"the SDP's 'a=group:DUP' names {len(tags)} legs; ST 2022-7 as "
            f"this library models it is two, which is also all Rivermax's "
            f"RMX_MAX_DUP_STREAMS carries"
        )

    if len(set(tags)) != len(tags):
        raise ValueError(
            f"the SDP's 'a=group:DUP' names {' '.join(tags)}, which is one "
            f"block twice and so one path rather than two"
        )
    blocks: dict[str, _MediaSection] = {}
    for section in sections:
        if section.mid in blocks:
            raise ValueError(
                f"two of the SDP's 'm=video' blocks carry 'a=mid:"
                f"{section.mid}', so a tag naming that block names both"
            )
        blocks[section.mid] = section

    legs = []
    for tag in tags:
        if tag not in blocks:
            raise ValueError(
                f"the SDP's 'a=group:DUP' names {tag!r}, which no 'm=video' "
                f"block carries an 'a=mid:' for"
            )
        legs.append(_flow(blocks[tag], tag))
    first, second = legs
    return first, second


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
    """
    parameters = _format_parameters(text)
    if not parameters:
        raise ValueError("the SDP has no 'a=fmtp:<payload> ...' format line")

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

    return SdpVideo(
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


def _format_parameters(text: str) -> dict[str, str]:
    """``a=fmtp:96 width=1920; height=1080; interlace`` as a mapping.

    Bare flags map to the empty string, which is what makes ``interlace``
    detectable by presence.

    Scoped to the video media section for the same reason ``parse_sdp`` is:
    a multi-essence SDP carries an ``a=fmtp:`` per essence, and reading the
    audio one would describe a frame geometry that does not exist.
    """
    in_video = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("m="):
            if in_video:
                break  # past the video section; a later fmtp is another media
            in_video = _is_video(line)
            continue
        if not in_video or not line.startswith("a=fmtp:"):
            continue
        _, _, rest = line.partition(" ")
        parameters: dict[str, str] = {}
        for field in rest.split(";"):
            name, _, value = field.strip().partition("=")
            if name:
                parameters[name] = value.strip()
        return parameters
    return {}


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
    return _offer(
        (first, second),
        video,
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
    payload_type: int,
    sender_type: str | None,
    any_source: bool,
    session_name: str,
    session_id: int,
    ttl: int,
) -> str:
    """The offer one or two flows share, validated before a line is written.

    One writer for both, because a single-leg offer and one leg of a pair are
    the same media section under the same rules: two copies of them are two
    chances to disagree.
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
    if len(set(flows)) != len(flows):
        raise ValueError(
            f"both legs name {flows[0].destination_ip} port "
            f"{flows[0].destination_port} and the same sender, which is one "
            f"leg written twice rather than the two paths ST 2022-7 protects "
            f"with"
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
    redundant = len(flows) > 1
    tags = _DUP_TAGS if redundant else ("",)
    if redundant:
        lines.append(f"a=group:{_DUP} {' '.join(tags)}")
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
        if tag:
            lines.append(f"a=mid:{tag}")
        lines.append(f"a=rtpmap:{payload_type} raw/{RTP_CLOCK_RATE}")
        lines.append(f"a=fmtp:{payload_type} {parameters}")
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
