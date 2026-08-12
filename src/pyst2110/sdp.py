"""SDP: the little of RFC 4566 that names a flow and its video format.

Enough to read the addresses out of an ST 2110 offer, and enough of
ST 2110-20's ``a=fmtp:`` to size the frame a transmitter is about to pace
(SPEC §spec:sdp). Both directions: a receiver reads an SDP it is handed, and
a transmitter is configured by one it produces.

Colorimetry is carried through, not interpreted: what a consumer does with
``BT2020`` is its own concern, and this records which token the SDP said.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from fractions import Fraction

__all__ = [
    "RTP_CLOCK_RATE",
    "STANDARD_UDP_SIZE_LIMIT",
    "SdpFlow",
    "SdpVideo",
    "format_sdp",
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

# ST 2110-20 section 7.2: width and height are "integers between 1 and 32767
# inclusive" — which is also all the SRD Row Number and Offset fields hold.
_MAX_RASTER = 32767
_MAX_PORT = 65535
# RFC 3550 section 5.1 gives the payload type seven bits.
_MAX_PAYLOAD_TYPE = 127
_DEFAULT_PAYLOAD_TYPE = 96
_MAX_TTL = 255
# A UDP datagram's own length field is sixteen bits, so no MAXUDP above this
# describes a datagram that can exist.
_MAX_UDP_SIZE = 65535
# RFC 4566 section 5.2 wants the session id "based on a 64-bit NTP timestamp".
_MAX_SESSION_ID = (1 << 64) - 1
# ST 2110-20 section 7.4.2 lists the depths a sender may declare. 16f is
# half-float and not modelled here (§road:future).
_DEPTHS = (8, 10, 12, 16)

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
    """
    destination_ip = ""
    destination_port = 0
    source_ip = ""

    # Scoped to the first video section, not last-wins over the document.
    # RFC 4566: a media section's own attributes override the session-level
    # ones "for the respective media", so a 2110 SDP carrying video then
    # audio would otherwise yield the audio port — and a flow attached to it
    # receives nothing.
    in_video = False
    seen_video = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("m="):
            if seen_video:
                break  # a later section cannot override the one we took
            in_video = line[len("m=") :].split()[:1] == ["video"]
            if in_video:
                seen_video = True
                destination_port = _media_port(line)
            continue
        if line.startswith("c=") and (in_video or not seen_video):
            # Session-level c= applies until a media section supplies its own.
            destination_ip = _connection_address(line)
        elif line.startswith("a=source-filter") and (in_video or not seen_video):
            source_ip = _source_address(line)

    if not destination_ip:
        raise ValueError("the SDP has no 'c=IN IP4 <address>' connection line")
    if not seen_video:
        raise ValueError("the SDP has no 'm=video <port> ...' media section")
    if not destination_port:
        raise ValueError("the SDP's video section declares port 0, which disables it")
    return SdpFlow(destination_ip, destination_port, source_ip)


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
    if not port.isdigit():
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

    return SdpVideo(
        width=int(parameters["width"]),
        height=int(parameters["height"]),
        frame_rate=_frame_rate(parameters["exactframerate"]),
        depth=int(parameters.get("depth", 10)),
        sampling=parameters.get("sampling", ""),
        colorimetry=parameters.get("colorimetry", _UNSPECIFIED),
        # A flag with no value: SMPTE ST 2110-20 writes bare "interlace".
        interlaced="interlace" in parameters,
        # Section 7.3: absent means the Standard UDP Size Limit is in use.
        max_udp=int(parameters.get("MAXUDP", STANDARD_UDP_SIZE_LIMIT)),
    )


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
            in_video = line[len("m=") :].split()[:1] == ["video"]
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
            return Fraction(int(numerator), int(denominator))
        return Fraction(int(numerator))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"not an exactframerate: {text}") from exc


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
    session_name: str = " ",
    session_id: int = 0,
    ttl: int = 64,
) -> str:
    """Write the SDP offer that describes this flow and format.

    The document RFC 4566 section 5 requires, in the order it requires, with
    the media type parameters of ST 2110-20 section 7.2 on the ``a=fmtp:``
    line and the 90 kHz clock of section 7.1 on the ``a=rtpmap:`` one.
    Records end with CRLF, as RFC 4566 specifies.

    ``session_id`` fills both the session identifier and its version in the
    ``o=`` line. It defaults to zero, which is repeatable rather than unique:
    RFC 4566 wants the tuple globally unique, so a caller offering several
    sessions passes its own. ``session_name`` defaults to the single space
    section 5.3 prescribes where a session has no meaningful name.

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
    destination = _address(flow.destination_ip, "destination")
    origin = _address(flow.source_ip, "source") if flow.source_ip else None
    _integer(flow.destination_port, "port", 1, _MAX_PORT)
    _integer(payload_type, "payload type", 0, _MAX_PAYLOAD_TYPE)
    _integer(ttl, "TTL", 0, _MAX_TTL)
    _integer(session_id, "session id", 0, _MAX_SESSION_ID)
    if _LINE_TERMINATORS.intersection(session_name):
        raise ValueError("a session name cannot carry a line terminator")

    # RFC 4566 section 5.7: an IPv4 multicast address "MUST also have a time
    # to live (TTL) value present", and for IPv6 it "MUST NOT be present".
    scope = f"/{ttl}" if destination.version == 4 and destination.is_multicast else ""
    host = str(origin) if origin else _unspecified_host(destination)
    lines = [
        "v=0",
        f"o=- {session_id} {session_id} IN {_addrtype(origin or destination)} {host}",
        f"s={session_name}",
        "t=0 0",
        f"m=video {flow.destination_port} RTP/AVP {payload_type}",
        f"c=IN {_addrtype(destination)} {destination}{scope}",
    ]
    if origin is not None:
        lines.append(
            f"a=source-filter: incl IN {_filter_addrtype(destination, origin)} "
            f"{destination} {origin}"
        )
    lines.append(f"a=rtpmap:{payload_type} raw/{RTP_CLOCK_RATE}")
    lines.append(f"a=fmtp:{payload_type} {_media_type_parameters(video)}")
    return "".join(f"{line}\r\n" for line in lines)


def _media_type_parameters(video: SdpVideo) -> str:
    """The ``a=fmtp:`` parameters for a format, in ST 2110-20 section 7 order.

    Section 7.1 fixes the punctuation: entries "separated by the semicolon
    (';') character followed by whitespace", with "no semicolon character
    after the last item".

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
    if video.frame_rate <= 0:
        raise ValueError(f"an exactframerate of {video.frame_rate} is not a rate")

    parameters = [
        f"sampling={video.sampling}",
        f"width={video.width}",
        f"height={video.height}",
        f"exactframerate={_exact_frame_rate(video.frame_rate)}",
        f"depth={video.depth}",
        f"colorimetry={video.colorimetry}",
        f"PM={_PACKING_MODE}",
        f"SSN={_standard_number(video.colorimetry)}",
    ]
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


def _token(value: str, name: str) -> None:
    """Refuse a media type parameter value that is not a single token.

    ST 2110-20 section 7.1 allows "no whitespace within the name or value", so
    this is the standard's own rule — and it is also what stops a value from
    forging a parameter, or a line, of its own.
    """
    if not value or any(character.isspace() for character in value):
        raise ValueError(f"{name}={value!r} is not a single ST 2110-20 token")


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


def _filter_addrtype(
    destination: ipaddress.IPv4Address | ipaddress.IPv6Address,
    source: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> str:
    """The ``<addrtype>`` of ``a=source-filter``, which covers both addresses.

    RFC 4570 section 3 allows the wildcard where the filter spans more than
    one family, which a v4 destination filtering a v6 source does: naming
    either family would describe the other address wrongly.
    """
    if destination.version == source.version:
        return _addrtype(destination)
    return "*"


def _unspecified_host(
    destination: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> str:
    """The ``o=`` address where no source filter names a sender.

    RFC 4566 wants a unicast address there. With no sender declared there is
    none to give, so the unspecified address of the destination's own family
    says so rather than naming a host that is not the origin.
    """
    return "0.0.0.0" if destination.version == 4 else "::"  # noqa: S104
