"""SDP: the little of RFC 4566 that names a flow and its video format.

Enough to read the addresses out of an ST 2110 offer, and enough of
ST 2110-20's ``a=fmtp:`` to size the frame a transmitter is about to pace
(SPEC §spec:sdp).

Colorimetry is carried through, not interpreted: what a consumer does with
``BT.2020`` is its own concern, and this records which token the SDP said.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

__all__ = ["SdpFlow", "SdpVideo", "parse_sdp", "parse_video_format"]


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
    interlaced: bool = False
    # The largest UDP payload the sender will emit. ST 2110-20 carries it as
    # an optional fmtp parameter and the SDK documents 1460 as its default,
    # which is what a 1500-byte MTU leaves after the IP and UDP headers.
    max_udp: int = 1460

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
        # A flag with no value: SMPTE ST 2110-20 writes bare "interlace".
        interlaced="interlace" in parameters,
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
