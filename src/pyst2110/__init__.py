"""SMPTE ST 2110 protocol: RTP and RFC 4175 headers, geometry, SDP.

The layer between a transport and a raster. A NIC binding, a socket or a
capture file supplies packets; this says what their headers mean and where
their payloads belong; the consumer moves the pixels (SPEC
§spec:scope-boundary).

Nothing here needs a NIC, a licence or a vendor SDK.
"""

from pyst2110.framing import FrameTracker as FrameTracker
from pyst2110.framing import SequenceTracker as SequenceTracker
from pyst2110.framing import frame_boundaries as frame_boundaries
from pyst2110.geometry import byte_offset as byte_offset
from pyst2110.geometry import choose_payload_size as choose_payload_size
from pyst2110.geometry import fits_raster as fits_raster
from pyst2110.geometry import line_bytes as line_bytes
from pyst2110.geometry import packets_per_frame as packets_per_frame
from pyst2110.geometry import packets_per_line as packets_per_line
from pyst2110.geometry import pgroup as pgroup
from pyst2110.payload import PayloadHeaders as PayloadHeaders
from pyst2110.payload import parse_payload_headers as parse_payload_headers
from pyst2110.rtp import FIXED_HEADER_SIZE as FIXED_HEADER_SIZE
from pyst2110.rtp import RtpHeaders as RtpHeaders
from pyst2110.rtp import parse_rtp as parse_rtp
from pyst2110.sdp import RTP_CLOCK_RATE as RTP_CLOCK_RATE
from pyst2110.sdp import STANDARD_UDP_SIZE_LIMIT as STANDARD_UDP_SIZE_LIMIT
from pyst2110.sdp import SdpFlow as SdpFlow
from pyst2110.sdp import SdpVideo as SdpVideo
from pyst2110.sdp import format_sdp as format_sdp
from pyst2110.sdp import parse_sdp as parse_sdp
from pyst2110.sdp import parse_video_format as parse_video_format

__all__ = [
    "FIXED_HEADER_SIZE",
    "RTP_CLOCK_RATE",
    "STANDARD_UDP_SIZE_LIMIT",
    "FrameTracker",
    "PayloadHeaders",
    "RtpHeaders",
    "SdpFlow",
    "SdpVideo",
    "SequenceTracker",
    "byte_offset",
    "choose_payload_size",
    "fits_raster",
    "format_sdp",
    "frame_boundaries",
    "line_bytes",
    "packets_per_frame",
    "packets_per_line",
    "parse_payload_headers",
    "parse_rtp",
    "parse_sdp",
    "parse_video_format",
    "pgroup",
]
