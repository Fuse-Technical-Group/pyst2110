"""Where the wire fields sit, declared once for the reader and the writer.

Internal. RFC 3550 section 5.1 fixes the RTP header's octets and RFC 4175
section 4.2 the payload header's, and a parse and a builder that each restate
them agree with each other whatever either one says: a round trip through both
passes a matched pair of wrong offsets, and only a hand-written vector catches
it (§spec:testing). So the octets are stated here and imported.

Offsets are relative to the unit that owns them — the RTP fields to the start
of the fixed header, the SRD fields to the start of their own six-octet
header — so a module reading a packet-relative position adds its own base.
"""

from __future__ import annotations

#: Octets of RFC 3550 fixed header, before any CSRC list or extension.
FIXED_HEADER_SIZE = 12

# Within the fixed header.
SEQUENCE = 2
TIMESTAMP = 4
SSRC = 8

# Byte 0: V(2) P(1) X(1) CC(4). Byte 1: M(1) PT(7).
VERSION_SHIFT = 6
PADDING_MASK = 0x20
EXTENSION_MASK = 0x10
CSRC_COUNT_MASK = 0x0F
MARKER_MASK = 0x80
PAYLOAD_TYPE_MASK = 0x7F

#: The version every packet under this standard carries, in place: ST 2110-20
#: section 6.1.2 provides for an extension but requires none, so P, X and CC
#: are clear beside it.
RTP_VERSION = 2
VERSION_2 = RTP_VERSION << VERSION_SHIFT

#: RFC 3551 section 6 leaves 96-127 dynamic, which is where a raw video format
#: is bound; the first of them is what an offer and a header both default to.
DYNAMIC_PAYLOAD_TYPE = 96

# RFC 4175 section 4.2: two octets of Extended Sequence Number, then one or
# more six-octet Sample Row Data headers.
EXTENDED_SEQUENCE_SIZE = 2
SRD_SIZE = 6
# Within one SRD header.
SRD_LENGTH = 0
SRD_ROW = 2
SRD_OFFSET = 4

# The F bit tops the row word and the C bit tops the offset word, leaving
# fifteen bits of value under each. The SRD Length has no flag above it.
FLAG_MASK = 0x8000
VALUE_MASK = 0x7FFF

#: One past the largest a field of this width holds — the modulus a value
#: written into one wraps in, and the bound a caller's own is checked against.
U16_MODULUS = 1 << 16
U32_MODULUS = 1 << 32
