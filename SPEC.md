# pyst2110 — Specification

Derived from [REQUIREMENTS.md](REQUIREMENTS.md).

## Problem statement §spec:problem-statement

*Status: complete*

SMPTE ST 2110-20 carries uncompressed video over RTP. Reading a flow means
parsing an RTP header (RFC 3550) and an RFC 4175 payload header per packet,
tracking sequence numbers to see loss, and finding frame boundaries from the
marker bit. Sending one means building those headers and pacing them.

None of that is vendor work, and none of it is pixel work. pyst2110 is the
layer between: it reads and writes ST 2110 headers over numpy arrays, and
computes the geometry that says where a packet's payload belongs in a raster.

### Scope boundary §spec:scope-boundary

pyst2110 owns the ST 2110 protocol: RTP and RFC 4175 header layouts, frame
boundaries, sequence-loss accounting, pgroup geometry, and the little of SDP
that describes a flow and its video format.

It does not own the transport. Acquiring packets is a NIC binding's job —
a vendor binding, a socket, a capture file — and this library depends on
none of them. It does not own pixels either: converting samples
to and from a raster is a per-pixel-format kernel that belongs next to a
consumer's other kernels, where it can run on a GPU.

The boundary is therefore **descriptors**. Given a chunk's header bytes,
this library says which frame each packet belongs to and where in the raster
its payload lands. The consumer performs the move.

*Why a separate library rather than either neighbour*: a transport binding
that carried RFC 4175 would make a vendor SDK the dependency for parsing a
capture file, and where that SDK is licensed — as Rivermax is — it would
trap open protocol work behind a licence with no claim on it, testable only
where a NIC and a licence are. A consumer that carried it would take on a
transport's vocabulary to reach its own pixels, and the next consumer would
write it again.

### Prior art §spec:prior-art

The reference implementations are the SMPTE and IETF documents themselves.
GStreamer's `rtpvrawdepay`/`rtpvrawpay` and FFmpeg's `rtpdec_rfc4175` are
the closest working code; both are per-packet C inside a media framework
rather than a vectorized library, and neither is importable from Python.

An earlier version of this material lived in a transport binding's
examples, written there because every example needed it. It is the seed
for these modules and the source of the initial tests.

## Interface shape §spec:interface-shape

*Status: complete*

Every parse takes a two-dimensional `uint8` array — one row per packet,
stride columns wide — and returns one array per field, packet-aligned. That
is the shape a header-data-split receiver already hands out, and slicing a
contiguous capture into rows costs nothing.

Payload descriptors are the one exception, and are segment-aligned: the
continuation flag makes a packet's segment count data-dependent
(§spec:payload-header), so no packet-aligned array can hold them. They carry
the row index they came from instead.

*Why arrays rather than a packet object*: a per-packet Python object at
250,000 packets a second is not affordable (§req:priorities), and an object
per packet is the one interface that cannot be made fast later without
changing every caller.

Functions are free where they read, and classes only where state spans
calls — a sequence tracker carries the last number it saw, a frame
assembler carries the packets of a frame in progress. Nothing else holds
state.

Rows beyond a packet's actual length are ignored rather than trusted: a
receiver writes at most the stride and reports the true size separately, so
every parse takes the sizes when a field's position depends on them.

## RTP §spec:rtp

*Status: complete*

The fixed twelve-byte header of RFC 3550 section 5.1, parsed vectorized
across a chunk. SMPTE ST 2110-20 section 6.1.2 constrains what its fields
carry rather than where they sit.

The **marker bit** is the frame boundary. ST 2110-20 section 6.1.2 sets it
on the last packet of a frame, and on the last packet of each field when
interlaced, so a rising edge in the marker column ends one and the next
packet begins another. An interlaced flow marks twice per frame, so what a
receiver counts there is fields. The rising edge rather than the bit: a
duplicated final packet, or the same packet arriving on both legs of a
redundant pair, would otherwise end a frame that had already ended.

The **sequence number** is sixteen bits and wraps. Loss is counted by
differencing successive numbers in the thirty-two-bit space the extended
sequence number completes (§spec:payload-header), where a gap and a wrap
stay distinguishable for hours at ST 2110-20 rates.

Where a caller has only the RTP field, the same differencing runs modulo
2^16. That distinguishes a wrap from a gap for any gap short of RFC 3550's
MAX_DROPOUT — the field wraps every fifth of a second at these rates, so
beyond that the two are indistinguishable in the protocol, not merely in
this implementation. Such a jump is reported as a resync and left out of the
loss count, since a number nobody can act on is worse than an admission of
ignorance.

The header is read as one twelve-octet unit. A packet reporting a size
under that carries none of it, so every field reads zero rather than the
bytes a reused ring still holds there — otherwise a short datagram from
another sender in the same group contributes a sequence number and an SSRC
to a receiver's metrics out of the previous frame. A version of zero says
the packet was too short, no sender emitting one.

An **extension** shifts the payload header, so its bit is read rather than
assumed clear. ST 2110-20 section 6.1.2 provides for one — where the X bit
is set, an RFC 8285 header extension follows the SSRC — and a receiver that
ignored it would read the extension's first octets as a payload header and
report a line that does not exist.

## RFC 4175 payload header §spec:payload-header

*Status: complete*

After the RTP header: an extended sequence number, then one or more line
segments, laid out in RFC 4175 section 4.2 and constrained by ST 2110-20
section 6.1.4.

The continuation flag on a segment says another follows, so the count is
data-dependent. The parse walks a bounded number of segments and stays
vectorized across every packet at each one, because a Python loop over a
quarter of a million packets a second is not affordable (§req:priorities).
ST 2110-20 permits three and RFC 4175 sets no limit, so the bound is the
caller's with three as the default. A packet still declaring a continuation
at the bound is flagged rather than trimmed in silence — dropping part of a
raster without saying so is the failure worth naming.

The flag is per packet, because the instruction it carries is to discard
that packet. A chunk-wide count says one packet in four thousand is poison
and gives no way to find it, which leaves only two answers and both are
wrong: gather everything, or drop a whole frame over one crafted packet.
Such a packet contributes no descriptors either. Its unparsed SRD headers
put every source it did parse six octets early, so what survives is wrong
rather than merely incomplete, and a descriptor nobody should act on is
better not emitted.

The extended sequence number is the high sixteen bits of a thirty-two-bit
sequence, which is what makes loss detection survive a flow fast enough to
wrap the RTP field inside a frame.

A segment resolves to a **descriptor**: which line, which byte offset
within it, and how many bytes. That is what a consumer's gather kernel
needs and the whole of what this library produces about a payload. The
header's own offset is a sample position rather than a byte count, so
§spec:geometry is what turns one into the other.

Segment lengths are reported as declared and not checked against the packet.
Under a header-data split the payload sits in a buffer this parse never
sees, so bounding a gather belongs to the consumer, against the buffer it
actually reads. A segment's source offset within its packet follows from
those lengths, so it inherits the same caveat: it can point past the
packet, and a consumer sizing a read as `size - source` gets a negative
that an unsigned length turns enormous.

A descriptor's row and offset are equally unchecked, for a different
reason: they are fifteen-bit fields naming a place in a raster this parse
has no format for, so a packet may name row 32767 of a 1080-row image.
Bounding them is §spec:geometry's, and belongs before the placement
arithmetic — the consumer's gather is a kernel over device memory
(§spec:scope-boundary), where a row past the image writes outside the
raster rather than raising.

## Geometry §spec:geometry

*Status: complete*

ST 2110-20 packs samples into pgroups — the smallest whole number of pixels
that fills a whole number of bytes, which depends on sampling and depth
(YCbCr-4:2:2 10-bit is 2 pixels in 5 bytes). A payload header's offset is a
sample position, so converting one to a byte offset needs the format.

From the format come line length, packets per line at a given payload size,
packets per frame, and the largest payload size that divides a line without
a remainder. A transmitter needs all four to size what it sends; a receiver
needs line length to place what it gets.

That payload size is searched over pgroups per packet rather than over
bytes: both RFC 4175 and ST 2110-20 require a segment length be a multiple
of the pgroup, and a divisor of the line length need not be one. One pgroup
per packet always tiles, so any limit at or above a pgroup has an answer.

Placing a descriptor is two steps, and the check is the first of them.
Scaling a sample position by the pgroup cannot tell a position inside the
image from one past it — both scale — so a mask says which descriptors name
a place inside the raster, row inside the image and sample position inside
the row, and the conversion is documented as requiring it. An interlaced
flow numbers rows within a field, so the row bound there is half the
frame's height (§spec:payload-header).

4:2:0 sampling is refused rather than approximated. Its pgroups span two
sample rows, so a line is not a whole number of them and none of the
arithmetic above holds.

## SDP §spec:sdp

*Status: in progress*

Enough of RFC 4566 and ST 2110-20's `a=fmtp:` to name a flow — connection
address, media port, source filter — and its video format: width, height,
frame rate, sampling, depth, interlace, and the sender's maximum UDP
payload.

Parsing and emitting are both here, because a receiver reads an SDP it is
handed and a transmitter is configured by one it produces. The video format
it yields is what §spec:geometry computes from.

Colorimetry is carried through, not interpreted. What a consumer does with
`BT.2020` is its own concern; this library records which token the SDP said.

## Transmit headers §spec:transmit-headers

*Status: not started*

Building the header block for a frame: RTP headers with the payload type,
SSRC and marker bit set on the last packet, and RFC 4175 payload headers
whose line numbers and offsets walk the raster. Built once for a frame
shape and stamped per frame with its sequence numbers and media timestamp,
because the layout repeats and only two fields move.

The **media timestamp** is the RTP clock — 90 kHz for video — sampled at
the frame's own rate, which is what a receiver locks to. Deriving it from a
frame index rather than a wall clock keeps a transmitter's timestamps exact
across arbitrary run lengths.

## Testing §spec:testing

*Status: in progress*

Everything here runs on a bare runner. There is no hardware to mark around
and no SDK to build against, so the default selection is the whole suite —
which is the point of the split (§spec:scope-boundary).

Header parsing is tested against **bytes written by hand** from the
standards' own field diagrams, not against this library's own writer.
Round-tripping the writer through the reader proves they agree with each
other and nothing about either agreeing with the wire; both are tested
against fixed vectors, and the round trip is an additional property.

Captures from real senders are the strongest evidence available without a
NIC. Where a fixture is taken off the wire its provenance is recorded
beside it — which sender, which format — because a capture whose origin is
unknown proves nothing about conformance.

## Packaging §spec:packaging

*Status: in progress*

A pure-Python package with numpy as its only runtime dependency, published
under MIT. No build step and no compiled extension: the parses are numpy
expressions, and a C extension would trade the portability that makes this
library usable from a capture-analysis script for a speed nothing has yet
asked for.

*Why publishable at all*: the licence and the absence of vendor content are
what let this material be public (§spec:scope-boundary), and a library that
cannot be depended on by path alone is what makes the three-repository
split real rather than nominal.
