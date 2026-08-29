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
capture file, and where that SDK is licensed — as the vendor ones are — it
would trap open protocol work behind a licence with no claim on it, testable
only where a NIC and a licence are. A consumer that carried it would take on a
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

Two kinds of array are aligned to something else, each because a packet is
not what the value describes.

Payload descriptors are **segment-aligned**: the continuation flag makes a
packet's segment count data-dependent (§spec:payload-header), so no
packet-aligned array can hold them. They carry the row index they came from
instead.

`Vrx.datum_delta_ns` and `Measurement.datum_delta_ns` are **frame-aligned**:
a frame has one T_VD and one phase error against it (§spec:timing), so the
value is per frame and not per packet. Their length is the frame count, and
`frame_starts` is what maps one to the other.

An array that is neither packet-aligned nor named here does not exist. A
function returning one of these says so in its docstring, beside the shape.

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

The placement itself is here too, and is one function rather than an
expression each caller repeats: a row and a sample position resolved into an
octet offset in a frame buffer. It is the library's central output, computed
identically on both paths, and the scaling runs in both directions — a
receiver turns a sample position into octets, a sender turns the octets a
packet carries into the position that names them.

That payload size is searched over pgroups per packet rather than over
bytes: both RFC 4175 and ST 2110-20 require a segment length be a multiple
of the pgroup, and a divisor of the line length need not be one. One pgroup
per packet always tiles, so any limit at or above a pgroup has an answer.

Placing a descriptor is two steps, and the check is the first of them.
Scaling a sample position by the pgroup cannot tell a position inside the
image from one past it — both scale — so a mask says which descriptors name
a place inside the raster, row inside the image and sample position inside
the row, and the conversion is documented as requiring it. An interlaced
flow numbers rows within a field, so the row bound there is the first
field's row count (§spec:payload-header) — which an odd height makes one
more than half, section 6.1.5 giving the temporally first field the extra
line. One function computes it, and the receive bound and the transmit
split both call it rather than each halving for themselves.

4:2:0 sampling is refused rather than approximated. Its pgroups span two
sample rows, so a line is not a whole number of them and none of the
arithmetic above holds.

## SDP §spec:sdp

*Status: complete*

Enough of RFC 4566 and ST 2110-20's `a=fmtp:` to name a flow — connection
address, media port, source filter — and its video format: width, height,
frame rate, sampling, depth, colorimetry, interlace, and the sender's
maximum UDP size.

A sender's required parameters are not one document's. ST 2110-20 section
7.2 requires eight; ST 2110-21 section 8.1 adds `TP` for every video RTP
stream. A conformance review of ST 2110-20 section 7 does not meet the
ninth, which is how the offer came to go out without it — and why the test
asserts the union the two standards require rather than the set a reviewer
read.

Parsing and emitting are both here, because a receiver reads an SDP it is
handed and a transmitter is configured by one it produces. The video format
it yields is what §spec:geometry computes from.

Colorimetry is carried through, not interpreted. What a consumer does with
`BT2020` is its own concern; this library records which token the SDP said,
and ST 2110-20 has a token for a colorimetry nobody stated.

A parameter whose absence carries meaning is written only where it differs
from that meaning. An absent `MAXUDP` *is* the standard limit, so writing
the default would claim a limit was negotiated when none was.

`TP` describes the pacing of whatever puts the packets on the wire, and
that is not this library — nothing here paces a packet
(§spec:scope-boundary). So it is a caller's parameter, and the caller that
owns the pacer owns the value. The default is Narrow, on three grounds.
It is the safer of the two narrow claims: section 7.1.2's network
compatibility model divides by `43200 × R_ACTIVE × T_FRAME` where Narrow
Linear's divides by `43200 × T_FRAME`, so with `R_ACTIVE` below one Narrow
permits the larger `C_MAX` — and a sender that declares more burst than it
produces is carried by a network provisioned for it, where one that
declares less is the one whose packets a switch drops. Section 7.1.2's own
note calls a sender that packs full standard-sized packets from a locked
and phased signal and sends them as they fill compliant to the type, which
is what a hardware pacer fed by §spec:transmit-headers is. And a Narrow
receiver *should* accept a Narrow Linear sender (section 7.2.3), so of the
two it is Narrow Linear that a receiver treats as the extra. Wide is a
value and not the default: it exists for senders whose pacing software
decides, and claiming ninety times the receiver buffer a hardware pacer
needs misdescribes the sender as surely as claiming too little.

ST 2110-21 section 8.2's optional parameters ride with the format: `TROFF`
in whole microseconds, bounded inside one frame period as section 6.2
defines TR_OFFSET, and `CMAX` as the sender's claim about its own peak.
Both are read as absent-means-default and written only when set, like
`MAXUDP` above; §spec:timing is what consumes them.

A multicast offer says who may send, and says so deliberately.
ST 2110-10 section 8.4 asks a sender to signal `a=source-filter`, and
RFC 4570 makes the line's absence mean the group accepts every sender —
a real any-source-multicast session, so the line is not made
unconditional. What was wrong was that omission was the default: a caller
who forgot looked exactly like one who meant it, and the difference
surfaced as a transmit SDK's refusal rather than as an error here. A
multicast offer now names a sender or declares any-source; neither and
both raise. A unicast destination has no group to filter and needs
neither.

The ST 2110-10 clock attributes are not written, and sections 8.2 and 8.3
make them *shall* — this is a known gap in ST 2110-10 conformance, not an
optional parameter declined. They name a PTP grandmaster and a media clock
this library has no model of, and a synchronisation claimed but not held
is worse than one left to the caller that owns it (§road:future).

Every value a caller supplies is validated before it is written, an SDP
being line-structured. The bound is not CRLF but every character a line
split starts a record at: RFC 4566 ends a record with two of them and a
reader splitting the document begins one at ten, and the reader is what a
forged record has to fool. Addresses are written as the address parse
rendered them, never as they arrived — validating one string and writing
another is the gap an injection goes through — and an IPv6 scope
identifier, which accepts nearly any character and names an interface no
peer shares, is refused rather than stripped. Integers are checked as
integers: a dataclass annotation is a promise, not a check.

A source filter names one address type for two addresses. RFC 4570 permits
the wildcard that would cover two only where the destination is an FQDN,
and this emitter writes IP literals — so a v4 group filtering a v6 sender
has no spelling. It also describes no flow, that sender reaching that
group with nothing, and is refused where it is built.

## Transmit headers §spec:transmit-headers

*Status: complete*

The header block for a frame: RTP headers carrying the payload type, SSRC
and the marker that ends a frame, and RFC 4175 payload headers whose line
numbers and offsets walk the raster. Built once for a frame shape and
stamped per frame, because the layout repeats and only two fields move.

The **media timestamp** is the RTP clock — 90 kHz for video — sampled at
the frame's own rate, which is what a receiver locks to. Both it and the
sequence number derive from a frame index rather than from a running
counter, so a transmitter's hundred-thousandth frame is as exact as its
first: a rate of 60000/1001 advances a half tick a frame, which
accumulating turns into drift and computing from an index does not.

One SRD header a packet. A payload size that divides a line never straddles
a row, so the continuation bit stays clear — ST 2110-20's General Packing
Mode, and what the emitted offer declares (§road:future).

The block is **reused**: stamping writes into the array it returned last
time. Allocating a frame of headers per frame is the cost the split between
building and stamping exists to avoid, so a caller holding two frames at
once is the one that copies. Every array a stamp derives on the way is
allocated beside the block for the same reason — a claim about allocation
that the stamp then made per call would be worth less than none.

Every value that reaches a header is bounded against the bits that hold it,
and bounded identically wherever it enters. A width past the SRD Offset's
fifteen sets the Line Continuation bit and announces a segment the packet
never carried — which this library's own parse then reads out of sample
data — and a payload past the SRD Length's sixteen declares less than it
sends. One parameter has no two answers, so an offer read in is checked
where an offer written out is.

A declared UDP limit bounds the datagram, not the sample data inside it.
Turning one into the other is a subtraction of the headers, and getting it
wrong overruns the limit on exactly those rasters whose lines divide in the
gap — so the conversion is named rather than left to a caller's arithmetic.

Interlace is carried rather than refused, the receive path already
modelling it: the marker ends each field, each field carries its own
timestamp, and row numbers restart within a field.

## ST 2110-21 timing §spec:timing

*Status: complete*

SMPTE ST 2110-21 constrains when a sender's packets leave: a Packet Read
Schedule (sections 6.2-6.4) says when each packet of a frame is due, and
two leaky buckets — the Network Compatibility Model's `C_INST` against
`C_MAX` (section 6.6.1) and the Virtual Receiver Buffer's level against
`VRX_FULL` (section 6.6.2) — say whether the actual instants conform.
`pyst2110.timing` computes all of it: the schedule and its read times, the
per-type limits of section 7.1, both bucket levels over a capture, and a
measurement that names the sender type the numbers satisfy.

*Why here rather than in a pacing engine*: the model is protocol
arithmetic over per-packet timestamps — arrays in, arrays out
(§spec:interface-shape) — and it names no vendor and no transport
(§spec:scope-boundary). The same definitions serve both sides: a
transmitter derives its departure grid from `read_times`, and an analyzer
judges a capture of anybody's transmitter against the identical schedule.
The formulas are cited to their sections in the module rather than
restated here; the tests hold hand-computed values for 1080p50 and
2160p59.94 against them.

Arithmetic is exact. Schedule constants are `Fraction` seconds, instants
are integer nanoseconds since the SMPTE Epoch of ST 2059-1 (the PTP epoch,
TAI), and every division floors over integers, so a 60000/1001 rate never
rounds until a result leaves as nanoseconds. Both bucket passes are
closed-form array expressions — the reflected walk of `C_INST` is a
running minimum, the buffer level a scheduled-read count — checked against
per-packet reference loops written straight from section 6.6's prose.

**The schedule is realised on the nanosecond grid.** T_PRj is generally a
fraction of a nanosecond; emission timestamps are integers. A read counts
as passed once its truncated instant has, which is the instant `read_times`
emits — otherwise a sender that departs exactly on this library's own grid
measures one packet high forever, an artifact of sub-nanosecond arithmetic
no capture resolves.

**Anchoring is the standard's by default, with the measured alternatives
named.** The drain grid of section 6.6.1 sits at multiples of T_DRAIN
since the epoch; `origin="first"` anchors it to the first packet instead,
which is what EBU LIST measures and the only choice for timestamps that
are not PTP-locked. The frame datum T_VD is the nearest `N x T_FRAME +
TR_OFFSET` by default; `datum="rtp"` recovers N from the frame's RTP
timestamp — the RTP Clock timebase section 6.6.2 evaluates on, and the
anchor that catches a sender misaligned to its own media clock by whole
frames — and `datum="first-packet"` is LIST's pacing-only measure, which
forgives any constant phase error.

The measurement's profile is judged on the declared schedule, as LIST
judges it: a Type N sender whose peaks fit only Type W's limits is
reported `2110TPW` as a claim about the numbers, not a re-evaluation
against Wide's own linear schedule (section 6.5 is what licenses the
comparison).

**The edition is the sender's, not the analyzer's.** ST 2110-21 has two
editions, and clause 6.3.3's gapped interlaced second field is the one
place they part: the 2022 edition reads it `T_FRAME/2 + T_LINE/2` past the
datum, the 2017 edition `T_FRAME/2`. The gap is `N_PACKETS/2160` read
spacings for a 1125-line raster — two whole packet intervals at 4320
packets a frame — so a sender declaring `SSN=ST2110-20:2017` measured
against the 2022 schedule reports a phase error it does not have.
`read_schedule` and `measure` take an `edition`, defaulting to 2022, and
it shall name the edition the sender was built to. Clauses 6.3.2 and 6.4
are the same text in both, so no progressive and no linear schedule moves.

One reading comes from the earlier edition. Table 1's TRO_DEFAULT cells
for the 525- and 625-line systems are clipped in the published 2022 PDF —
the expression ends at `INT((TOTAL - HEIGHT)/2) +` — so the missing
summands are taken from ST 2110-21:2017 Table 1, whose cells are fixed
ratios with no HEIGHT term at all: 20/525, 26/625, 22/1125. R_ACTIVE reads
the same way, fixed per system in 2017 and parameterized by HEIGHT in
2022. The parameterized form reproduces every 2017 value at the reference
heights and departs from it at any other — a raster clause 6.3.1 gives no
gapped schedule at all, so the two editions are not modelled separately
here.

Not modelled, deferred to §road:future: ST 2110-22 compressed streams
(their own document), PsF's `segmented` SDP parameter (an SDP declaring it
reads as interlaced, which Table 1's 1125-line row happens to cover), and
field pairing for an interlaced capture that opens mid-frame — pairing
trusts the RFC 4175 F bit where the caller supplies it and arrival order
where not.

## ST 2022-7 redundancy §spec:redundancy

*Status: complete*

ST 2022-7 sends one essence twice by two paths and lets the receiver take
whichever copy of a packet arrives first. What that asks of this library is
arithmetic over sequence numbers rather than over pixels, and the document
that describes a pair.

`pyst2110.redundancy.reconstruct` takes two legs — extended sequence numbers
and arrival times — and reports which numbers the pair delivered between
them, which each leg would have missed alone, which neither carried, and how
far apart in time the two copies of one packet arrive. It reads no payload
and holds no state, so a caller hands it whatever it has recorded.

**The span both legs cover is what can be judged, and it is not the union.**
Two recordings rarely start and stop on the same packet, so a leg whose
capture ended earlier has a tail the other lacks; read as loss, that tail
would dwarf everything real. The measurement is bounded to the sequence
range present on both, and a pair sharing no range at all is refused rather
than reported as total loss on both sides — two captures of different
windows are not a redundant pair, and saying so beats returning a number.

**Where a leg carries the same number twice, the earliest arrival is the
one kept**, because that is the copy a receiver would have taken.

**The path differential is only as good as the clocks the two legs were
timed against.** Where the legs arrive on two ports of one adapter those are
two hardware clocks, each disciplined separately, and the offset between
them lands in the measurement. A skew smaller than that offset is not a
reading of the network. The figure is reported signed and unqualified; what
it may be compared against is the caller's to know.

**A redundant pair is one offer, not two.** RFC 7104 groups the legs with
a session-level `a=group:DUP` naming the `a=mid:` tags of two `m=video`
blocks, and each block carries its own `c=` and `a=source-filter`.
`parse_dup_sdp` reads that document and `format_dup_sdp` writes it. The
legs come back in the order the group names them, which is the order that
decides which leg is which everywhere downstream; a document whose `DUP`
tags do not match its media blocks is refused rather than read as a
single-leg offer with an oddity, because a sender that emitted one leg
where two were meant sends unprotected essence and reports success.

A pair is two legs and no more: two is what the reconstruction takes and
what Rivermax carries, `RMX_MAX_DUP_STREAMS` being two, so a group naming
three describes a document nothing downstream can take.

An offer this library writes should be one a transmit SDK accepts
unchanged. Rivermax's `rmx_output_media_set_sdp` requires that the count
of `DUP` tags correspond to the count of `m=video` blocks, which is the
same rule stated from the other side. The writer refuses that failure in
its own direction: two legs naming one destination, port and sender are one
path written twice, and an offer saying otherwise claims a protection the
sender has not got.

**A single-leg offer stays a single-leg offer.** `parse_sdp` returns one
flow and keeps returning one; nothing about `a=group:DUP` changes what a
document without it means. This matters because the port field already
tolerates `20000/2` — a count RFC 4566 permits and that is not what makes
an offer redundant. Handed a document that does carry the group, `parse_sdp`
refuses rather than answer with whichever block came first: a consumer
reading a destination out of a two-leg offer that way joins, or sends, one
leg of two and reports success, which is the failure above reached by the
other road.

*Why a second parse rather than a wider one*: a caller holding one flow
today should not have to change to keep holding one, and the two documents
are not the same shape — resolving tags against blocks and returning the
legs in the group's order has nowhere to go in a single-flow return.

*Why the reconstruction and not the transport*: which packets arrived is a
transport question and belongs to whatever moved them, but choosing between
two copies means reading a sequence number, which the transport layer does
not do (§spec:scope-boundary).

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

Which cases carry a vector follows from that: a format verified only through
the round trip is a format neither side is tied to the documents for. So a
raster small enough to work out on paper, the interlaced layout, and the
target format each have one, the last as named packets of a frame too large
to write out whole.

Captures from real senders are the strongest evidence available without a
NIC. Where a fixture is taken off the wire its provenance is recorded
beside it — which sender, which format — because a capture whose origin is
unknown proves nothing about conformance.

They live in `tests/captures/` as compressed `.npz`, and the provenance
travels **inside the file** rather than in a note beside it, so a fixture
cannot be moved or copied away from its own history. A test asserts the
provenance is there, which is what stops it being dropped when the arrays
are regenerated. The window is a contiguous span rather than a sample:
every packet in range is present as captured, because a decimated capture
reads as loss and would prove the opposite of what it was taken for.

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
