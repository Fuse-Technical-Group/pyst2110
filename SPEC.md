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

*Chunk-wide is a floor and not an answer.* An expression over a whole chunk
can still cost far more than the octets it reads, and the two places that did
are worth naming because they are different failures. The parse gathered at a
per-packet offset where the offset was the same number for every packet
(§spec:conforming-fast-path). `SequenceTracker.observe` — which holds no wire
format at all, and is here rather than there for that reason — took a modulo
where both sequence spaces are powers of two and a mask is exact, and copied
the chunk to prepend the one scalar a straddling step needs. Measured at
2160p60 and MAXUDP 1460, the tracker went from 0.34 ms a frame to 0.20.

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
caller's with three as the default.

A packet declaring a sample row the walk did not read is flagged rather than
trimmed in silence, and **two flags rather than one**, because two things stop
the walk and the remedies differ. A packet still declaring a continuation at
the bound is `overflowed`: the sender declared past the caller's policy, and
the answer is a wider bound or a refused flow. A packet whose next SRD header
lies past the end of the buffers the caller handed over is `unreadable`: the
packet is compliant and the parse was handed too little of it, and the answer
is more buffer. Nothing about the second is particular to a header-data split
(§spec:split-segments) — any `sizes` entry ending mid-header reaches it. A
consumer wanting the pair as one condition takes the union; one told only the
union cannot tell the two apart, and they are not each other's fix.

Either flag is per packet, because the instruction it carries is to discard
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

## Segments across a header-data split §spec:split-segments

*Status: complete*

A header-data-split receiver holds a packet in two buffers, and the split is a
fixed offset while a payload header is not: ST 2110-20 permits three SRD
headers where the cut clears one, so a two- or three-segment packet leaves its
later headers at the head of the *payload* buffer.

That was reasoned to be a shape "a conforming sender at the standard datagram
size never emits." Hardware says otherwise. A Matrox ConvertIP at 1080p60
YCbCr-4:2:2 10-bit sends 1320-octet segments against a 4800-octet line, which
4800 does not divide, so **a quarter of its packets carry two segments
spanning two lines** — under the `PM=2110GPM` its SDP declares, which is the
packing mode that permits exactly this. A receiver bounded at the header
buffer drops a quarter of every frame.

`parse_payload_headers` therefore takes the payload buffer as an argument and
walks a packet's segments across the seam. Given it, every segment a packet
declares yields a descriptor. Withheld, a packet whose later headers the cut
displaced is `unreadable` and contributes none — the parse was handed one
buffer of the two, which is a fault on this side rather than the sender's. A
packet still declaring a continuation at the segment bound is `overflowed`
whether the buffer was offered or not: the bound is the caller's policy and
the payload buffer does not lift it (§spec:payload-header).

The two are separate columns because the remedy differs — pass the buffer
against one, raise the bound or refuse the sender against the other — and
because `unreadable` was not previously reported at all. Before this, a packet
whose second SRD header lay past the buffer lost its continuation silently and
emitted the segments that did fit, each with a `source` six octets early per
header nobody read. Any `sizes` entry ending mid-header did it, split or not,
so the flag is the wider fix and the split is only the commonest way to reach
the fault.

**The walk crosses the seam only for the packets that need it.** Stitching
every packet's two buffers into one contiguous view cost a per-chunk
allocation and two copies — 1.55 ms of a 2160p60 frame period, measured on a
consumer's bench. The continuation flag of the first segment already says
which packets continue, and it is read on the header buffer alone, so the
second pass runs over that subset, and over a window no wider than the headers
it reads rather than the payload behind them. A chunk that tiles its lines
selects nothing and measures what it measured before the buffer was a
parameter; the ConvertIP's own shape pays 0.55 ms of its 16.67 ms frame
period, on a 4096-packet chunk where 1024 packets cross
(`tools/parse-benchmark.py`).

`data_offset` keeps its meaning — the octets of payload header before a
packet's sample data, counting every header parsed, those read out of the
payload buffer included. The same octets therefore parse identically whether a
caller presents them as one row or as two, which is the gate the second pass
is held to. What does not survive the split is the coincidence that made
`data_offset` the payload buffer's own base: it is that only for a one-segment
packet, and a consumer indexing that buffer subtracts the cut from `source`
instead.

*Why the buffer is a parameter rather than an assumption.* A consumer may not
be able to offer it. Under a device payload ring the payload buffer is on an
accelerator and the host cannot address it, so the descriptors this parse
would need are out of reach at the moment it runs. Withholding it is
therefore a supported call rather than a degraded one, and the flag remains
the answer for that case.

*Why not widen the split instead.* A cut wide enough for three headers puts
sample data in the header buffer of every one-segment packet — the common
case — so a consumer's gather would source one frame from two buffers. The
split cuts at the shortest header for that reason, and this parse takes the
consequence rather than moving it (§spec:payload-header).

## The conforming fast path §spec:conforming-fast-path

*Status: complete*

Vectorized is not the same as fast, and the difference here is a factor of
twelve. Every parse runs over a whole chunk (§spec:interface-shape), which is
what §req:priorities asks for and what rules out a per-packet Python object.
What it does not rule out is a vectorized expression costing far more than the
data it reads: a chunk's header block at 4096 packets is 80 KiB, and the
general parse takes 1.48 ms over it — about 55 MB/s, three orders below what
numpy does with 80 KiB resident in cache.

**Where the time goes.** The payload offset is per packet, because a CSRC list
or an RTP header extension moves where the payload begins (§spec:rtp). Reading
a field at a per-packet offset is a gather: two index arrays, a bounds mask
and a promotion to sixty-four bits, for every one of the nine reads a segment
walk makes. Reading the same field at a *uniform* offset is a column slice.

**A conforming ST 2110-20 sender emits neither**, and one SRD a packet at the
standard datagram size. Every offset in such a chunk is the same number, and
the whole parse is column slices over the block read as big-endian sixteen-bit
words.

The parse therefore takes one of two paths, chosen from the chunk itself:

- The **fast path**, where every packet's flags octet is exactly version two
  with no padding, no extension and no CSRC list, no first segment sets its
  continuation flag, and every packet carries a whole header. Fields are read
  at their fixed offsets in their own width and widened once on the way out.
- The **general path**, unchanged, for every other chunk: a CSRC list, an RTP
  extension, more than one segment, a padded packet, a version that is not
  two, or a packet shorter than a whole header.

*The predicate is the whole octet, not the two bits that move the offset.*
Padding does not move where a payload begins, so a padded packet could take the
fast path — but reading the octet whole is one comparison where a mask and a
test are two, and it settles four fields as constants at the same time. The
stricter test is therefore the cheaper one, and what it costs is that a padded
packet — which no ST 2110-20 video sender emits — takes the general path.

*The choice is read, never promised.* Both parses decide from the chunk's own
bytes: `parse_payload_headers` reads the flags octet for itself rather than
inferring conformance from the offsets handed to it, because those are a
caller's array, and the path a chunk takes shall not be a caller's to pick. A sender that
changes its packetization mid-flow changes the path at the next chunk, and
nothing outside the bytes is consulted.

Two shapes fall back for a reason that is not the sender's: an odd stride and
a sub-block sliced out of a wider buffer are legitimate chunks that no column
slice can reinterpret. They take the general path rather than raising, and
they are how the suite reaches that path — a gate that stubbed the column read
out would be comparing the library against something that is not the library.

**Every descriptor is reported wide, whichever path read it.** A sixteen-bit
field on the wire is not sixteen bits in a consumer's hands: a row is scaled by
the pgroups in a line and an offset by a pgroup's octets, and under NEP 50 a
Python multiplier adopts the array's own width rather than widening it. Row
2159 of a 2160-line raster times 1152 pgroups a line is 2,487,168, which an
unsigned sixteen-bit array reports as 62,336 — not an error, a different part
of the picture. The narrow read is the fast path's own business and is widened
before it is returned, at a cost measured at 1.5 ns a packet.

*Why not the fast path alone*: a CSRC list, an RTP extension and a multi-SRD
packet are all legal. A library that parsed only the shape it prefers would
report a line that does not exist rather than a packet it could not read,
which is the failure §spec:payload-header exists to avoid.

**The two paths agree field for field.** That is a gate rather than a claim:
the suite parses the same packets both ways and compares every array — values,
shapes and widths — over hand-written vectors and over a chunk carrying a real
sender's captured sequence stream, outage included (§spec:testing). The
general path is the reference; where the two disagree, the fast path is what
is wrong. The conforming cases assert the columns really read them, so two
general parses cannot pass the gate by agreeing with each other.

**What it costs, and what pays for it.** Measured on a 4096-packet chunk of
conforming one-SRD headers, and reproducible with `tools/parse-benchmark.py`
rather than taken on trust:

| entry point | general | fast |
| --- | --- | --- |
| `parse_rtp` | 60.2 ns a packet | 15.1 |
| `parse_payload_headers` | 279.7 | 14.0 |
| both | 360.8 | 29.9 |

At 2160p60 and MAXUDP 1460 — 17,280 packets a frame — that is 6.23 ms of a
16.68 ms frame period against 0.52. The consumer that found it was spending
9.35 ms of that period inside this library and losing about a third of the
wire to the shortfall.

**Buffer policy: the scratch is the parse's, the returns are the caller's.**
The cyclic collector is not on this path — numpy arrays are freed by refcount
and this code makes no cycles, so freezing or disabling the collector buys
nothing, and saying otherwise would be wrong. What costs is *allocation*:
malloc, first-touch page faults and the free, for every temporary above
numpy's small-array cache, which a 4096-element array is well past. So an
operation writes through `out=` wherever a destination already exists, and a
flag is read by one comparison rather than by a mask and a test.
`src/pyst2110/transmit.py` holds its scratch across calls because it is an
object with a lifetime; the parses are free functions holding no state by
design (§spec:payload-header), so what they eliminate is temporaries within a
call, not allocation across calls. A parse that owned scratch would be a new
object, not a change to these.

The arrays a parse *returns* are freshly allocated and owned by the caller.
Borrowing them back would be a contract change — §req:users names
capture-analysis scripts as first-class callers, and they hold arrays across
calls — so it needs its own audit rather than a performance pass's side
effect.

*Why this rather than a compiled extension*: §spec:packaging trades a C
extension away for portability, against "a speed nothing has yet asked for".
Something has now asked, and the answer did not need one — what cost the time
was the gather, not the interpreter. A compiled parse stays available for a
raster that asks again, and it would arrive as a third path under this same
equality gate rather than as a replacement for either.

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

Single-component key signals read their pgroups from a table of their own
and are otherwise this section's arithmetic unchanged (§spec:key-signals).

## Key signals §spec:key-signals

*Status: not started*

A key signal — the alpha, matte or coverage channel accompanying a fill —
is an ST 2110-20 flow carrying one component where a fill carries three.
The library builds and reads it on the path every other sampling takes:
`sampling=KEY` has a pgroup, so line length, payload sizing, packet
counts, placement and header building all follow from §spec:geometry with
no second code path.

ST 2110-20 section 6.2.6 Table 4 gives the pgroups — one octet a pixel at
8 bits, five octets to four pixels at 10, three to two at 12, two to one
at 16. The 10-bit entry is the 4:4:4 table's divided by three, which is
what one component of the same depth costs: the bit packing is unchanged
and only the component count moves.

**A key flow's offer is constrained beyond its sampling.** Section 7.4.1
requires `colorimetry=ALPHA` and forbids a TCS value, and section 7.6
makes a TCS meaningless wherever the sampling is KEY. ALPHA is a 2022
value, so the offer carries `SSN=ST2110-20:2022` rather than the 2017
default — which §spec:sdp already derives from the colorimetry token. A
key offer is therefore written correctly today and the geometry alone
refuses it.

*Why the geometry rather than a key-shaped entry point*: a key flow
differs from a fill flow in one number, the octets a pgroup covers.
Everything a caller does with it — size a payload, count packets, place a
descriptor, stamp a header — is arithmetic already written, and a second
surface would be that arithmetic again under another name.

**A fill and its key are two flows, and this library does not pair them.**
The association belongs to the consumer: ST 2110-20 says only that key
signals "are used in relationship to fill signals" and defines no
grouping, and what groups them on a real fabric is NMOS IS-04 and IS-05,
a layer above this one (§spec:scope-boundary). It is not the
`a=group:DUP` of §spec:redundancy — that names one essence sent twice for
path diversity, where a fill and a key are two essences whose payloads
differ. A caller driving both from one frame index reads identical RTP
timestamps out of §spec:transmit-headers, both flows carrying one rate,
and that is the alignment a receiver has to work from.

`depth=16f` stays out. Table 4 gives it the same two-octet pgroup as 16,
but a half-float depth is not an integer and §spec:sdp parses and bounds
depths as integers on both paths — so admitting it is a depth model
rather than a table entry (§road:future).

The signal's own definition — what a key sample means and how it relates
to the fill's — is SMPTE RP 157, and none of it is carried here. This
library computes where a key sample sits, as it does for every other
sampling, and interprets none (§req:constraints).

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
asked for. A consumer has since asked, and the answer was a numpy expression
that reads the chunk differently rather than a language change
(§spec:conforming-fast-path) — which is what keeps this true.

*Why publishable at all*: the licence and the absence of vendor content are
what let this material be public (§spec:scope-boundary), and a library that
cannot be depended on by path alone is what makes the three-repository
split real rather than nominal.
