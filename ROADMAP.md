# pyst2110 — Roadmap

Derived from [SPEC.md](SPEC.md). Sections are in build-dependency order.
Completed work is removed; presence here means the work is not done.

## Publication §road:publish

### PyPI release §road:pypi

Publish to PyPI under MIT with a trusted-publisher workflow
(§spec:packaging). Deferred until a consumer depends on a released
version rather than a git ref — the split is real once the dependency is
a version, and nothing is served by publishing before the surface has
one consumer's use behind it.

## Key signals §road:key-signals

### Key pgroup geometry §road:key-geometry

Add ST 2110-20 Table 4 to the sampling-to-pgroup map in
`src/pyst2110/geometry.py`, so line length, payload sizing, packet counts,
placement and `FrameHeaders` all carry a `sampling=KEY` flow
(§spec:key-signals).

### Key offer constraints §road:key-offer

Refuse a mismatched key offer in `src/pyst2110/sdp.py` on both paths — a
`sampling=KEY` flow declaring any colorimetry but `ALPHA`, and an `ALPHA`
colorimetry on any other sampling (§spec:key-signals). Depends on
§road:key-geometry.

**Verify:** A 1920x1080 10-bit `KEY` flow at 60 fps reports a `(5, 4)`
pgroup, a chosen payload of 1200 octets, two packets a line and 2160 a
frame, and `FrameHeaders` builds and stamps that block. Its emitted offer
carries `sampling=KEY`, `colorimetry=ALPHA` and `SSN=ST2110-20:2022`, and
no TCS. The same format declaring `colorimetry=BT709` is refused naming
the parameter.

## The conforming fast path §road:fast-path

The parse chooses between a column read and the general walk from the chunk's
own bytes, and the two agree field for field under a CI gate
(§spec:conforming-fast-path). Measured at 360.8 ns a packet against 29.9 on a
4096-packet chunk — 6.23 ms of a 2160p60 frame period against 0.52.

### What the column read left behind §road:fast-path-second-pass

Take the remaining per-chunk waste the fast path exposed, in
`src/pyst2110/rtp.py`, `src/pyst2110/payload.py`, `src/pyst2110/framing.py` and
`src/pyst2110/_chunk.py` (§spec:conforming-fast-path, §spec:interface-shape).
Measured on a 4096-packet chunk, each independently:

- A thirty-two-bit field read as two sixteen-bit columns, where the chunk views
  as `>u4` for nothing: `parse_rtp` fast 15.0 to 10.3 ns a packet.
- `SequenceTracker.observe` has no conforming case of its own — a gapless chunk
  still makes three bool passes, a masked subtract, a `flatnonzero` and a
  `cumsum`, where every step being one makes the span update closed-form:
  50.4 to 15.5 us a chunk.
- The general path's post-loop work scales with `max_segments` rather than with
  the segments the loop reached, and materializes a `count x max_segments`
  index array to mask it down again: 265 to 119 ns a packet.
- `frame_boundaries` still shifts by one with the `np.concatenate` the tracker
  stopped using, and `FrameTracker.observe` copies a `cumsum` that is already
  int64.
- With no sizes, `limits` fills and scans a 32 KiB constant array so
  `_conforming` can read one number off it.

**Measure in place, not in isolation.** A reduction that replaced a mask and
`.any()` on the offset column benchmarked faster alone and made the fast path
twice as slow in situ — numpy's reduction over a big-endian strided column is
the unbuffered loop. Every item above is quoted from an isolated measurement
and shall be re-measured through `tools/parse-benchmark.py` before it lands.

### Returns the caller borrows §road:fast-path-borrowed-returns

Let a caller hand the parse the arrays to fill, so a steady-state receiver
allocates nothing per chunk, in `src/pyst2110/rtp.py` and
`src/pyst2110/payload.py` (§spec:conforming-fast-path). Blocked on a spec
paragraph and a consumer audit, not on the code: the returns are owned by the
caller today, and §req:users names capture-analysis scripts that hold arrays
across calls. The one surveyed consumer's `FrameAssembler` rescales every
field into new arrays inside the same call, so a borrowed contract would suit
it — one consumer is not the audit.

**Verify:** A receive loop over a hundred chunks allocates no per-chunk array
after the first, and every existing caller reads the same values it reads
today.

## ST 2022-7 on a frame path §road:redundancy-cost

`redundancy.reconstruct` costs 235.3 ns a packet — 963.8 us for a 4096-packet
dual-leg chunk, and 4.07 ms of a 2160p60 frame period — measured against this
tree at 2026-08-28, with a 600-packet hole in one leg reading 3.88 ms. That is
the order `parse_payload_headers` was at before §spec:conforming-fast-path, on
a function nothing has looked at.

**Not a regression and not yet a fault.** No consumer calls it per chunk
today, so nothing is paying this. What makes it worth a section rather than a
Future bullet is that a consumer is coming: backlit_molecule's ST 2022-7 work
is specified, and the ConvertIP bench rig already runs paired legs, so the
first dual-leg receiver walks into a frame budget a quarter spent before it
places a pixel. §req:priorities is the standing obligation — a parse is fast
enough for the frame budget — and §spec:redundancy states the arithmetic
without stating its cost, which is the gap a `/compose:plan` pass should
close before this is built.

**Where the time goes is unknown.** The figure is a measurement and the cause
is not: nobody has profiled it, and the fast path's own lesson is that the
obvious candidate can be wrong in place — a reduction that benchmarked faster
in isolation halved the column read's speed in situ. Profile before
prescribing, and re-measure through the benchmark.

### The reconstruction inside the frame budget §road:redundancy-in-budget

Bring `redundancy.reconstruct` to the order the chunk parse now runs at, in
`src/pyst2110/redundancy.py` (§spec:redundancy, §req:priorities). Profile
first: the cost is measured and its cause is not.

### The benchmark covers the pair §road:redundancy-benchmark

Extend `tools/parse-benchmark.py` to time a dual-leg chunk beside the single
parses it already reports, so a second host confirms or contradicts the figure
above rather than taking it (§spec:redundancy). Depends on
§road:redundancy-in-budget for something to report.

**Verify:** Run the benchmark on a host and confirm it reports the pair beside
the parses, and that reconstruction has left the hundreds of nanoseconds a
packet. Feed `reconstruct` two legs sharing no sequence range and confirm it
still refuses rather than reporting total loss on both, and two legs where one
carries a number twice and confirm the earliest arrival is still the one kept
— the arithmetic §spec:redundancy fixes does not move because it got faster.
Feed it the `convertip-2022-7-outage` capture and confirm the pair's reported
loss, and the path differential, are what they were before.

## Future §road:future

- **ipmx**: IPMX is ST 2110 with a different SDP profile and variable
  frame rates. The header layouts are unchanged; the geometry and SDP
  parsing are what would move.
- **st-2110-30-and-40**: audio and ancillary data. Different payload
  headers, same RTP and the same array-shaped interface.
- **float-formats**: `depth=16f` is half-float, section 7.4.2, sharing
  16-bit's pgroup — and `src/pyst2110/sdp.py` refuses it twice over, in
  `int()` on the way in and against the depths it permits on the way out,
  so it needs a depth that is not an integer as much as a table entry.
  `sampling=KEY` left this bullet for §road:key-signals.
- **sdp-clock-attributes**: the one known ST 2110-10 non-conformance in
  the emitted offer. Section 8.2 — "All stream descriptions shall have a
  `ts-refclk` attribute" — and section 8.3 — "All stream descriptions
  shall have a media-level `mediaclk` attribute" — and `format_sdp`
  writes neither (`src/pyst2110/sdp.py`). Both name something this
  library has no model of: a PTP grandmaster's clock identity and domain,
  and whether the media clock is locked to it (`mediaclk:direct=0`) or
  free-running (`mediaclk:sender`). Neither is derivable from a frame
  index, so a caller that owns the clock appends them. Worth taking on
  once something here knows what the clock is. A receiver that enforces
  ST 2110-21 section 7.2.3 refuses an offer without them.
- **transmit-block-packing**: `FrameHeaders` writes one SRD header a
  packet, which is ST 2110-20's General Packing Mode
  (`src/pyst2110/transmit.py`). The Block Packing Mode of section 6.3.3
  packs 7x180 octets and spans sample rows with the continuation bit,
  which needs a payload size the geometry here does not compute and a
  multi-SRD builder.
- **timing-psf-and-field-pairing**: what §spec:timing defers. The
  `segmented` SDP parameter of ST 2110-20 is not parsed, so PsF reads as
  interlaced — which Table 1's 1125-line row covers — rather than as the
  segment schedule it is; and an interlaced capture that opens mid-frame
  pairs fields by arrival order unless the caller supplies the F bit.
- **compressed-2110-22**: JPEG XS carries its own payload header
  (RFC 9134) and no raster geometry, so it shares only the RTP parse.
