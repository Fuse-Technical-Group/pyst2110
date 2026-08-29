# pyst2110 — Roadmap

Derived from [SPEC.md](SPEC.md). Sections are in build-dependency order.
Completed work is removed; presence here means the work is not done.

## ST 2022-7 redundancy §road:redundancy

The reconstruction is delivered (§spec:redundancy). What remains is the
document that describes a redundant pair, which a sender needs before it
can send one.

### A duplicated offer §road:dup-sdp

Deliver the `a=group:DUP` half of §spec:redundancy, both ways: parse a
session-level group and its `a=mid:` tags against two `m=video` blocks
into two flows, and emit one. `parse_sdp` returns a single flow today
(`src/pyst2110/sdp.py`), and `format_sdp` writes a single media block.

The consumer is `Fuse-Technical-Group/packet-engine`'s two-leg transmit
path, which reads a flow's destination out of the SDP and so cannot
describe two legs until this can. Refuse a
document whose `DUP` tags and media blocks disagree — a sender that
emitted one leg where two were meant sends unprotected essence and
reports success.

**Verify:** A DUP offer this library writes round-trips through its own
parse to the same two flows, and both are tested against a hand-written
document from RFC 7104's own example rather than against the writer.

## Publication §road:publish

### PyPI release §road:pypi

Publish to PyPI under MIT with a trusted-publisher workflow
(§spec:packaging). Deferred until a consumer depends on a released
version rather than a git ref — the split is real once the dependency is
a version, and nothing is served by publishing before the surface has
one consumer's use behind it.

## Future §road:future

- **ipmx**: IPMX is ST 2110 with a different SDP profile and variable
  frame rates. The header layouts are unchanged; the geometry and SDP
  parsing are what would move.
- **st-2110-30-and-40**: audio and ancillary data. Different payload
  headers, same RTP and the same array-shaped interface.
- **key-and-float-formats**: two formats ST 2110-20 permits and
  `src/pyst2110/geometry.py` refuses. `sampling=KEY` has a pgroup table of
  its own (Table 4: one octet per pixel at 8 bits, five per four at 10,
  three per two at 12, two per pixel at 16 and 16f). `depth=16f` is
  half-float, section 7.4.2, sharing 16-bit's pgroup — and
  `src/pyst2110/sdp.py` refuses it twice over, in `int()` on the way in and
  against the depths it permits on the way out, so it needs a depth that is
  not an integer as much as a table entry.
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
