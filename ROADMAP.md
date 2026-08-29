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
