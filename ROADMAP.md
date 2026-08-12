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

## Future §road:future

- **ipmx**: IPMX is ST 2110 with a different SDP profile and variable
  frame rates. The header layouts are unchanged; the geometry and SDP
  parsing are what would move.
- **st-2110-30-and-40**: audio and ancillary data. Different payload
  headers, same RTP and the same array-shaped interface.
- **st-2022-7**: reconstructing one stream from two redundant legs by
  sequence number. Needs a consumer carrying two flows before its
  interface can be designed against anything real.
- **key-and-float-formats**: two formats ST 2110-20 permits and
  `src/pyst2110/geometry.py` refuses. `sampling=KEY` has a pgroup table of
  its own (Table 4: one octet per pixel at 8 bits, five per four at 10,
  three per two at 12, two per pixel at 16 and 16f). `depth=16f` is
  half-float, section 7.4.2, sharing 16-bit's pgroup — and
  `src/pyst2110/sdp.py` rejects it in `int()` before geometry sees it, so
  it needs a depth that is not an integer as much as a table entry.
- **sdp-clock-attributes**: a conformant ST 2110-10 SDP carries
  `a=ts-refclk:` and `a=mediaclk:`, and `format_sdp` writes neither
  (`src/pyst2110/sdp.py`). They name a PTP grandmaster and a media clock
  this library has no model of, so a caller that owns the clock appends
  them. Worth taking on once something here knows what the clock is.
- **transmit-block-packing**: `FrameHeaders` writes one SRD header a
  packet, which is ST 2110-20's General Packing Mode
  (`src/pyst2110/transmit.py`). The Block Packing Mode of section 6.3.3
  packs 7x180 octets and spans sample rows with the continuation bit,
  which needs a payload size the geometry here does not compute and a
  multi-SRD builder.
- **compressed-2110-22**: JPEG XS carries its own payload header
  (RFC 9134) and no raster geometry, so it shares only the RTP parse.
