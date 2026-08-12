# pyst2110 — Roadmap

Derived from [SPEC.md](SPEC.md). Sections are in build-dependency order.
Completed work is removed; presence here means the work is not done.

## Transmit path §road:transmit-slice

The same headers, written. A consuming runtime's ST 2110 sink backend is
what needs them.

### SDP emit §road:sdp-emit

Write an SDP describing a flow this library can also parse
(`src/pyst2110/sdp.py`) (§spec:sdp). Parsing landed with the seed; a
transmitter is configured by an offer it produces, and an ST 2110 receiver
is patched by one over IS-05, so emitting is what makes the pair
symmetric. No dependency.

### Frame header block §road:transmit-headers

Build a frame's RTP and RFC 4175 headers once for a format and payload
size, and stamp per frame with sequence numbers and media timestamp
(`src/pyst2110/transmit.py`) (§spec:transmit-headers). Builds on the pgroup and packet
arithmetic in `src/pyst2110/geometry.py`, which the receive path landed.

**Verify:** Feed the built headers back through the receive parse and
recover the frame shape they were built for. Then transmit them over a
NIC binding and confirm a commercial receiver locks.

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
- **maxudp**: read `MAXUDP` off the `a=fmtp:` line into
  `SdpVideo.max_udp` (`src/pyst2110/sdp.py`), then default
  `choose_payload_size`'s limit from it. The field exists but nothing
  populates it, so a caller passing it today gets the 1460 default rather
  than the size the sender declared.
- **compressed-2110-22**: JPEG XS carries its own payload header
  (RFC 9134) and no raster geometry, so it shares only the RTP parse.
