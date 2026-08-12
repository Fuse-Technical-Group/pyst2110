# 0.2.0 (2026-08-12)

The ST 2110-20 receive path.

### Features

* **rtp:** parse the RFC 3550 fixed header over a chunk, vectorized
* **geometry:** pgroup tables, line length, packets per line and per frame
* **framing:** track sequence loss and frame boundaries across chunks
* **payload:** parse RFC 4175 line segments into payload descriptors
* **geometry:** bound a descriptor against the raster it names
* **sdp:** read a flow and its video format out of an offer

### Fixes

* **framing:** count loss in the thirty-two-bit sequence space, so an
  outage longer than the sixteen-bit dropout bound is reported rather
  than reclassified as a resync
* **payload:** flag the overflowed packet rather than counting it, so a
  consumer can act on the instruction to discard it
* **rtp:** bound the fixed-header reads by the packet's own size
* **geometry:** name a non-positive width instead of failing inside the
  payload-size search
