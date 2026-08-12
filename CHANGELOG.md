## [0.3.1](https://github.com/Fuse-Technical-Group/pyst2110/compare/v0.3.0...v0.3.1) (2026-08-12)


### Bug Fixes

* **sdp:** make a sender's offer conform to ST 2110-21 and RFC 4570 ([#2](https://github.com/Fuse-Technical-Group/pyst2110/issues/2)) ([6e90a12](https://github.com/Fuse-Technical-Group/pyst2110/commit/6e90a121dfd94f7a4fe59c7752e1cb403631bf8b))

# [0.3.0](https://github.com/Fuse-Technical-Group/pyst2110/compare/v0.2.0...v0.3.0) (2026-08-12)


### Bug Fixes

* bound every raster value against the field that carries it ([decb1ca](https://github.com/Fuse-Technical-Group/pyst2110/commit/decb1ca78a3d484e0bec8f4ec1d634ed8afe248b))
* **sdp:** validate every value an offer writes, and write what was validated ([b66cc9f](https://github.com/Fuse-Technical-Group/pyst2110/commit/b66cc9f3ef27b163c4d757f58fd883b7e1cfe97d))
* **transmit:** refuse a raster with no rows by name ([878c915](https://github.com/Fuse-Technical-Group/pyst2110/commit/878c9153ee743f4671508a1aa51c882f559b5cae))
* **transmit:** size a payload against the UDP limit rather than past it ([f0d0ae6](https://github.com/Fuse-Technical-Group/pyst2110/commit/f0d0ae63e7c69bcf456841daf46b36918332e33b))


### Features

* **sdp:** write an SDP offer describing a flow and its video format ([84c096f](https://github.com/Fuse-Technical-Group/pyst2110/commit/84c096fa7326caa9d37536bde5690a4ab621e236))
* **transmit:** build a frame's headers once and stamp them per frame ([706918e](https://github.com/Fuse-Technical-Group/pyst2110/commit/706918e9ce6a0e7867b00262825939fa659f02c2))

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
