## [0.7.2](https://github.com/Fuse-Technical-Group/pyst2110/compare/v0.7.1...v0.7.2) (2026-08-29)


### Performance Improvements

* read a thirty-two-bit field as one column, and a gapless chunk in closed form ([30bdd43](https://github.com/Fuse-Technical-Group/pyst2110/commit/30bdd4306af84f8e016e006c707b1b47531d6acd))

## [0.7.1](https://github.com/Fuse-Technical-Group/pyst2110/compare/v0.7.0...v0.7.1) (2026-08-29)


### Bug Fixes

* **parse:** report every descriptor wide, and select from the chunk's bytes ([ec90fb5](https://github.com/Fuse-Technical-Group/pyst2110/commit/ec90fb5f9414927cb2e9eddc7a587a25e798fd65))


### Performance Improvements

* **framing:** take the sequence step by mask and slice, not modulo and copy ([d2981fc](https://github.com/Fuse-Technical-Group/pyst2110/commit/d2981fcffcbef848de17f252f0b9ce4c889c8634))
* **parse:** read a conforming chunk by column instead of by gather ([7fae6b5](https://github.com/Fuse-Technical-Group/pyst2110/commit/7fae6b59a3c28d6c24de91c8f3506af3278f82b5))

# [0.7.0](https://github.com/Fuse-Technical-Group/pyst2110/compare/v0.6.1...v0.7.0) (2026-08-29)


### Bug Fixes

* **api:** re-export the ST 2022-7 reconstruction from the package root ([4aec630](https://github.com/Fuse-Technical-Group/pyst2110/commit/4aec630b0534e996b9b6788925cc0bf4fba2c3ad))
* **sdp:** refuse a pair whose two legs describe one path ([0d7d078](https://github.com/Fuse-Technical-Group/pyst2110/commit/0d7d078f535f93655ccb1c1a85d8695dbb09c8b9))


### Features

* **sdp:** read and write the duplicated offer of an ST 2022-7 pair ([115b0e4](https://github.com/Fuse-Technical-Group/pyst2110/commit/115b0e431ecf498f76f23e1871796f899db83184))

## [0.6.1](https://github.com/Fuse-Technical-Group/pyst2110/compare/v0.6.0...v0.6.1) (2026-08-28)


### Performance Improvements

* **payload:** leave the SRD walk when no packet continues ([1e3a5e5](https://github.com/Fuse-Technical-Group/pyst2110/commit/1e3a5e5438889fa4ce99c91f2d14541a65c8de81))

# [0.6.0](https://github.com/Fuse-Technical-Group/pyst2110/compare/v0.5.0...v0.6.0) (2026-08-28)


### Features

* **redundancy:** reconstruct one flow from two ST 2022-7 legs ([5f6d296](https://github.com/Fuse-Technical-Group/pyst2110/commit/5f6d29608b1b21bf81adfd36d592dfd4d5f3a0be))

# [0.5.0](https://github.com/Fuse-Technical-Group/pyst2110/compare/v0.4.0...v0.5.0) (2026-08-27)


### Features

* **timing:** select the ST 2110-21 edition the sender was built to ([3e6d2a5](https://github.com/Fuse-Technical-Group/pyst2110/commit/3e6d2a570d9f2cda11890d87c6a83e0e00373d64))

# [0.4.0](https://github.com/Fuse-Technical-Group/pyst2110/compare/v0.3.1...v0.4.0) (2026-08-27)


### Bug Fixes

* **sdp:** bound exactframerate and read only ASCII decimal digits ([6b99c11](https://github.com/Fuse-Technical-Group/pyst2110/commit/6b99c11f892f38545355a7c6885cfebdc176e653))
* **timing:** bound the phase term, and count the RTP timestamp column ([2473885](https://github.com/Fuse-Technical-Group/pyst2110/commit/2473885d099e4cad6c945d36d921a364210cb599))


### Features

* **sdp:** carry the ST 2110-21 section 8 parameters both ways ([db4dd4f](https://github.com/Fuse-Technical-Group/pyst2110/commit/db4dd4ffbc6be3c4da066da52e73f4b1957991af))
* **timing:** model ST 2110-21 schedules, limits and traffic buckets ([fac8ee2](https://github.com/Fuse-Technical-Group/pyst2110/commit/fac8ee20cf164f478455c58c9e37db272b051854))

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
