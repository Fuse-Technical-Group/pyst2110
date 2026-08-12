# pyst2110

SMPTE ST 2110 protocol for Python: RTP and RFC 4175 header parsing, frame
boundaries, sequence-loss accounting, pgroup geometry, and SDP.

> [REQUIREMENTS.md](REQUIREMENTS.md) — the problem.
> [SPEC.md](SPEC.md) — the design and its rationale.
> [ROADMAP.md](ROADMAP.md) — what is not built yet.

| Layer | Owns |
| --- | --- |
| A transport binding (a NIC binding, a socket, a capture file) | Moving bytes |
| **pyst2110** | RTP + RFC 4175 headers, geometry, SDP |
| The consuming runtime | Pixels — packing, unpacking, colour |

The layer between a transport and a raster: it says what packet headers
mean and where their payloads belong, and never moves a pixel itself.
numpy is the only runtime dependency — no vendor SDK, no transport
binding, no NIC and no licence, so it runs anywhere CI does.

## Install

A git dependency pinned by tag. Publication to PyPI waits for a consumer
depending on a released version (§road:pypi):

```bash
uv add "pyst2110 @ git+https://github.com/Fuse-Technical-Group/pyst2110@v0.2.0"
```

## Usage

```python
from pyst2110 import parse_sdp, parse_video_format

offer = open("flow.sdp").read()
flow = parse_sdp(offer)          # destination address, port, source filter
video = parse_video_format(offer)  # width, height, rate, sampling, depth
```

Parsing is vectorized over whole chunks: every function takes a
two-dimensional `uint8` array, one row per packet, and returns one array
per field — the shape a header-data-split receiver already hands out.

Header fields are reported as the wire declared them, this being a parse
and not a filter. A packet is free to name a row outside the image, so
`fits_raster` masks the descriptors that name a place inside the flow's
raster and a consumer places only those:

```python
fits = fits_raster(video, payload.line, payload.offset)
starts = payload.line[fits] * line_bytes(video) + byte_offset(video, payload.offset[fits])
```

## API

Everything is re-exported from the top-level package, and the
docstrings there are the authority; `help(pyst2110)` is the index.

The receive path is built: SDP parsing, the RFC 3550 header parse,
format geometry, sequence and frame tracking, and RFC 4175 payload
descriptors. Transmit headers and SDP emit are specified and not yet
built — see [ROADMAP.md](ROADMAP.md) for the order they land in.

## Development

```bash
uv sync
bash tools/ci.sh
```

`tools/ci.sh` is the gate CI runs — ruff, mypy, pytest.

## License

MIT — see [LICENSE](LICENSE). The standards this implements are public and
an implementation of them should be too.
