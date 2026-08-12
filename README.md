# pyst2110

SMPTE ST 2110 protocol for Python: RTP and RFC 4175 headers read and
written, frame boundaries, sequence-loss accounting, pgroup geometry, and
SDP.

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
fits = fits_raster(video, payload.line, payload.offset_samples)
starts = raster_offset(video, payload.line[fits], payload.offset_samples[fits])
```

Sending is the same shape in reverse. A frame's headers are built once
for a format and payload size, then stamped per frame with the only two
fields that move — the sequence numbers and the media timestamp:

```python
from pyst2110 import FrameHeaders, choose_payload_size, format_sdp, max_payload_size

payload_size = choose_payload_size(video, max_payload_size(video))
frame = FrameHeaders(video, payload_size, ssrc=0x1234ABCD)
for index in range(frames):
    headers = frame.stamp(index)   # (packets, 20) uint8, one row per packet
    ...                            # send each header with frame.frame_offset_octets
offer = format_sdp(flow, video)    # the SDP describing what was just sent
```

`frame_offset_octets` says which octets of the frame buffer each packet
carries. Moving them is the consumer's, as on the receive side.

`stamp` hands back the same array every time, restamped in place — that
is what keeps the loop above from allocating a frame of headers per
frame. So `headers` is only valid until the next `stamp`: a caller
queueing two frames at once copies the first.

## API

Everything is re-exported from the top-level package, and the
docstrings there are the authority; `help(pyst2110)` is the index.

Both paths are built: SDP parsing and emit, the RFC 3550 header parse,
format geometry, sequence and frame tracking, RFC 4175 payload
descriptors, and the transmit header block. What is not built is listed
in [ROADMAP.md](ROADMAP.md).

## Development

```bash
uv sync
bash tools/ci.sh
```

`tools/ci.sh` is the gate CI runs — ruff, mypy, pytest.

## License

MIT — see [LICENSE](LICENSE). The standards this implements are public and
an implementation of them should be too.
