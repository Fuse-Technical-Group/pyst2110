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
| **pyst2110** | RTP + RFC 4175 headers, geometry, SDP, ST 2110-21 timing |
| The consuming runtime | Pixels — packing, unpacking, colour |

The layer between a transport and a raster: it says what packet headers
mean and where their payloads belong, and never moves a pixel itself.
numpy is the only runtime dependency — no vendor SDK, no transport
binding, no NIC and no licence, so it runs anywhere CI does.

## Install

From PyPI:

```bash
uv add pyst2110
```

## Usage

```python
from pyst2110 import parse_sdp, parse_video_format

offer = open("flow.sdp").read()
flow = parse_sdp(offer)  # destination address, port, source filter
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

The offsets index a frame buffer of `frame_bytes(video)` octets, which
holds the pgroups line after line in wire order. Every sampling ST 2110-20
lists is carried: RGB and YCbCr, CLYCbCr and ICtCp at 4:4:4, 4:2:2 and
4:2:0, and XYZ, at the depths its tables give. A 4:2:0 pgroup spans two
rows, so one line of that buffer holds a pair of rows, and `pgroup_rows`
reports the two. The sample range a flow declares with `RANGE` is
`video.range`, one of `RANGES`, and like the colorimetry it is carried
without interpretation.

Sending is the same shape in reverse. A frame's headers are built once
for a format and payload size, then stamped per frame with the only two
fields that move — the sequence numbers and the media timestamp:

```python
from pyst2110 import FrameHeaders, choose_payload_size, format_sdp, max_payload_size

payload_size = choose_payload_size(video, max_payload_size(video))
frame = FrameHeaders(video, payload_size, ssrc=0x1234ABCD)
for index in range(frames):
    headers = frame.stamp(index)  # (packets, 20) uint8, one row per packet
    ...  # send each header with frame.frame_offset_octets
offer = format_sdp(flow, video, session_name="my sender")  # what was sent
```

The offer carries the media type parameters ST 2110-20 section 7.2 and
ST 2110-21 section 8.1 require of a sender. `TP` describes the pacing,
which is the transport's and not this library's, so `sender_type=` is
the caller's to set — it defaults to `2110TPN`. A multicast `flow` names
its sender or passes `any_source=True`; see §spec:sdp for why that is a
choice rather than a default.

An ST 2022-7 redundant pair is one offer and not two. `format_dup_sdp`
writes RFC 7104's grouping — a session-level `a=group:DUP` over two
`m=video` blocks, each with its own address and sender — and
`parse_dup_sdp` reads both legs back, in the order the group names them:

```python
from pyst2110 import format_dup_sdp, parse_dup_sdp

offer = format_dup_sdp(red, blue, video, session_name="my sender")
first, second = parse_dup_sdp(offer)
```

A document whose `DUP` tags and media blocks disagree is refused rather
than read as a single-leg offer, and `parse_sdp` refuses a grouped offer
for the same reason: a sender that emitted one leg where two were meant
sends unprotected essence and reports success.

One essence over several flows is one offer too. SMPTE RP 2110-23 carries
a picture as four or sixteen ST 2110-20 flows, each a sub-image, grouped
by `a=group:MULTI-2SI` (two-sample interleave) or `a=group:MULTI-SD`
(square division). `format_group_sdp` writes the group from the
constituent flows and their shared sub-image format, and
`parse_group_sdp` reads it back:

```python
from pyst2110 import format_group_sdp, parse_group_sdp, to_essence

offer = format_group_sdp(flows, sub_image, session_name="my sender")
group = parse_group_sdp(offer)  # division, flows in order, sub-image format
raster = {
    "width": group.essence_width,
    "height": group.essence_height,
    "division": group.division,
    "count": len(group.flows),
}
x, y = to_essence(2, columns, row, **raster)  # where constituent 2's pixels sit
```

`parse_sdp` and `parse_video_format` refuse a group document and name
`parse_group_sdp`, because a constituent read alone is a quarter of the
picture. `pyst2110.division` maps essence pixels to constituent pixels
and back, closed form over coordinate arrays, and its `split` and
`reassemble` are the reference move an accelerated gather is checked
against. The temporal `PHASED` division and ST 2022-7 over a group are
refused by name.

`session_name` defaults to the single space RFC 4566 section 5.3 prescribes
for a session with no meaningful name — and at least one transmit SDK refuses
it, so name the session where a sender will read the offer back. NVIDIA
Rivermax reports `'x=<token>' format not found` and fails stream creation,
accepting the same document once it is named.

`frame_offset_octets` says which octets of the frame buffer each packet
carries. Moving them is the consumer's, as on the receive side.

`stamp` hands back the same array every time, restamped in place — that
is what keeps the loop above from allocating a frame of headers per
frame. So `headers` is only valid until the next `stamp`: a caller
queueing two frames at once copies the first.

## API

Everything is re-exported from the top-level package, and the
docstrings there are the authority; `help(pyst2110)` is the index.

Both paths are built: SDP parsing and emit — the single-leg offer, the
ST 2022-7 pair and the RP 2110-23 group — the RP 2110-23 division mapping,
the RFC 3550 header parse, format geometry, sequence and
frame tracking, RFC 4175 payload descriptors, the transmit header block,
the ST 2022-7 reconstruction of one flow from two legs, and the ST 2110-21
timing model — read schedules, sender limits, and the two leaky buckets
that judge a capture's emission instants. What is not built is listed in
[ROADMAP.md](ROADMAP.md).

## Development

```bash
uvx --from "git+https://github.com/repentsinner/bug-free-happiness@v0.10.0#subdirectory=.github/tools/python" bfh-check
```

`bfh-check` runs the checks CI runs, at CI's tool versions: `uv sync --locked`, ruff's lint and format checks, pyright, and pytest. Its release is the one `.github/workflows/quality.yml` pins.

`tests/test_properties.py` checks the parsers and round trips on inputs
Hypothesis generates. CI draws 100 examples a property; for a longer local
search, run `HYPOTHESIS_PROFILE=thorough uv run pytest tests/test_properties.py`,
which draws 10,000.

On a pull request that changes `src/pyst2110/`, the advisory `mutation`
workflow mutates the modules it touched, on the base branch and on the change,
and lists in its job summary each mutant the change's tests let survive. To run
the same report locally, across every module and every core:

```bash
uv run --group mutation python tools/mutation_report.py run \
    _chunk division framing geometry payload redundancy rtp sdp timing transmit \
    --out survivors.json
```

## License

MIT — see [LICENSE](LICENSE). The standards this implements are public and
an implementation of them should be too.
