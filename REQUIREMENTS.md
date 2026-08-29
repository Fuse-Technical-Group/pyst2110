# pyst2110 — Requirements

## Problem statement §req:problem-statement

A Python program receiving or sending SMPTE ST 2110-20 video has to read
RTP and RFC 4175 headers itself. No transport library does it: a NIC
binding moves bytes, and the standards that say what those bytes mean are
published separately from any vendor's SDK.

So each consumer writes the same parse. It has been written twice
already — once in a transport binding's examples, once in the receiver
that reads them — and a third consumer would write it again.

## Who this serves §req:users

The runtime engineer integrating ST 2110 video into a Python pipeline.
They own pixels and pacing policy. They do not want to own the bit layout
of an RFC 4175 payload header, and they should not have to install a
vendor SDK to find out what one says.

## Priorities §req:priorities

1. **Correct against the published standards.** RFC 3550, RFC 4175 and
   SMPTE ST 2110-20 decide the layouts. Where a reading is ambiguous the
   specification records which one this library takes and why.
2. **Fast enough for the frame budget.** A 4K60 flow is 259,000 packets a
   second at a jumbo datagram and 1,036,830 at the standard one, which is
   the size a conforming fabric carries. Per-packet Python is not an
   option, so every parse is vectorized over a whole chunk at once — and
   vectorized is a floor rather than the answer, because a chunk-wide
   expression can still cost more than the bytes it reads.
3. **Runnable with no hardware.** A NIC, a licence and a vendor SDK are
   all absent from CI, so nothing here may need them.
4. **Reusable beyond one transport.** A capture file and a socket are as
   valid a source of packets as a NIC is.

## Constraints §req:constraints

- **No vendor dependency.** This library depends on no SDK and no
  transport binding. A binding is one consumer, not a dependency — the
  arrow points the other way, and it points only one way.
- **No pixels.** Converting samples to and from a raster is a
  per-pixel-format kernel belonging next to a consumer's other kernels,
  where it can run on a GPU. This library computes *where* pixels go and
  never moves them.
- **Arrays in, arrays out.** The interface is numpy. A consumer hands
  over the bytes it already has, however it got them.
- **Permissive licence.** MIT. The standards are public and an
  implementation of them should be too.

## Success §req:success

A video runtime receives an ST 2110-20 flow from a commercial
transmitter and renders it, and transmits one a commercial receiver
accepts — with every RTP and RFC 4175 header on both paths parsed or
built here, and no ST 2110 bit layout written in the runtime or in its
transport binding.
