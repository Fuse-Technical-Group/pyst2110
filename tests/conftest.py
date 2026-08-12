"""Fixtures shared by the header-parsing tests.

Only the array shaping lives here. Packets are written field by field in the
test that reads them back, from the standards' own diagrams, so that no parse
is tested against a writer of ours (§spec:testing).
"""

from __future__ import annotations

import numpy as np


def chunk(*packets: list[int], stride: int = 0) -> np.ndarray:
    """A ``(packets, stride)`` uint8 view, right-padded with zeros."""
    width = max(stride, max(len(packet) for packet in packets))
    rows = np.zeros((len(packets), width), dtype=np.uint8)
    for index, packet in enumerate(packets):
        rows[index, : len(packet)] = packet
    return rows
