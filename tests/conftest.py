"""Fixtures shared by the header-parsing tests.

Only the array shaping lives here. Packets are written field by field in the
test that reads them back, from the standards' own diagrams, so that no parse
is tested against a writer of ours (§spec:testing).
"""

from __future__ import annotations

import numpy as np

from pyst2110 import _chunk


def chunk(*packets: list[int], stride: int = 0) -> np.ndarray:
    """A ``(packets, stride)`` uint8 view, right-padded with zeros."""
    width = max(stride, max(len(packet) for packet in packets))
    rows = np.zeros((len(packets), width), dtype=np.uint8)
    for index, packet in enumerate(packets):
        rows[index, : len(packet)] = packet
    return rows


class GeneralPathError(Exception):
    """Raised in place of a general parse, so the path taken is observable.

    Both parses choose between a conforming fast path and the general one from
    the chunk's own bytes (SPEC §spec:conforming-fast-path), and the choice is
    deliberately invisible in the result — the two agree field for field, which
    is what `test_path_equality.py` gates. So a test that means to pin the
    *selection* has to watch the parse rather than its output.
    """


def watch_paths(monkeypatch, *modules) -> None:
    """Make the general parse in each module announce itself by raising.

    A call that returns took the fast path; one that raises
    :class:`GeneralPathError` took the general one.
    """

    def refuse(*_args, **_kwargs):
        raise GeneralPathError

    for module in modules:
        monkeypatch.setattr(module, "_general", refuse)


def general_only(monkeypatch) -> None:
    """Withhold the 16-bit view, so every parse takes the general path.

    The view is what the fast path is made of, and a chunk that cannot be read
    as words is already a case both parses have to handle, so this is the
    library's own fallback rather than a seam opened for the tests.
    """
    monkeypatch.setattr(_chunk, "u16_view", lambda packets: None)
