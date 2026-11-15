"""Report the mutants a change leaves alive in the modules it touched.

``run`` mutates the named modules with mutmut, runs the suite against each
mutant, and writes the survivors to JSON. ``compare`` reads two such files, the
base branch's and the change's, and writes a Markdown report of the survivors
the change introduced.

Survivors are matched by what the mutation does — the function and the lines
it changed — and not by mutmut's numbering, which shifts whenever code above a
mutant moves. A survivor present on both sides is one already known, usually an
equivalent mutation no test can kill, so the report leaves it out.

    uv run --group mutation python tools/mutation_report.py run timing sdp \
        --out head.json
    python tools/mutation_report.py compare base.json head.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PACKAGE = "pyst2110"
# A mutant's name ends in its number, which moves when code above it does.
_NUMBER = re.compile(r"__mutmut_\d+$")
# The report lists this many new survivors in full and counts the rest.
_LISTED = 40


def _mutmut(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - this interpreter, fixed command
        [sys.executable, "-m", "mutmut", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def _changed_lines(name: str) -> list[str]:
    shown = _mutmut("show", name).stdout.splitlines()
    return [
        line.rstrip()
        for line in shown
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]


def run(modules: list[str], out: Path, workers: int) -> int:
    """Mutates ``modules``, runs the suite, and writes the survivors to ``out``."""
    patterns = [f"{PACKAGE}.{module}.*" for module in modules]
    result = _mutmut("run", "--max-children", str(workers), *patterns)
    if result.returncode:
        sys.stderr.write(result.stdout[-4000:] + result.stderr[-4000:])
        return result.returncode
    survivors = []
    for line in _mutmut("results").stdout.splitlines():
        name, _, status = line.strip().rpartition(": ")
        if status != "survived":
            continue
        changed = _changed_lines(name)
        survivors.append(
            {
                "name": name,
                "function": _NUMBER.sub("", name),
                "signature": "\n".join([_NUMBER.sub("", name), *changed]),
                "diff": changed,
            }
        )
    out.write_text(json.dumps({"modules": modules, "survivors": survivors}, indent=1))
    return 0


def compare(base: Path, head: Path) -> str:
    """Returns the Markdown report of survivors ``head`` has and ``base`` lacks."""
    before = json.loads(base.read_text())["survivors"]
    after = json.loads(head.read_text())
    known = {survivor["signature"] for survivor in before}
    new = [s for s in after["survivors"] if s["signature"] not in known]
    gone = len(known - {s["signature"] for s in after["survivors"]})
    modules = ", ".join(f"`{module}`" for module in after["modules"])
    lines = [
        "## Mutation report",
        "",
        f"Modules this change touched: {modules}.",
        "",
        f"- **{len(new)}** new surviving mutants: changes no test caught.",
        f"- **{gone}** survivors from the base branch now caught or gone.",
        f"- **{len(after['survivors'])}** surviving in these modules in all.",
    ]
    if new:
        lines += [
            "",
            "Each is a change to the code that every test still passes. Pin the "
            "behaviour it breaks with a test, or leave it where the mutation "
            "cannot change what the code does.",
            "",
        ]
        for survivor in new[:_LISTED]:
            lines += [f"`{survivor['function']}`", "```diff", *survivor["diff"], "```"]
        if len(new) > _LISTED:
            lines.append(f"…and {len(new) - _LISTED} more.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Runs the ``run`` or ``compare`` subcommand."""
    parser = argparse.ArgumentParser(prog="mutation_report")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="mutate modules and record survivors")
    run_parser.add_argument("modules", nargs="+", help="module names under pyst2110")
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    compare_parser = commands.add_parser("compare", help="report new survivors")
    compare_parser.add_argument("base", type=Path)
    compare_parser.add_argument("head", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "run":
        return run(arguments.modules, arguments.out, arguments.workers)
    sys.stdout.write(compare(arguments.base, arguments.head))
    return 0


if __name__ == "__main__":
    sys.exit(main())
