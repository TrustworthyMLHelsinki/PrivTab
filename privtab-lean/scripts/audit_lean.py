#!/usr/bin/env python3
"""Audit the DP-MHCA proofs for banned shortcuts, statement drift, and axioms."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDITED_FILES = (
    Path("PrivTabLean/DPMHCA.lean"),
    Path("PrivTabLean/FrozenStatements.lean"),
    Path("PrivTabLean/TanhAttention.lean"),
    Path("PrivTabLean/FrozenTanhAttention.lean"),
)
FROZEN_HASHES = Path("scripts/frozen.sha256")

BANNED = [
    "sorry",
    "sorryAx",
    "native_decide",
    "admit",
    "unsafe",
    "implemented_by",
    "ofReduceBool",
]


def strip_lean_comments(source: str) -> str:
    """Remove nested block comments and line comments while preserving line numbers."""
    result: list[str] = []
    index = 0
    depth = 0
    while index < len(source):
        pair = source[index : index + 2]
        if depth == 0 and pair == "--":
            while index < len(source) and source[index] != "\n":
                result.append(" ")
                index += 1
        elif pair == "/-":
            depth += 1
            result.extend("  ")
            index += 2
        elif depth > 0 and pair == "-/":
            depth -= 1
            result.extend("  ")
            index += 2
        else:
            character = source[index]
            result.append(character if depth == 0 or character == "\n" else " ")
            index += 1
    return "".join(result)


def check_frozen_hashes() -> list[str]:
    """Ensure the reviewed definitions and theorem statements have not changed."""
    failures: list[str] = []
    pins = PROJECT_ROOT / FROZEN_HASHES
    for line in pins.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected, relative_name = line.split(maxsplit=1)
        frozen_file = PROJECT_ROOT / relative_name
        actual = hashlib.sha256(frozen_file.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(
                f"{relative_name}: frozen statement checksum changed "
                f"(expected {expected}, found {actual})"
            )
    return failures


def check_banned_keywords() -> list[str]:
    """Check executable proof code for prohibited shortcuts and axiom declarations."""
    token = re.compile(
        r"(?<![A-Za-z0-9_])(?:" + "|".join(map(re.escape, BANNED)) + r")(?![A-Za-z0-9_])"
    )
    failures: list[str] = []
    for relative_path in AUDITED_FILES:
        path = PROJECT_ROOT / relative_path
        code = strip_lean_comments(path.read_text(encoding="utf-8"))
        for line_number, line in enumerate(code.splitlines(), 1):
            if re.match(r"\s*axiom\b", line):
                failures.append(f"{relative_path}:{line_number}: custom axiom declaration")
            for match in token.finditer(line):
                failures.append(
                    f"{relative_path}:{line_number}: banned token {match.group()!r}"
                )
    return failures


def run_command(command: list[str], failure_message: str) -> tuple[str, list[str]]:
    """Run a project command and return its combined output and any failure."""
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.stdout, [] if result.returncode == 0 else [failure_message]


def run_kernel_checks() -> tuple[str, list[str]]:
    """Compile frozen statement checks and the kernel-side axiom assertions."""
    return run_command(
        ["lake", "build", "PrivTabLean.ProofAudit"],
        "Lean rejected a frozen statement or kernel axiom check.",
    )


def main() -> int:
    failures = check_frozen_hashes()
    failures.extend(check_banned_keywords())

    build_output, build_failures = run_command(
        ["lake", "build"], "The project does not build cleanly."
    )
    failures.extend(build_failures)

    lean_output, kernel_failures = run_kernel_checks()
    failures.extend(kernel_failures)

    if failures:
        print("Lean proof audit failed:")
        for failure in failures:
            print(f"- {failure}")
        if build_failures:
            print("\nLake output:")
            print(build_output.rstrip())
        if kernel_failures:
            print("\nProof-audit output:")
            print(lean_output.rstrip())
        return 1

    print(f"Frozen statement checksum matches {FROZEN_HASHES}.")
    for path in AUDITED_FILES:
        print(f"No banned tokens or custom axiom declarations found in {path}.")
    print("The full Lake build passes.")
    print("Frozen theorem statements match the proved theorem statements.")
    print("Kernel axiom audit passed: only propext, Classical.choice, and Quot.sound are used.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
