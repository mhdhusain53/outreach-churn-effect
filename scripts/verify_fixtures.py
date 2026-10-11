#!/usr/bin/env python3
"""Recompute SHA-256 digests for held-out fixtures and compare to restore_heldout.py."""
from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "task" / "tests" / "fixtures" / "heldout"
RESTORE = ROOT / "task" / "tests" / "restore_heldout.py"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_digests() -> dict[str, str]:
    tree = ast.parse(RESTORE.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "EXPECTED_SHA256":
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"EXPECTED_SHA256 not found in {RESTORE}")


def main() -> int:
    expected = expected_digests()
    failed = 0
    for name, digest in sorted(expected.items()):
        path = FIXTURE_DIR / name
        if not path.is_file():
            print(f"MISSING  {name}")
            failed += 1
            continue
        got = sha256_file(path)
        if got == digest:
            print(f"OK       {name}")
        else:
            print(f"MISMATCH {name}")
            print(f"  expected {digest}")
            print(f"  got      {got}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
