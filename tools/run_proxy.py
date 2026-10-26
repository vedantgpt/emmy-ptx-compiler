#!/usr/bin/env python3
"""Compile every problems_proxy IR, then score it on the hosted runner.

    .venv\\Scripts\\python.exe tools/run_proxy.py
    .venv\\Scripts\\python.exe tools/run_proxy.py --compile-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROXY = ROOT / "problems_proxy"
COMPILER = ROOT / "compile.py"
SUBMIT = ROOT / "submit.py"


def problems() -> list[Path]:
    return sorted(p for p in PROXY.iterdir() if p.is_dir() and (p / "meta.json").exists())


def compile_one(pdir: Path) -> tuple[bool, str]:
    meta = json.loads((pdir / "meta.json").read_text())
    logs = []
    for k in meta["kernels"]:
        ir = pdir / f"{k['entry']}.ir.json"
        r = subprocess.run(
            [sys.executable, str(COMPILER), str(ir)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            return False, f"compile fail: {(r.stderr or r.stdout)[:400]}"
        if not r.stdout.lstrip().startswith("."):
            return False, f"not PTX: {r.stdout[:120]!r}"
        logs.append(f"{k['entry']} {len(r.stdout)}B")
    return True, "; ".join(logs)


def submit_one(pdir: Path) -> tuple[bool, str]:
    r = subprocess.run(
        [sys.executable, str(SUBMIT), str(pdir), "--compiler", str(COMPILER)],
        capture_output=True, text=True,
    )
    text = (r.stdout or "") + (r.stderr or "")
    ok = r.returncode == 0 and "FAIL" not in text and "PASS" in text
    return ok, text.strip()[-500:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile-only", action="store_true")
    args = ap.parse_args()
    rows = []
    ids = problems()
    if not ids:
        print("no proxy problems — run tools/make_proxy_set.py first")
        return 2
    for pdir in ids:
        ok, msg = compile_one(pdir)
        status = "COMPILE_OK" if ok else "COMPILE_FAIL"
        if ok and not args.compile_only:
            ok, msg = submit_one(pdir)
            status = "PASS" if ok else "FAIL"
        print(f"{status:<13} {pdir.name}  {msg.splitlines()[-1] if msg else ''}")
        rows.append((pdir.name, status, msg))
    n_pass = sum(1 for _, s, _ in rows if s in ("PASS", "COMPILE_OK"))
    n_fail = len(rows) - n_pass
    print(f"\n{n_pass} ok, {n_fail} failed / {len(rows)} proxy problems")
    (PROXY / "RESULTS.json").write_text(json.dumps(
        [{"id": i, "status": s, "tail": m[-300:]} for i, s, m in rows], indent=2
    ))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
