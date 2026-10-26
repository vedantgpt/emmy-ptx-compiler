#!/usr/bin/env python3
"""Compile every Kernel IR we have, then score new world problems on the runner.

    .venv\\Scripts\\python.exe tools/run_all_irs.py
    .venv\\Scripts\\python.exe tools/run_all_irs.py --compile-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPILER = ROOT / "compile.py"
SUBMIT = ROOT / "submit.py"


def all_irs() -> list[Path]:
    found = []
    for folder in (ROOT / "problems", ROOT / "problems_proxy", ROOT / "problems_world"):
        if not folder.is_dir():
            continue
        found.extend(sorted(folder.glob("*/*.ir.json")))
    return found


def compile_ir(ir: Path) -> tuple[bool, str]:
    r = subprocess.run([sys.executable, str(COMPILER), str(ir)], capture_output=True, text=True)
    if r.returncode != 0:
        return False, (r.stderr or r.stdout)[-300:]
    if not r.stdout.lstrip().startswith("."):
        return False, "not PTX"
    return True, f"{len(r.stdout)}B"


def extra_isa() -> tuple[bool, str]:
    r = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "extra_isa.py")],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return False, (r.stderr or r.stdout)[-300:]
    ptx = r.stdout
    need = ("ldmatrix.sync.aligned", "mma.sync.aligned", "cp.async.bulk.tensor", "%tid.y")
    miss = [n for n in need if n not in ptx]
    return (not miss), ("ok " + str(len(ptx)) if not miss else f"missing {miss}")


def submit(pdir: Path) -> tuple[bool, str]:
    r = subprocess.run(
        [sys.executable, str(SUBMIT), str(pdir), "--compiler", str(COMPILER)],
        capture_output=True, text=True,
    )
    text = ((r.stdout or "") + (r.stderr or "")).strip()
    return r.returncode == 0 and "PASS" in text and "FAIL" not in text, text[-400:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--submit", choices=["world", "all", "none"], default="world")
    args = ap.parse_args()
    rows = []
    fail = 0
    for ir in all_irs():
        ok, msg = compile_ir(ir)
        status = "COMPILE_OK" if ok else "COMPILE_FAIL"
        if not ok:
            fail += 1
        print(f"{status:<13} {ir.parent.name}/{ir.name}  {msg.splitlines()[-1] if msg else ''}")
        rows.append((str(ir), status, msg))
    ok, msg = extra_isa()
    status = "COMPILE_OK" if ok else "COMPILE_FAIL"
    if not ok:
        fail += 1
    print(f"{status:<13} examples/extra_isa.py  {msg}")
    rows.append(("extra_isa", status, msg))

    if not args.compile_only and args.submit != "none":
        folders = [ROOT / "problems_world"]
        if args.submit == "all":
            folders = [ROOT / "problems", ROOT / "problems_proxy", ROOT / "problems_world"]
        for folder in folders:
            if not folder.is_dir():
                continue
            for pdir in sorted(p for p in folder.iterdir() if p.is_dir() and (p / "meta.json").exists()):
                ok, msg = submit(pdir)
                status = "PASS" if ok else "FAIL"
                if not ok:
                    fail += 1
                print(f"{status:<13} {pdir.name}  {(msg.splitlines()[-1] if msg else '')}")
                rows.append((pdir.name, status, msg))

    print(f"\n{len(rows) - fail} ok, {fail} failed / {len(rows)} checks")
    out = ROOT / "problems_world"
    out.mkdir(exist_ok=True)
    (out / "RESULTS.json").write_text(json.dumps(
        [{"id": i, "status": s, "tail": m[-200:]} for i, s, m in rows], indent=2
    ))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
