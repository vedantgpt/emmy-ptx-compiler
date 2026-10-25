#!/usr/bin/env python3
"""Emmy PTX compiler studio.

    python tools/studio.py

Opens a local UI at http://127.0.0.1:8765 that walks Kernel IR → PTX,
shows GPU mapping, and can hit the hosted RTX 5090 runner.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).with_name("studio.html")
HOST = "127.0.0.1"
PORT = int(os.environ.get("EMMY_STUDIO_PORT", "8765"))


def _load_compiler():
    spec = importlib.util.spec_from_file_location("emmy_compile", ROOT / "compile.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["emmy_compile"] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load_compiler()


def problem_dirs() -> list[dict]:
    out = []
    problems = ROOT / "problems"
    if not problems.is_dir():
        return out
    for d in sorted(problems.iterdir()):
        if not d.is_dir():
            continue
        irs = list(d.glob("*.ir.json"))
        if not irs:
            continue
        meta = {}
        mp = d / "meta.json"
        if mp.is_file():
            meta = json.loads(mp.read_text(encoding="utf-8"))
        k = (meta.get("kernels") or [{}])[0]
        out.append(
            {
                "id": d.name,
                "ir": str(irs[0].relative_to(ROOT)).replace("\\", "/"),
                "entry": k.get("entry"),
                "grid": k.get("grid"),
                "block": k.get("block"),
                "smem_bytes": k.get("smem_bytes", 0),
                "code": meta.get("code", ""),
            }
        )
    return out


def ptxas_path() -> str:
    env = os.environ.get("PTXAS")
    if env:
        return env
    cuda = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4")
    cand = Path(cuda) / "bin" / ("ptxas.exe" if os.name == "nt" else "ptxas")
    if cand.is_file():
        return str(cand)
    found = shutil.which("ptxas")
    return found or "ptxas"


def ptxas_info(ptx: str, arch: str = "sm_89") -> dict:
    arch = arch if arch.startswith("sm_") else "sm_89"
    ptxas = ptxas_path()
    try:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "k.ptx"
            cubin = Path(td) / "k.cubin"
            src.write_text(ptx, encoding="ascii")
            r = subprocess.run(
                [ptxas, "-v", f"-arch={arch}", "-o", str(cubin), str(src)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            log = ((r.stderr or "") + (r.stdout or "")).strip()
            regs = None
            for line in log.splitlines():
                low = line.lower()
                if "register" in low:
                    parts = line.replace(",", " ").split()
                    for i, p in enumerate(parts):
                        if p.isdigit() and i + 1 < len(parts) and "register" in parts[i + 1].lower():
                            regs = int(p)
                            break
            errors = [ln for ln in log.splitlines() if "error" in ln.lower()]
            warnings = [ln for ln in log.splitlines() if "warning" in ln.lower()]
            return {
                "ok": r.returncode == 0,
                "log": log,
                "registers": regs,
                "errors": errors,
                "warnings": warnings,
                "arch": arch,
                "path": ptxas,
            }
    except Exception as exc:
        return {
            "ok": False,
            "log": str(exc),
            "registers": None,
            "errors": [str(exc)],
            "warnings": [],
            "arch": arch,
            "path": ptxas,
        }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("studio: " + (fmt % args) + "\n")

    def _send(self, code: int, body, ctype: str = "application/json"):
        data = body if isinstance(body, (bytes, bytearray)) else (
            json.dumps(body).encode("utf-8") if ctype.endswith("json") else str(body).encode("utf-8")
        )
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._send(200, STATIC.read_bytes(), "text/html; charset=utf-8")
            return
        if u.path == "/api/problems":
            self._send(200, {"problems": problem_dirs()})
            return
        if u.path == "/api/status":
            self._send(
                200,
                {
                    "ptxas": ptxas_path(),
                    "cuda_path": os.environ.get("CUDA_PATH") or "",
                    "ptx_target": "sm_89",
                    "ptx_version": "8.0",
                    "runner": "http://runner.riftstack.ai:48701",
                    "contract": "python compile.py path/to/kernel.ir.json",
                },
            )
            return
        if u.path == "/api/compile":
            q = parse_qs(u.query)
            pid = (q.get("id") or [None])[0]
            arch = (q.get("arch") or ["sm_89"])[0]
            problems = {p["id"]: p for p in problem_dirs()}
            if pid not in problems:
                self._send(404, {"error": "unknown problem"})
                return
            ir = ROOT / problems[pid]["ir"]
            t0 = time.perf_counter()
            try:
                bundle = C.compile_bundle(ir)
            except Exception as exc:
                self._send(
                    500,
                    {
                        "error": str(exc),
                        "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                        "problem": problems[pid],
                    },
                )
                return
            bundle["elapsed_ms"] = round((time.perf_counter() - t0) * 1000)
            bundle["ptxas"] = ptxas_info(bundle["ptx"], arch)
            bundle["problem"] = problems[pid]
            self._send(200, bundle)
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/submit":
            self._send(404, {"error": "not found"})
            return
        q = parse_qs(u.query)
        pid = (q.get("id") or [None])[0]
        if not pid:
            self._send(400, {"error": "missing id"})
            return
        py = ROOT / ".venv" / "Scripts" / "python.exe"
        exe = str(py) if py.is_file() else sys.executable
        try:
            r = subprocess.run(
                [exe, str(ROOT / "submit.py"), pid, "--compiler", str(ROOT / "compile.py")],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=120,
            )
            self._send(200, {"code": r.returncode, "stdout": r.stdout, "stderr": r.stderr})
        except Exception as exc:
            self._send(500, {"error": str(exc)})


def main() -> int:
    if not STATIC.is_file():
        print("missing tools/studio.html", file=sys.stderr)
        return 1
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}/"
    print(f"compiler studio on {url}", flush=True)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
