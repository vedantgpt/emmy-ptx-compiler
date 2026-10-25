#!/usr/bin/env python3
"""Kernel IR -> PTX compiler.

Contract:  python compile.py path/to/kernel.ir.json > out.ptx

Pipeline (every kernel, no per-problem special cases):

  1. Load the IR with Graph.from_dict. The entry name is KernelOp.name.
  2. ABI comes from sibling meta.json arg_order (not the node's inputs list).
  3. Prologue: param loads, cvta.to.global, tid/cta/gid, optional Tile N-guard
     when the launch has more threads than cells (or aux_threads).
  4. Walk statements in order. Map SSA names to virtual PTX registers.
     ptxas allocates the physical registers.
  5. Local peepholes that do not change results: FMA (mul+add / mul+accum),
     CSE of index expressions, consecutive-loop fusion on the same axis,
     IR-flagged or straight-line unroll bounded by UNROLL_EXPANSION_CAP,
     pairing consecutive f32 loads into ld.v2/v4, address [ptr+imm].

Shared memory larger than the 48 KB static cap is emitted as
`.extern .shared` so launch smem_bytes from meta.json can size it.
"""

from __future__ import annotations

import json
import math
import re
import struct
import sys
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import (
    BinaryExpr,
    Builtin,
    CastExpr,
    FuncCallExpr,
    Literal,
    SimplifyCtx,
    TernaryExpr,
    Var,
)
from emmy.compiler.ir.kernel.ir import (
    CpAsyncCommit,
    CpAsyncCopy,
    CpAsyncWait,
    KernelOp,
    Smem,
    Sync,
    Tile,
    TreeHalve,
    WarpShuffle,
)
from emmy.compiler.ir.stmt import (
    Accum,
    Assign,
    Cond,
    Init,
    Let,
    Load,
    Loop,
    StridedLoop,
    Write,
)

PTX_VERSION = "8.0"
PTX_TARGET = "sm_89"
# sm_89 static shared cap (PTX ISA), not a problem size.
SMEM_STATIC_LIMIT = 48 * 1024
# Bound fully-expanded innermost loops so ptxas / I-cache stay healthy.
# Derived from body size × trip count, never from a named problem.
UNROLL_EXPANSION_CAP = 1536

DTYPE_NBYTES = {
    "f32": 4, "float": 4, "float32": 4,
    "f16": 2, "half": 2, "float16": 2, "__half": 2,
    "bf16": 2,
    "f64": 8, "double": 8,
    "i32": 4, "int": 4, "int32": 4, "s32": 4,
    "i64": 8, "int64": 8, "s64": 8,
    "u32": 4, "u64": 8,
    "u8": 1, "i8": 1,
}

PTX_LDST = {
    "f32": "f32", "float": "f32", "float32": "f32",
    "f16": "f16", "half": "f16",
    "f64": "f64",
    "i32": "s32", "int": "s32", "int32": "s32", "s32": "s32",
    "i64": "s64",
    "u32": "u32", "u64": "u64",
}

F32_OPS = {
    "add": "add.f32",
    "sum": "add.f32",
    "subtract": "sub.f32",
    "sub": "sub.f32",
    "multiply": "mul.f32",
    "mul": "mul.f32",
    "divide": "div.rn.f32",
    "true_divide": "div.rn.f32",
    "div": "div.rn.f32",
    "maximum": "max.f32",
    "amax": "max.f32",
    "max": "max.f32",
    "minimum": "min.f32",
    "amin": "min.f32",
    "min": "min.f32",
}
S32_OPS = {
    "add": "add.s32",
    "sum": "add.s32",
    "subtract": "sub.s32",
    "sub": "sub.s32",
    "multiply": "mul.lo.s32",
    "mul": "mul.lo.s32",
    "maximum": "max.s32",
    "max": "max.s32",
    "minimum": "min.s32",
    "min": "min.s32",
}
CMP_ASSIGN = {
    "gt": "gt", "greater": "gt",
    "lt": "lt", "less": "lt",
    "ge": "ge", "greater_equal": "ge",
    "le": "le", "less_equal": "le",
    "eq": "eq", "equal": "eq",
    "ne": "ne", "not_equal": "ne",
}
INT_DTYPES = {"i32", "int", "int32", "s32", "u32"}

CMP_OPS = {"<", "<=", ">", ">=", "==", "!=", "&&", "||"}


def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def ilog2(n: int) -> int:
    return n.bit_length() - 1


_IMM_RE = re.compile(r"^-?\d+$")


def parse_imm(x) -> int | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return int(x)
    if isinstance(x, str) and _IMM_RE.fullmatch(x):
        return int(x)
    return None


def lit_int(expr) -> int | None:
    if isinstance(expr, Literal) and not isinstance(expr.value, bool):
        if expr.dtype in ("float", "f32"):
            return None
        if isinstance(expr.value, int) or (isinstance(expr.value, float) and expr.value == int(expr.value)):
            return int(expr.value)
    return None


def f32_hex(value: float) -> str:
    bits = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    return f"0f{bits:08X}"


def dim_int(dim) -> int:
    if isinstance(dim, int):
        return int(dim)
    if hasattr(dim, "as_static"):
        try:
            return int(dim.as_static())
        except Exception:
            pass
    expr = getattr(dim, "expr", dim)
    if isinstance(expr, Literal):
        return int(expr.value)
    value = getattr(expr, "value", None)
    if value is not None:
        return int(value)
    return int(dim)


def axis_extent(axis) -> int:
    return dim_int(axis.extent)


def dtype_name(dt) -> str:
    if dt is None:
        return "f32"
    if isinstance(dt, str):
        return dt
    return getattr(dt, "name", None) or str(dt)


def ptx_type(dt: str) -> str:
    return PTX_LDST.get(dt, "f32")


def nbytes_of(dt: str) -> int:
    return DTYPE_NBYTES.get(dt, 4)


def body_stmts(body):
    if body is None:
        return ()
    return tuple(body)


def walk_stmts(body):
    for s in body_stmts(body):
        yield s
        for child in getattr(s, "nested", lambda: ())():
            yield from walk_stmts(child)


def collect_smem(body) -> list[Smem]:
    seen: dict[str, Smem] = {}
    for s in walk_stmts(body):
        if isinstance(s, Smem) and s.name not in seen:
            seen[s.name] = s
    return list(seen.values())


def tensor_shape(tensor) -> tuple[int, ...]:
    shape = getattr(tensor, "shape", ()) or ()
    return tuple(dim_int(d) for d in shape)


def load_shapes(raw: dict, kop: KernelOp, meta_kernel: dict | None) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, str] = {}
    for node in raw.get("nodes", {}).values():
        for out in node.get("outputs") or []:
            name = out.get("name")
            if name and out.get("shape") is not None:
                shapes[name] = tuple(int(x) for x in out["shape"])
                dtypes[name] = out.get("dtype") or "f32"
    for mapping in (kop.inputs, kop.outputs):
        for name, tensor in mapping.items():
            if name not in shapes:
                sh = tensor_shape(tensor)
                if sh:
                    shapes[name] = sh
            dt = dtype_name(getattr(tensor, "dtype", None))
            dtypes.setdefault(name, dt)
    if meta_kernel:
        for arg in meta_kernel.get("args") or []:
            name = arg.get("name")
            if name and arg.get("shape") is not None:
                shapes.setdefault(name, tuple(int(x) for x in arg["shape"]))
                dtypes.setdefault(name, arg.get("dtype") or "f32")
    for sm in collect_smem(kop.body):
        shapes[sm.name] = tuple(int(e) for e in sm.extents)
        dtypes[sm.name] = "f32" if sm.dtype in ("float", "float32", "") else sm.dtype
    return shapes, dtypes


def load_meta_kernel(ir_path: Path, entry: str) -> dict | None:
    meta_path = ir_path.parent / "meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for k in meta.get("kernels") or []:
        if k.get("entry") == entry:
            return k
    kernels = meta.get("kernels") or []
    return kernels[0] if len(kernels) == 1 else None


def arg_order_for(kop: KernelOp, meta_kernel: dict | None) -> list[str]:
    if meta_kernel and meta_kernel.get("arg_order"):
        return list(meta_kernel["arg_order"])
    return list(kop.inputs) + list(kop.outputs)


def find_kernel(graph: Graph) -> KernelOp:
    for node in graph.nodes.values():
        if isinstance(node.op, KernelOp):
            return node.op
    raise SystemExit("no KernelOp in IR")


def identity_of(op) -> float:
    name = op if isinstance(op, str) else getattr(op, "name", "add")
    table = {
        "add": 0.0, "sum": 0.0,
        "multiply": 1.0, "prod": 1.0,
        "maximum": -1e30, "amax": -1e30, "max": -1e30,
        "minimum": 1e30, "min": 1e30,
    }
    ident = getattr(op, "identity", None) if not isinstance(op, str) else None
    if ident is not None:
        return float(ident)
    return table.get(name, 0.0)


class Emitter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.nf = 1
        self.nr = 1
        self.nrd = 1
        self.np = 1
        self.nl = 1
        self.env: dict[str, str] = {}
        self.kind: dict[str, str] = {}
        self.on_bind = None

    def emit(self, line: str = "") -> None:
        self.lines.append(line)

    def new_f(self) -> str:
        r = f"%f{self.nf}"
        self.nf += 1
        return r

    def new_r(self) -> str:
        r = f"%r{self.nr}"
        self.nr += 1
        return r

    def new_rd(self) -> str:
        r = f"%rd{self.nrd}"
        self.nrd += 1
        return r

    def new_p(self) -> str:
        r = f"%p{self.np}"
        self.np += 1
        return r

    def label(self, prefix: str = "L") -> str:
        name = f"{prefix}_{self.nl}"
        self.nl += 1
        return name

    def bind(self, name: str, reg: str, kind: str) -> str:
        if self.on_bind is not None:
            self.on_bind(name)
        self.env[name] = reg
        self.kind[name] = kind
        return reg

    def imm_s32(self, value: int) -> str:
        return str(int(value))

    def mov_s32(self, value) -> str:
        r = self.new_r()
        self.emit(f"        mov.s32         {r}, {value};")
        return r

    def imm_f32(self, value: float) -> str:
        r = self.new_f()
        self.emit(f"        mov.f32         {r}, {f32_hex(value)};")
        return r


class Compiler:
    def __init__(self, kop: KernelOp, shapes: dict, dtypes: dict, args: list[str], meta: dict | None = None):
        self.kop = kop
        self.shapes = shapes
        self.dtypes = dtypes
        self.args = args
        self.meta = meta or {}
        self.e = Emitter()
        self.ptrs: dict[str, str] = {}
        self.smem: dict[str, Smem] = {s.name: s for s in collect_smem(kop.body)}
        self.smem_base: dict[str, str] = {}
        self.smem_offsets: dict[str, int] = {}
        self.smem_total = 0
        self.use_dynamic_smem = False
        self.cse: dict[str, tuple[str, str]] = {}
        self.mem: dict[str, tuple[str, ...]] = {}
        self.e.on_bind = self.invalidate
        self.stats: dict[str, int] = {
            "fma": 0,
            "vec_loads": 0,
            "cse_hits": 0,
            "unrolled_iters": 0,
            "fused_loops": 0,
            "cp_async": 0,
            "warp_shuffles": 0,
            "tree_halves": 0,
        }

    @staticmethod
    def _pack_smem(smems: list[Smem]) -> tuple[dict[str, int], int]:
        cursor = 0
        offsets: dict[str, int] = {}
        for sm in smems:
            total = 1
            for ext in sm.extents:
                total *= int(ext)
            nb = nbytes_of("f32" if sm.dtype in ("float", "float32", "") else sm.dtype)
            align = max(int(sm.align or 0), nb, 4)
            if align:
                cursor = (cursor + align - 1) // align * align
            offsets[sm.name] = cursor
            cursor += total * nb
        return offsets, cursor

    def run(self) -> str:
        e = self.e
        smems = collect_smem(self.kop.body)
        self.smem_offsets, self.smem_total = self._pack_smem(smems)
        self.use_dynamic_smem = self.smem_total > SMEM_STATIC_LIMIT
        e.emit(f".version {PTX_VERSION}")
        e.emit(f".target {PTX_TARGET}")
        e.emit(".address_size 64")
        e.emit("")
        if self.use_dynamic_smem and smems:
            e.emit(".extern .shared .align 16 .b8 _emmy_smem[];")
            e.emit("")
        else:
            for sm in smems:
                total = 1
                for ext in sm.extents:
                    total *= int(ext)
                nb = nbytes_of("f32" if sm.dtype in ("float", "float32", "") else sm.dtype)
                align = max(int(sm.align or 0), 4)
                e.emit(f".shared .align {align} .b8 {sm.name}[{total * nb}];")
            if smems:
                e.emit("")
        params = ",\n".join(f"        .param .u64 param_{name}" for name in self.args)
        e.emit(f".visible .entry {self.kop.name}(")
        e.emit(params)
        e.emit(")")
        bt = self._launch_block()
        if bt:
            e.emit(f".reqntid {bt}")
        e.emit("{")
        body_at = len(e.lines)
        self.prologue()
        self.stmts(self.kop.body)
        e.emit("        ret;")
        e.emit("}")
        decls = [
            f"        .reg .pred      %p<{max(self.e.np, 2)}>;",
            f"        .reg .f32       %f<{max(self.e.nf, 2)}>;",
            f"        .reg .b32       %r<{max(self.e.nr, 2)}>;",
            f"        .reg .b64       %rd<{max(self.e.nrd, 2)}>;",
            "",
        ]
        e.lines[body_at:body_at] = decls
        return "\n".join(e.lines) + "\n"

    def _launch_block(self) -> int | None:
        for s in walk_stmts(self.kop.body):
            if isinstance(s, Tile) and s.block_threads:
                return int(s.block_threads)
        block = self.meta.get("block")
        if block:
            return int(block[0])
        return None

    def prologue(self) -> None:
        e = self.e
        for name in self.args:
            raw = e.new_rd()
            ptr = e.new_rd()
            e.emit(f"        ld.param.u64    {raw}, [param_{name}];")
            if name in self.smem:
                e.emit(f"        cvta.to.shared.u64     {ptr}, {raw};")
            else:
                e.emit(f"        cvta.to.global.u64     {ptr}, {raw};")
            self.ptrs[name] = ptr
        if self.use_dynamic_smem and self.smem:
            pool = e.new_r()
            e.emit(f"        mov.u32         {pool}, _emmy_smem;")
            for sm in self.smem.values():
                base32 = e.new_r()
                off = int(self.smem_offsets.get(sm.name, 0))
                if off:
                    e.emit(f"        add.u32         {base32}, {pool}, {off};")
                else:
                    e.emit(f"        mov.u32         {base32}, {pool};")
                self.smem_base[sm.name] = base32
        else:
            for sm in self.smem.values():
                base32 = e.new_r()
                e.emit(f"        mov.u32         {base32}, {sm.name};")
                self.smem_base[sm.name] = base32
        tid = e.new_r()
        ntid = e.new_r()
        cta = e.new_r()
        gid = e.new_r()
        e.emit(f"        mov.u32         {tid}, %tid.x;")
        e.emit(f"        mov.u32         {ntid}, %ntid.x;")
        e.emit(f"        mov.u32         {cta}, %ctaid.x;")
        e.emit(f"        mad.lo.s32      {gid}, {cta}, {ntid}, {tid};")
        e.bind("_tid", tid, "s32")
        e.bind("_ntid", ntid, "s32")
        e.bind("_cta", cta, "s32")
        e.bind("_gid", gid, "s32")
        e.bind("threadIdx.x", tid, "s32")
        e.bind("blockDim.x", ntid, "s32")
        e.bind("blockIdx.x", cta, "s32")
        lane = e.new_r()
        warp = e.new_r()
        e.emit(f"        and.b32         {lane}, {tid}, 31;")
        e.emit(f"        shr.u32         {warp}, {tid}, 5;")
        e.bind("lane", lane, "s32")
        e.bind("warp", warp, "s32")

    def invalidate(self, name: str) -> None:
        pat = re.compile(rf"(^|[^A-Za-z0-9_]){re.escape(name)}([^A-Za-z0-9_]|$)")
        if self.cse:
            self.cse = {k: v for k, v in self.cse.items() if not pat.search(k)}
        if self.mem:
            self.mem = {k: v for k, v in self.mem.items() if not pat.search(k)}

    def _push_flow(self) -> tuple[dict, dict]:
        return dict(self.cse), dict(self.mem)

    def _pop_flow(self, saved: tuple[dict, dict]) -> None:
        self.cse, self.mem = saved

    def _kill_mem(self, buf: str) -> None:
        prefix = buf + "["
        self.mem = {k: v for k, v in self.mem.items() if not k.startswith(prefix)}

    def stmts(self, body) -> None:
        sl = list(body_stmts(body))
        done: set[int] = set()
        i = 0
        while i < len(sl):
            s = sl[i]
            if id(s) in done:
                i += 1
                continue
            nxt = sl[i + 1] if i + 1 < len(sl) else None
            if isinstance(s, Loop):
                fused = list(body_stmts(s.body))
                j = i + 1
                while j < len(sl) and self._can_fuse_loops(s, sl[j]):
                    fused.extend(body_stmts(sl[j].body))
                    j += 1
                if j > i + 1:
                    self.stats["fused_loops"] += j - i
                    self.emit_loop(s, body_override=fused, seed_any=True)
                    i = j
                    continue
            if isinstance(s, Load) and self._try_vec_load(s, sl[i + 1 :], done):
                i += 1
                continue
            if self._try_fma(s, nxt, sl[i + 2 :]):
                i += 2
                continue
            if self._try_fma_assign(s, nxt, sl[i + 2 :]):
                i += 2
                continue
            self.stmt(s)
            i += 1

    def _consecutive_index(self, i0, i1) -> bool:
        if not i0 or not i1 or len(i0) != len(i1):
            return False
        for a, b in zip(i0[:-1], i1[:-1]):
            if a.pretty() != b.pretty():
                return False
        a, b = i0[-1], i1[-1]
        la, lb = lit_int(a), lit_int(b)
        if la is not None and lb is not None:
            return lb == la + 1
        if isinstance(b, BinaryExpr) and b.op == "+" and lit_int(b.right) == 1 and b.left.pretty() == a.pretty():
            return True
        if (
            isinstance(a, BinaryExpr)
            and a.op == "+"
            and lit_int(a.right) == 0
            and isinstance(b, BinaryExpr)
            and b.op == "+"
            and lit_int(b.right) == 1
            and a.left.pretty() == b.left.pretty()
        ):
            return True
        return False

    def _try_vec_load(self, s: Load, rest, done: set[int]) -> bool:
        if len(list(s.names)) != 1:
            return False
        dt = dtype_name(s.dtype) or self.dtypes.get(s.input, "f32")
        if dt not in ("f32", "float", "float32"):
            return False
        chain = [s]
        last = s
        for t in rest:
            if isinstance(t, (Sync, CpAsyncCopy, Loop, StridedLoop, Cond, TreeHalve)):
                break
            if isinstance(t, Write):
                if t.output == s.input:
                    break
                continue
            if not isinstance(t, Load) or id(t) in done:
                continue
            if t.input != s.input or len(list(t.names)) != 1:
                continue
            if self._consecutive_index(last.index, t.index):
                chain.append(t)
                last = t
                if len(chain) == 4:
                    break
            elif t.input == s.input:
                break
        vec = 4 if len(chain) >= 4 else 2 if len(chain) >= 2 else 0
        if vec == 0:
            return False
        chain = chain[:vec]
        space = "shared" if s.input in self.smem else "global"
        addr = self.addr_of(s.input, s.index, space)
        regs = [self.e.new_f() for _ in chain]
        packed = ", ".join(regs)
        self.e.emit(f"        ld.{space}.v{vec}.f32 {{{packed}}}, [{addr}];")
        for load, reg in zip(chain, regs):
            name = list(load.names)[0]
            self.e.bind(name, reg, "f32")
            idx = ", ".join(ix.pretty() for ix in load.index)
            self.mem[f"{load.input}[{idx}]|1"] = (reg,)
            done.add(id(load))
        self.stats["vec_loads"] += 1
        return True

    def _can_fuse_loops(self, a, b) -> bool:
        if not isinstance(b, Loop):
            return False
        if a.axis.name != b.axis.name:
            return False
        if axis_extent(a.axis) != axis_extent(b.axis):
            return False
        if self._has_nested_loop(a.body) or self._has_nested_loop(b.body):
            return False
        return True

    def _try_fma(self, s, nxt, rest) -> bool:
        if not isinstance(s, Assign) or not isinstance(nxt, Accum):
            return False
        op = s.op.name if hasattr(s.op, "name") else str(s.op)
        acc_op = nxt.op.name if hasattr(nxt.op, "name") else str(nxt.op)
        if op != "multiply" or acc_op not in ("add", "sum") or len(s.args) != 2:
            return False
        if nxt.value != s.name or s.name == nxt.name:
            return False
        if nxt.base not in (None, nxt.name):
            return False
        if any(self._uses_name(t, s.name) for t in rest):
            return False
        if nxt.name not in self.e.env:
            self.e.bind(nxt.name, self.e.imm_f32(identity_of(nxt.op)), "f32")
        dest = self.e.env[nxt.name]
        a, b = self.as_f32(s.args[0]), self.as_f32(s.args[1])
        self.e.emit(f"        fma.rn.f32      {dest}, {a}, {b}, {dest};")
        self.e.bind(nxt.name, dest, "f32")
        self.stats["fma"] += 1
        return True

    def _try_fma_assign(self, s, nxt, rest) -> bool:
        if not isinstance(s, Assign) or not isinstance(nxt, Assign):
            return False
        op = s.op.name if hasattr(s.op, "name") else str(s.op)
        nop = nxt.op.name if hasattr(nxt.op, "name") else str(nxt.op)
        if op not in ("multiply", "mul") or nop not in ("add", "sum") or len(s.args) != 2 or len(nxt.args) != 2:
            return False
        if s.name not in nxt.args or s.name == nxt.name:
            return False
        if any(self._uses_name(t, s.name) for t in rest):
            return False
        other = nxt.args[0] if nxt.args[1] == s.name else nxt.args[1]
        dest = self.e.env.get(nxt.name) if self.e.kind.get(nxt.name) == "f32" else self.e.new_f()
        a, b = self.as_f32(s.args[0]), self.as_f32(s.args[1])
        c = self.as_f32(other)
        self.e.emit(f"        fma.rn.f32      {dest}, {a}, {b}, {c};")
        self.e.bind(nxt.name, dest, "f32")
        self.stats["fma"] += 1
        return True

    def _has_nested_loop(self, body) -> bool:
        for s in body_stmts(body):
            if isinstance(s, (Loop, StridedLoop)):
                return True
            for child in getattr(s, "nested", lambda: ())():
                if self._has_nested_loop(child):
                    return True
        return False

    def _stmt_count(self, body) -> int:
        n = 0
        for s in body_stmts(body):
            n += 1
            for child in getattr(s, "nested", lambda: ())():
                n += self._stmt_count(child)
        return n

    def _body_is_straight(self, body) -> bool:
        for s in body_stmts(body):
            if isinstance(s, (Loop, StridedLoop, Cond, TreeHalve, WarpShuffle, Sync)):
                return False
        return True

    def _should_unroll(self, loop, end: int, body) -> bool:
        if end <= 0 or self._has_nested_loop(body):
            return False
        ops = max(self._stmt_count(body), 1)
        if ops * end > UNROLL_EXPANSION_CAP:
            return False
        if getattr(loop, "unroll", False):
            return True
        return self._body_is_straight(body)

    def _uses_name(self, stmt, name: str) -> bool:
        if hasattr(stmt, "deps") and name in stmt.deps():
            return True
        if isinstance(stmt, Write) and name in stmt.values:
            return True
        if isinstance(stmt, Assign) and name in stmt.args:
            return True
        if isinstance(stmt, Accum) and (stmt.value == name or stmt.base == name):
            return True
        for child in getattr(stmt, "nested", lambda: ())():
            for t in child:
                if self._uses_name(t, name):
                    return True
        return False

    def stmt(self, s) -> None:
        if isinstance(s, Tile):
            self.emit_tile(s)
        elif isinstance(s, Smem):
            return
        elif isinstance(s, Loop):
            self.emit_loop(s)
        elif isinstance(s, StridedLoop):
            self.emit_strided(s)
        elif isinstance(s, Cond):
            self.emit_cond(s)
        elif isinstance(s, Load):
            self.emit_load(s)
        elif isinstance(s, Write):
            self.emit_write(s)
        elif isinstance(s, Assign):
            self.emit_assign(s)
        elif isinstance(s, Accum):
            self.emit_accum(s)
        elif isinstance(s, Init):
            self.emit_init(s)
        elif isinstance(s, Let):
            self.emit_let(s)
        elif isinstance(s, WarpShuffle):
            self.emit_warpshuffle(s)
        elif isinstance(s, TreeHalve):
            self.emit_treehalve(s)
        elif isinstance(s, Sync):
            self.emit_sync(s)
        elif isinstance(s, CpAsyncCopy):
            self.emit_cp_async(s)
        elif isinstance(s, CpAsyncCommit):
            self.e.emit("        cp.async.commit_group;")
        elif isinstance(s, CpAsyncWait):
            self.e.emit(f"        cp.async.wait_group {int(s.group)};")
        else:
            print(f"unsupported stmt {type(s).__name__}", file=sys.stderr)
            raise SystemExit(2)

    def emit_tile(self, tile: Tile) -> None:
        e = self.e
        extents = [axis_extent(a) for a in tile.axes]
        n = 1
        for ext in extents:
            n *= ext
        cells = int(tile.block_threads) if tile.block_threads is not None else self._launch_block()
        aux = int(tile.aux_threads or 0)
        gid = e.env["_gid"]
        if aux:
            bt = int(tile.block_threads or cells or 0)
            tid = e.env["_tid"]
            cta = e.env["_cta"]
            rem = e.new_r()
            e.emit(f"        rem.s32         {rem}, {tid}, {bt};")
            gid = e.new_r()
            e.emit(f"        mad.lo.s32      {gid}, {cta}, {bt}, {rem};")
            e.bind("_gid", gid, "s32")
        guard = aux != 0 or not cells or n % int(cells) != 0
        skip = None
        saved = None
        if guard:
            skip = e.label("SKIP")
            p = e.new_p()
            e.emit(f"        setp.ge.s32     {p}, {gid}, {n};")
            e.emit(f"        @{p} bra        {skip};")
            saved = self._push_flow()
        strides = [1] * len(extents)
        acc = 1
        for i in range(len(extents) - 1, -1, -1):
            strides[i] = acc
            acc *= extents[i]
        raster = None
        if tile.raster_group and tile.raster_axes:
            names = [a.name for a in tile.axes]
            if tile.raster_axes[0] in names and tile.raster_axes[1] in names:
                mi = names.index(tile.raster_axes[0])
                ni = names.index(tile.raster_axes[1])
                if ni == mi + 1:
                    gi, oi = (mi, ni) if tile.raster_orient == "m" else (ni, mi)
                    g_eff = min(int(tile.raster_group), extents[gi])
                    if extents[gi] > 1 and g_eff > 1:
                        raster = (gi, oi, g_eff, mi, ni)
        decls_done = False
        a_reg = b_reg = None
        for i, axis in enumerate(tile.axes):
            s, ext = strides[i], extents[i]
            if raster is not None:
                gi, oi, g, mi, ni = raster
                if i == mi:
                    if not decls_done:
                        ea, eb = extents[gi], extents[oi]
                        sn = strides[ni]
                        sub = gid if sn == 1 else self.div_imm(gid, sn)
                        if mi != 0:
                            sub = self.rem_imm(sub, ea * eb)
                        rsub = sub
                        if ea % g == 0:
                            # a = (sub / (g*eb))*g + sub % g
                            # b = (sub % (g*eb)) / g
                            geb = g * eb
                            q = self.div_imm(rsub, geb)
                            a_reg = self.mad_imm(q, g, self.rem_imm(rsub, g))
                            b_reg = self.div_imm(self.rem_imm(rsub, geb), g)
                        else:
                            rgrp = self.div_imm(rsub, g * eb)
                            # rsz = min(g, ea - rgrp*g)
                            tmp = self.mul_imm(rgrp, g)
                            remain = self.sub_reg(e.imm_s32(ea), tmp)
                            rsz = e.new_r()
                            pmin = e.new_p()
                            e.emit(f"        setp.lt.s32     {pmin}, {remain}, {g};")
                            e.emit(f"        selp.s32        {rsz}, {remain}, {g}, {pmin};")
                            geb = g * eb
                            a_reg = self.mad_imm(rgrp, g, self.rem_reg(self.rem_imm(rsub, geb), rsz))
                            b_reg = self.div_reg(self.rem_imm(rsub, geb), rsz)
                        decls_done = True
                    e.bind(axis.name, a_reg if i == gi else b_reg, "s32")
                    continue
                if i == ni:
                    e.bind(axis.name, a_reg if i == gi else b_reg, "s32")
                    continue
            term = gid if s == 1 else self.div_imm(gid, s)
            if i == 0:
                e.bind(axis.name, term, "s32")
            else:
                e.bind(axis.name, self.rem_imm(term, ext), "s32")
        self.stmts(tile.body)
        if skip:
            e.emit(f"{skip}:")
            if saved is not None:
                self._pop_flow(saved)

    def mul_imm(self, src: str, imm: int) -> str:
        si = parse_imm(src)
        if si is not None:
            return str(si * imm)
        if imm == 0:
            return "0"
        if imm == 1:
            return src
        if imm == -1:
            d = self.e.new_r()
            self.e.emit(f"        neg.s32         {d}, {src};")
            return d
        d = self.e.new_r()
        if is_pow2(abs(imm)):
            self.e.emit(f"        shl.b32         {d}, {src}, {ilog2(abs(imm))};")
            if imm < 0:
                n = self.e.new_r()
                self.e.emit(f"        neg.s32         {n}, {d};")
                return n
            return d
        self.e.emit(f"        mul.lo.s32      {d}, {src}, {imm};")
        return d

    def div_imm(self, src: str, imm: int) -> str:
        si = parse_imm(src)
        if si is not None and imm:
            return str(si // imm)
        if imm == 1:
            return src
        d = self.e.new_r()
        if is_pow2(imm):
            self.e.emit(f"        shr.u32         {d}, {src}, {ilog2(imm)};")
            return d
        self.e.emit(f"        div.s32         {d}, {src}, {imm};")
        return d

    def rem_imm(self, src: str, imm: int) -> str:
        si = parse_imm(src)
        if si is not None and imm:
            return str(si % imm)
        d = self.e.new_r()
        if is_pow2(imm):
            self.e.emit(f"        and.b32         {d}, {src}, {imm - 1};")
            return d
        self.e.emit(f"        rem.s32         {d}, {src}, {imm};")
        return d

    def rem_reg(self, src: str, den: str) -> str:
        si, di = parse_imm(src), parse_imm(den)
        if si is not None and di:
            return str(si % di)
        d = self.e.new_r()
        self.e.emit(f"        rem.s32         {d}, {src}, {den};")
        return d

    def div_reg(self, src: str, den: str) -> str:
        si, di = parse_imm(src), parse_imm(den)
        if si is not None and di:
            return str(si // di)
        d = self.e.new_r()
        self.e.emit(f"        div.s32         {d}, {src}, {den};")
        return d

    def sub_reg(self, a: str, b: str) -> str:
        ia, ib = parse_imm(a), parse_imm(b)
        if ia is not None and ib is not None:
            return str(ia - ib)
        if ib == 0:
            return a
        if ia == 0:
            d = self.e.new_r()
            self.e.emit(f"        neg.s32         {d}, {b};")
            return d
        d = self.e.new_r()
        self.e.emit(f"        sub.s32         {d}, {a}, {b};")
        return d

    def mad_imm(self, a: str, imm: int, c: str) -> str:
        ia, ic = parse_imm(a), parse_imm(c)
        if ia is not None and ic is not None:
            return str(ia * imm + ic)
        if ia is not None:
            return self.add_s32(c, str(ia * imm))
        if ic == 0:
            return self.mul_imm(a, imm)
        if imm == 1:
            return self.add_s32(a, c)
        d = self.e.new_r()
        self.e.emit(f"        mad.lo.s32      {d}, {a}, {imm}, {c};")
        return d

    def add_s32(self, a: str, b: str) -> str:
        ia, ib = parse_imm(a), parse_imm(b)
        if ia is not None and ib is not None:
            return str(ia + ib)
        if ia == 0:
            return b
        if ib == 0:
            return a
        d = self.e.new_r()
        if ib is not None:
            self.e.emit(f"        add.s32         {d}, {a}, {ib};")
        elif ia is not None:
            self.e.emit(f"        add.s32         {d}, {b}, {ia};")
        else:
            self.e.emit(f"        add.s32         {d}, {a}, {b};")
        return d

    def seed_accums(self, body, seed: bool) -> None:
        if not seed:
            return
        seen: set[str] = set()
        for s in body_stmts(body):
            if isinstance(s, Accum) and s.name not in seen:
                seen.add(s.name)
                if s.name in self.e.env:
                    continue
                dt = dtype_name(s.dtype)
                ident = identity_of(s.op)
                if dt in ("f32", "float", "float32", None, ""):
                    self.e.bind(s.name, self.e.imm_f32(ident), "f32")
                else:
                    self.e.bind(s.name, self.e.imm_s32(int(ident)), "s32")

    def emit_loop(self, loop: Loop, body_override=None, seed_any: bool = False) -> None:
        e = self.e
        body = body_override if body_override is not None else loop.body
        self.seed_accums(body, True if seed_any else loop.seed)
        var = loop.axis.name
        end = axis_extent(loop.axis)
        if end <= 0:
            return
        if self._should_unroll(loop, end, body):
            self.stats["unrolled_iters"] += end
            for k in range(end):
                e.bind(var, str(k), "s32")
                self.stmts(body)
            return
        i = e.mov_s32(0)
        e.bind(var, i, "s32")
        saved = self._push_flow()
        head = e.label("LOOP")
        e.emit(f"{head}:")
        self.stmts(body)
        e.emit(f"        add.s32         {i}, {i}, 1;")
        self.invalidate(var)
        p = e.new_p()
        e.emit(f"        setp.lt.s32     {p}, {i}, {end};")
        e.emit(f"        @{p} bra        {head};")
        self._pop_flow(saved)

    def emit_strided(self, loop: StridedLoop) -> None:
        e = self.e
        self.seed_accums(loop.body, loop.seed)
        var = loop.axis.name
        start_lit = lit_int(loop.start)
        end_expr = loop.end
        end_lit = lit_int(end_expr) if end_expr is not None else axis_extent(loop.axis)
        step_lit = lit_int(loop.step)
        if (
            start_lit is not None
            and end_lit is not None
            and step_lit is not None
            and step_lit > 0
        ):
            vals = list(range(start_lit, end_lit, step_lit))
            ops = max(self._stmt_count(loop.body), 1)
            if (
                vals
                and not self._has_nested_loop(loop.body)
                and ops * len(vals) <= UNROLL_EXPANSION_CAP
                and (len(vals) == 1 or getattr(loop, "unroll", False))
            ):
                for v in vals:
                    e.bind(var, str(v), "s32")
                    self.stmts(loop.body)
                return
        start = self.expr_s32(loop.start)
        i = e.new_r()
        e.emit(f"        mov.s32         {i}, {start};")
        e.bind(var, i, "s32")
        end = end_lit if end_lit is not None else (
            self.expr_s32(end_expr) if end_expr is not None else str(axis_extent(loop.axis))
        )
        step = step_lit if step_lit is not None else self.expr_s32(loop.step)
        saved = self._push_flow()
        head = e.label("SLOOP")
        done = e.label("SLEND")
        e.emit(f"{head}:")
        p = e.new_p()
        e.emit(f"        setp.ge.s32     {p}, {i}, {end};")
        e.emit(f"        @{p} bra        {done};")
        self.stmts(loop.body)
        e.emit(f"        add.s32         {i}, {i}, {step};")
        self.invalidate(var)
        e.emit(f"        bra             {head};")
        e.emit(f"{done}:")
        self._pop_flow(saved)

    def emit_cond(self, cond: Cond) -> None:
        e = self.e
        p = self.expr_pred(cond.cond)
        els = e.label("ELSE") if cond.else_body else None
        done = e.label("CEND")
        skip = e.new_p()
        e.emit(f"        not.pred        {skip}, {p};")
        e.emit(f"        @{skip} bra      {els or done};")
        saved = self._push_flow()
        self.stmts(cond.body)
        self._pop_flow(saved)
        if cond.else_body:
            e.emit(f"        bra             {done};")
            e.emit(f"{els}:")
            saved = self._push_flow()
            self.stmts(cond.else_body)
            self._pop_flow(saved)
        e.emit(f"{done}:")

    def emit_init(self, s: Init) -> None:
        dt = dtype_name(s.dtype)
        if dt in ("f32", "float", "float32"):
            self.e.bind(s.name, self.e.imm_f32(float(s.identity)), "f32")
        else:
            self.e.bind(s.name, self.e.imm_s32(int(s.identity)), "s32")

    def emit_let(self, s: Let) -> None:
        dt = dtype_name(s.dtype)
        if dt in ("f32", "float", "float32") or (
            s.dtype is None and isinstance(s.value, Literal) and isinstance(s.value.value, float)
        ):
            if isinstance(s.value, Literal):
                self.e.bind(s.name, self.e.imm_f32(float(s.value.value)), "f32")
            else:
                self.e.bind(s.name, self.expr_f32(s.value), "f32")
        else:
            self.e.bind(s.name, self.expr_s32(s.value), "s32")

    def emit_load(self, s: Load) -> None:
        dt = dtype_name(s.dtype) or self.dtypes.get(s.input, "f32")
        pty = ptx_type(dt)
        space = "shared" if s.input in self.smem else "global"
        names = list(s.names)
        n = len(names)
        vec = n if n in (2, 4) and dt in ("f32", "float", "float32") else 1
        idx_key = ", ".join(ix.pretty() for ix in s.index)
        mem_key = f"{s.input}[{idx_key}]|{vec}"
        if mem_key in self.mem and len(self.mem[mem_key]) == n:
            kind = "f32" if dt in ("f32", "float", "float32", "f16", "half") else "s32"
            for name, reg in zip(names, self.mem[mem_key]):
                self.e.bind(name, reg, kind)
            return
        if vec > 1:
            addr = self.addr_of(s.input, s.index, space)
            regs = [self.e.new_f() for _ in names]
            packed = ", ".join(regs)
            self.e.emit(f"        ld.{space}.v{vec}.{pty} {{{packed}}}, [{addr}];")
            for name, reg in zip(names, regs):
                self.e.bind(name, reg, "f32")
            self.mem[mem_key] = tuple(regs)
            return
        regs_out: list[str] = []
        kind = "f32"
        for k, name in enumerate(names):
            idx = s.index
            if k:
                last = BinaryExpr("+", s.index[-1], Literal(k, "int"))
                idx = tuple(s.index[:-1]) + (last,)
            addr = self.addr_of(s.input, idx, space)
            if dt in ("f32", "float", "float32"):
                reg = self.e.new_f()
                self.e.emit(f"        ld.{space}.{pty}   {reg}, [{addr}];")
                self.e.bind(name, reg, "f32")
                regs_out.append(reg)
            elif dt in ("f16", "half"):
                tmp = self.e.new_r()
                dest = self.e.new_f()
                self.e.emit(f"        ld.{space}.f16    {tmp}, [{addr}];")
                self.e.emit(f"        cvt.f32.f16      {dest}, {tmp};")
                self.e.bind(name, dest, "f32")
                regs_out.append(dest)
            else:
                kind = "s32"
                reg = self.e.new_r()
                self.e.emit(f"        ld.{space}.{pty}   {reg}, [{addr}];")
                self.e.bind(name, reg, "s32")
                regs_out.append(reg)
        self.mem[mem_key] = tuple(regs_out)

    def emit_write(self, s: Write) -> None:
        self._kill_mem(s.output)
        dt = dtype_name(s.value_dtype) or self.dtypes.get(s.output, "f32")
        pty = ptx_type(dt)
        space = "shared" if s.output in self.smem else "global"
        vals = list(s.values)
        n = len(vals)
        if s.atomic:
            addr = self.addr_of(s.output, s.index, space)
            src = self.as_f32(vals[0])
            self.e.emit(f"        red.{space}.add.f32 [{addr}], {src};")
            return
        vec = n if n in (2, 4) and dt in ("f32", "float", "float32") and not s.atomic else 1
        if vec > 1:
            addr = self.addr_of(s.output, s.index, space)
            regs = [self.as_f32(v) for v in vals]
            packed = ", ".join(regs)
            self.e.emit(f"        st.{space}.v{vec}.{pty} [{addr}], {{{packed}}};")
            return
        for k, name in enumerate(vals):
            idx = s.index
            if k:
                last = BinaryExpr("+", s.index[-1], Literal(k, "int"))
                idx = tuple(s.index[:-1]) + (last,)
            addr = self.addr_of(s.output, idx, space)
            if dt in ("f32", "float", "float32"):
                self.e.emit(f"        st.{space}.{pty}   [{addr}], {self.as_f32(name)};")
            else:
                self.e.emit(f"        st.{space}.{pty}   [{addr}], {self.as_s32(name)};")

    def emit_assign(self, s: Assign) -> None:
        op = s.op.name if hasattr(s.op, "name") else str(s.op)
        args = list(s.args)
        dt = dtype_name(s.dtype)
        if dt in INT_DTYPES or (op in S32_OPS and dt not in ("f32", "float", "float32", "") and dt in INT_DTYPES):
            dest = self.e.env.get(s.name) if self.e.kind.get(s.name) == "s32" else self.e.new_r()
            if op in S32_OPS and len(args) == 2:
                a, b = self.as_s32(args[0]), self.as_s32(args[1])
                self.e.emit(f"        {S32_OPS[op]:<16}{dest}, {a}, {b};")
                self.e.bind(s.name, dest, "s32")
                return
            if op in ("neg", "negative") and args:
                self.e.emit(f"        neg.s32         {dest}, {self.as_s32(args[0])};")
                self.e.bind(s.name, dest, "s32")
                return
            if op in ("abs", "absolute") and args:
                self.e.emit(f"        abs.s32         {dest}, {self.as_s32(args[0])};")
                self.e.bind(s.name, dest, "s32")
                return
        dest = self.e.env.get(s.name) if self.e.kind.get(s.name) == "f32" else self.e.new_f()
        if op in F32_OPS and len(args) == 2:
            a, b = self.as_f32(args[0]), self.as_f32(args[1])
            self.e.emit(f"        {F32_OPS[op]:<16}{dest}, {a}, {b};")
        elif op in CMP_ASSIGN and len(args) == 2:
            cmp = CMP_ASSIGN[op]
            a, b = self.as_f32(args[0]), self.as_f32(args[1])
            p = self.e.new_p()
            self.e.emit(f"        setp.{cmp}.f32    {p}, {a}, {b};")
            one, z = self.e.imm_f32(1.0), self.e.imm_f32(0.0)
            self.e.emit(f"        selp.f32        {dest}, {one}, {z}, {p};")
        elif op == "exp2":
            self.e.emit(f"        ex2.approx.ftz.f32 {dest}, {self.as_f32(args[0])};")
        elif op in ("exp", "exp_fast"):
            x = self.as_f32(args[0])
            tmp = self.e.new_f()
            self.e.emit(f"        mul.f32         {tmp}, {x}, {f32_hex(1.0 / math.log(2.0))};")
            self.e.emit(f"        ex2.approx.ftz.f32 {dest}, {tmp};")
        elif op == "sqrt":
            self.e.emit(f"        sqrt.rn.f32     {dest}, {self.as_f32(args[0])};")
        elif op == "rsqrt":
            self.e.emit(f"        rsqrt.approx.ftz.f32 {dest}, {self.as_f32(args[0])};")
        elif op in ("reciprocal", "recip"):
            self.e.emit(f"        rcp.rn.f32      {dest}, {self.as_f32(args[0])};")
        elif op == "square" and args:
            x = self.as_f32(args[0])
            self.e.emit(f"        mul.f32         {dest}, {x}, {x};")
        elif op in ("clamp", "clip") and len(args) == 3:
            x, lo, hi = self.as_f32(args[0]), self.as_f32(args[1]), self.as_f32(args[2])
            tmp = self.e.new_f()
            self.e.emit(f"        max.f32         {tmp}, {x}, {lo};")
            self.e.emit(f"        min.f32         {dest}, {tmp}, {hi};")
        elif op == "relu":
            self.e.emit(f"        max.f32         {dest}, {self.as_f32(args[0])}, {f32_hex(0.0)};")
        elif op in ("leaky_relu",) and len(args) >= 1:
            x = self.as_f32(args[0])
            slope = self.as_f32(args[1]) if len(args) > 1 else self.e.imm_f32(0.01)
            p = self.e.new_p()
            z = self.e.imm_f32(0.0)
            self.e.emit(f"        setp.gt.f32     {p}, {x}, {z};")
            scaled = self.e.new_f()
            self.e.emit(f"        mul.f32         {scaled}, {x}, {slope};")
            self.e.emit(f"        selp.f32        {dest}, {x}, {scaled}, {p};")
        elif op in ("neg", "negative"):
            self.e.emit(f"        neg.f32         {dest}, {self.as_f32(args[0])};")
        elif op in ("abs", "absolute"):
            self.e.emit(f"        abs.f32         {dest}, {self.as_f32(args[0])};")
        elif op == "copy":
            self.e.emit(f"        mov.f32         {dest}, {self.as_f32(args[0])};")
        elif op in ("log", "log_fast"):
            tmp = self.e.new_f()
            self.e.emit(f"        lg2.approx.ftz.f32 {tmp}, {self.as_f32(args[0])};")
            self.e.emit(f"        mul.f32         {dest}, {tmp}, {f32_hex(math.log(2.0))};")
        elif op in ("log2", "lg2"):
            self.e.emit(f"        lg2.approx.ftz.f32 {dest}, {self.as_f32(args[0])};")
        elif op == "tanh":
            # tanh(x) ~ (exp(2x)-1)/(exp(2x)+1) with approx exp
            x = self.as_f32(args[0])
            two = self.e.imm_f32(2.0)
            twox = self.e.new_f()
            self.e.emit(f"        mul.f32         {twox}, {x}, {two};")
            ex = self.e.new_f()
            tmp = self.e.new_f()
            self.e.emit(f"        mul.f32         {tmp}, {twox}, {f32_hex(1.0 / math.log(2.0))};")
            self.e.emit(f"        ex2.approx.ftz.f32 {ex}, {tmp};")
            one = self.e.imm_f32(1.0)
            num = self.e.new_f()
            den = self.e.new_f()
            self.e.emit(f"        sub.f32         {num}, {ex}, {one};")
            self.e.emit(f"        add.f32         {den}, {ex}, {one};")
            self.e.emit(f"        div.rn.f32      {dest}, {num}, {den};")
        elif op == "sigmoid":
            x = self.as_f32(args[0])
            nx = self.e.new_f()
            self.e.emit(f"        neg.f32         {nx}, {x};")
            tmp = self.e.new_f()
            ex = self.e.new_f()
            self.e.emit(f"        mul.f32         {tmp}, {nx}, {f32_hex(1.0 / math.log(2.0))};")
            self.e.emit(f"        ex2.approx.ftz.f32 {ex}, {tmp};")
            one = self.e.imm_f32(1.0)
            den = self.e.new_f()
            self.e.emit(f"        add.f32         {den}, {one}, {ex};")
            self.e.emit(f"        rcp.rn.f32      {dest}, {den};")
        elif op == "silu":
            x = self.as_f32(args[0])
            nx = self.e.new_f()
            self.e.emit(f"        neg.f32         {nx}, {x};")
            tmp = self.e.new_f()
            ex = self.e.new_f()
            self.e.emit(f"        mul.f32         {tmp}, {nx}, {f32_hex(1.0 / math.log(2.0))};")
            self.e.emit(f"        ex2.approx.ftz.f32 {ex}, {tmp};")
            one = self.e.imm_f32(1.0)
            den = self.e.new_f()
            sigr = self.e.new_f()
            self.e.emit(f"        add.f32         {den}, {one}, {ex};")
            self.e.emit(f"        rcp.rn.f32      {sigr}, {den};")
            self.e.emit(f"        mul.f32         {dest}, {x}, {sigr};")
        elif op == "where" and len(args) == 3:
            p = self.as_pred(args[0])
            a, b = self.as_f32(args[1]), self.as_f32(args[2])
            self.e.emit(f"        selp.f32        {dest}, {a}, {b}, {p};")
        elif op == "pow" and len(args) == 2:
            # a**b = exp(b*log(a))
            a, b = self.as_f32(args[0]), self.as_f32(args[1])
            lg = self.e.new_f()
            self.e.emit(f"        lg2.approx.ftz.f32 {lg}, {a};")
            mul = self.e.new_f()
            self.e.emit(f"        mul.f32         {mul}, {lg}, {b};")
            self.e.emit(f"        ex2.approx.ftz.f32 {dest}, {mul};")
        else:
            print(f"unsupported assign op {op!r}", file=sys.stderr)
            raise SystemExit(2)
        self.e.bind(s.name, dest, "f32")

    def emit_accum(self, s: Accum) -> None:
        if s.name not in self.e.env:
            ident = identity_of(s.op)
            self.e.bind(s.name, self.e.imm_f32(ident), "f32")
        dest = self.e.env[s.name]
        rhs = self.as_f32(s.value)
        left = self.as_f32(s.base) if s.base else dest
        op = s.op.name if hasattr(s.op, "name") else str(s.op)
        inst = F32_OPS.get(op, "add.f32")
        self.e.emit(f"        {inst:<16}{dest}, {left}, {rhs};")
        self.e.bind(s.name, dest, "f32")

    def emit_warpshuffle(self, s: WarpShuffle) -> None:
        self.stats["warp_shuffles"] += 1
        e = self.e
        mask = e.new_r()
        e.emit(f"        activemask.b32  {mask};")
        step = int(s.length) // 2
        while step > 0:
            for st, sb in zip(s.state, s.state_b):
                src = self.as_f32(st)
                bits = e.new_r()
                dst = e.new_f()
                e.emit(f"        mov.b32         {bits}, {src};")
                e.emit(f"        shfl.sync.bfly.b32 {bits}, {bits}, {step}, 31, {mask};")
                e.emit(f"        mov.b32         {dst}, {bits};")
                e.bind(sb, dst, "f32")
            for asn in s.combine_states:
                self.emit_assign(asn)
            step >>= 1

    def emit_treehalve(self, s: TreeHalve) -> None:
        self.stats["tree_halves"] += 1
        e = self.e
        tid = self.as_s32(s.tid_var)
        step = int(s.length) // 2
        while step > 0:
            take = e.new_p()
            e.emit(f"        setp.lt.s32     {take}, {tid}, {step};")
            skip = e.label("TSTEP")
            nt = e.new_p()
            e.emit(f"        not.pred        {nt}, {take};")
            e.emit(f"        @{nt} bra       {skip};")
            saved = self._push_flow()
            svar = e.imm_s32(step)
            for buf, st, sb in zip(s.bufs, s.state, s.state_b):
                if s.inner is not None:
                    iv, scale = s.inner
                    slot = self.mad_imm(tid, int(scale), self.as_s32(iv))
                    slot_s = self.mad_imm(self.add_s32(tid, svar), int(scale), self.as_s32(iv))
                else:
                    slot = tid
                    slot_s = self.add_s32(tid, svar)
                e.bind(st, self.load_smem_elem(buf, slot), "f32")
                e.bind(sb, self.load_smem_elem(buf, slot_s), "f32")
            for asn in s.combine_states:
                self.emit_assign(asn)
            for buf, st in zip(s.bufs, s.state):
                if s.inner is not None:
                    iv, scale = s.inner
                    slot = self.mad_imm(tid, int(scale), self.as_s32(iv))
                else:
                    slot = tid
                self.store_smem_elem(buf, slot, self.as_f32(st))
            e.emit(f"{skip}:")
            self._pop_flow(saved)
            for buf in s.bufs:
                self._kill_mem(buf)
            self.emit_sync(Sync(barrier_id=s.barrier_id, count=s.barrier_count))
            step >>= 1
        for buf, st in zip(s.bufs, s.state):
            if s.inner is not None:
                iv, _scale = s.inner
                root = self.as_s32(iv)
            else:
                root = e.imm_s32(0)
            e.bind(st, self.load_smem_elem(buf, root), "f32")

    def emit_sync(self, s: Sync) -> None:
        if s.warp:
            self.e.emit("        bar.warp.sync   0xffffffff;")
            return
        if s.barrier_id == 0:
            self.e.emit("        bar.sync        0;")
        else:
            self.e.emit(f"        bar.sync        {s.barrier_id}, {int(s.count)};")

    def emit_cp_async(self, s: CpAsyncCopy) -> None:
        self.stats["cp_async"] += 1
        e = self.e
        smem_idx = self.flatten(s.smem, s.smem_index)
        if s.lane_index is not None:
            cols = int(self.shapes[s.smem][-1])
            lane_row = self.expr_s32(s.lane_index[0])
            lane_col = self.expr_s32(s.lane_index[1])
            lane = self.mad_imm(lane_row, cols, lane_col)
            smem_idx = self.add_s32(smem_idx, lane)
        src_idx = self.flatten(s.src, s.src_index)
        smem_nb = nbytes_of(self.dtypes.get(s.smem, "f32"))
        src_nb = nbytes_of(self.dtypes.get(s.src, "f32"))
        src_imm = parse_imm(src_idx)
        if src_imm is not None:
            off = src_imm * src_nb
            src = self.ptrs[s.src] if off == 0 else f"{self.ptrs[s.src]} + {off}"
        else:
            src = e.new_rd()
            e.emit(f"        mad.wide.s32    {src}, {src_idx}, {src_nb}, {self.ptrs[s.src]};")
        dst_imm = parse_imm(smem_idx)
        if dst_imm is not None:
            off = dst_imm * smem_nb
            dst = self.smem_base[s.smem] if off == 0 else f"{self.smem_base[s.smem]} + {off}"
        else:
            dst = e.new_r()
            e.emit(f"        mad.lo.s32      {dst}, {smem_idx}, {smem_nb}, {self.smem_base[s.smem]};")
        n = int(s.nbytes)
        q = "cg" if n == 16 else "ca"
        e.emit(f"        cp.async.{q}.shared.global [{dst}], [{src}], {n};")
        self._kill_mem(s.smem)

    def _divmod_recompose(self, q, r, dim: int):
        if not isinstance(q, BinaryExpr) or not isinstance(r, BinaryExpr):
            return None
        if q.op not in ("/", "//") or r.op != "%":
            return None
        if lit_int(q.right) != dim or lit_int(r.right) != dim:
            return None
        if q.left.pretty() == r.left.pretty():
            return q.left
        return None

    def flatten(self, buf: str, indices) -> str:
        if not indices:
            return "0"
        shape = self.shapes.get(buf)
        if shape and len(indices) == 2 and len(shape) == 2:
            rec = self._divmod_recompose(indices[0], indices[1], int(shape[1]))
            if rec is not None:
                return self.expr_s32(rec)
        if len(indices) == 1:
            return self.expr_s32(indices[0])
        if not shape or len(shape) != len(indices):
            acc = self.expr_s32(indices[0])
            for ix in indices[1:]:
                acc = self.add_s32(acc, self.expr_s32(ix))
            return acc
        acc = None
        for d, ix in enumerate(indices):
            stride = 1
            for k in range(d + 1, len(shape)):
                stride *= int(shape[k])
            term = self.expr_s32(ix)
            if acc is None:
                acc = term if stride == 1 else self.mul_imm(term, stride)
            elif stride == 1:
                acc = self.add_s32(acc, term)
            else:
                acc = self.mad_imm(term, stride, acc)
        return acc

    def addr_of(self, buf: str, indices, space: str) -> str:
        idx = self.flatten(buf, indices)
        nb = nbytes_of(self.dtypes.get(buf, "f32"))
        base = self.smem_base[buf] if space == "shared" else self.ptrs[buf]
        imm = parse_imm(idx)
        if imm is not None:
            off = imm * nb
            if off == 0:
                return base
            return f"{base} + {off}"
        if space == "shared":
            dst = self.e.new_r()
            self.e.emit(f"        mad.lo.s32      {dst}, {idx}, {nb}, {base};")
            return dst
        addr = self.e.new_rd()
        self.e.emit(f"        mad.wide.s32    {addr}, {idx}, {nb}, {base};")
        return addr

    def load_smem_elem(self, buf: str, idx_reg: str) -> str:
        nb = nbytes_of(self.dtypes.get(buf, "f32"))
        base = self.smem_base[buf]
        imm = parse_imm(idx_reg)
        dest = self.e.new_f()
        if imm is not None:
            off = imm * nb
            addr = base if off == 0 else f"{base} + {off}"
            self.e.emit(f"        ld.shared.f32   {dest}, [{addr}];")
            return dest
        addr = self.e.new_r()
        self.e.emit(f"        mad.lo.s32      {addr}, {idx_reg}, {nb}, {base};")
        self.e.emit(f"        ld.shared.f32   {dest}, [{addr}];")
        return dest

    def store_smem_elem(self, buf: str, idx_reg: str, src: str) -> None:
        self._kill_mem(buf)
        nb = nbytes_of(self.dtypes.get(buf, "f32"))
        base = self.smem_base[buf]
        imm = parse_imm(idx_reg)
        if imm is not None:
            off = imm * nb
            addr = base if off == 0 else f"{base} + {off}"
            self.e.emit(f"        st.shared.f32   [{addr}], {src};")
            return
        addr = self.e.new_r()
        self.e.emit(f"        mad.lo.s32      {addr}, {idx_reg}, {nb}, {base};")
        self.e.emit(f"        st.shared.f32   [{addr}], {src};")

    def as_f32(self, name: str) -> str:
        if name in self.e.env and self.e.kind.get(name) == "f32":
            return self.e.env[name]
        if name in self.e.env and self.e.kind.get(name) == "s32":
            d = self.e.new_f()
            self.e.emit(f"        cvt.rn.f32.s32  {d}, {self.e.env[name]};")
            return d
        try:
            return self.e.imm_f32(float(name))
        except ValueError:
            print(f"unknown f32 value {name!r}", file=sys.stderr)
            raise SystemExit(2)

    def as_s32(self, name: str) -> str:
        if name in self.e.env:
            if self.e.kind.get(name) == "s32":
                return self.e.env[name]
            d = self.e.new_r()
            self.e.emit(f"        cvt.rzi.s32.f32 {d}, {self.e.env[name]};")
            return d
        return self.e.imm_s32(int(float(name)))

    def as_pred(self, name: str) -> str:
        if name in self.e.env and self.e.kind.get(name) == "pred":
            return self.e.env[name]
        src = self.as_f32(name)
        p = self.e.new_p()
        z = self.e.imm_f32(0.0)
        self.e.emit(f"        setp.ne.f32     {p}, {src}, {z};")
        return p

    def expr_pred(self, expr) -> str:
        kind, reg = self.expr(expr)
        if kind == "pred":
            return reg
        p = self.e.new_p()
        if kind == "f32":
            z = self.e.imm_f32(0.0)
            self.e.emit(f"        setp.ne.f32     {p}, {reg}, {z};")
        else:
            self.e.emit(f"        setp.ne.s32     {p}, {reg}, 0;")
        return p

    def expr_s32(self, expr) -> str:
        kind, reg = self.expr(expr)
        if kind == "s32":
            return reg
        if kind == "pred":
            d = self.e.new_r()
            self.e.emit(f"        selp.s32        {d}, 1, 0, {reg};")
            return d
        d = self.e.new_r()
        self.e.emit(f"        cvt.rzi.s32.f32 {d}, {reg};")
        return d

    def expr_f32(self, expr) -> str:
        kind, reg = self.expr(expr)
        if kind == "f32":
            return reg
        d = self.e.new_f()
        if kind == "pred":
            one = self.e.imm_f32(1.0)
            z = self.e.imm_f32(0.0)
            self.e.emit(f"        selp.f32        {d}, {one}, {z}, {reg};")
            return d
        self.e.emit(f"        cvt.rn.f32.s32  {d}, {reg};")
        return d

    def expr(self, expr) -> tuple[str, str]:
        if hasattr(expr, "simplify"):
            try:
                expr = expr.simplify(SimplifyCtx.empty())
            except Exception:
                pass
        try:
            key = expr.pretty()
        except Exception:
            key = None
        if key and key in self.cse:
            self.stats["cse_hits"] += 1
            return self.cse[key]
        result = self._expr(expr)
        if key:
            self.cse[key] = result
        return result

    def _expr(self, expr) -> tuple[str, str]:
        if isinstance(expr, Literal):
            if expr.dtype in ("int", "i32", "s32") or isinstance(expr.value, int) and not isinstance(expr.value, bool) and expr.dtype != "float":
                if isinstance(expr.value, float) and expr.dtype == "float":
                    return "f32", self.e.imm_f32(float(expr.value))
                return "s32", self.e.imm_s32(int(expr.value))
            if isinstance(expr.value, float) or expr.dtype in ("float", "f32"):
                return "f32", self.e.imm_f32(float(expr.value))
            return "s32", self.e.imm_s32(int(expr.value))
        if isinstance(expr, Var):
            if expr.name in self.e.env:
                return self.e.kind[expr.name], self.e.env[expr.name]
            print(f"unbound var {expr.name!r}", file=sys.stderr)
            raise SystemExit(2)
        if isinstance(expr, Builtin):
            mapping = {
                "threadIdx.x": "_tid",
                "blockDim.x": "_ntid",
                "blockIdx.x": "_cta",
            }
            name = mapping.get(expr.name, expr.name)
            return "s32", self.e.env[name]
        if isinstance(expr, CastExpr):
            inner_k, inner = self.expr(expr.expr)
            dt = dtype_name(expr.dtype)
            if dt in ("f32", "float"):
                return "f32", self.expr_f32(expr.expr) if inner_k != "f32" else inner
            return "s32", self.expr_s32(expr.expr) if inner_k != "s32" else inner
        if isinstance(expr, TernaryExpr):
            p = self.expr_pred(expr.cond)
            tk, tr = self.expr(expr.if_true)
            fk, fr = self.expr(expr.if_false)
            if tk == "f32" or fk == "f32":
                a = tr if tk == "f32" else self.expr_f32(expr.if_true)
                b = fr if fk == "f32" else self.expr_f32(expr.if_false)
                d = self.e.new_f()
                self.e.emit(f"        selp.f32        {d}, {a}, {b}, {p};")
                return "f32", d
            a = tr if tk == "s32" else self.expr_s32(expr.if_true)
            b = fr if fk == "s32" else self.expr_s32(expr.if_false)
            d = self.e.new_r()
            self.e.emit(f"        selp.s32        {d}, {a}, {b}, {p};")
            return "s32", d
        if isinstance(expr, FuncCallExpr):
            names = []
            for i, a in enumerate(expr.args):
                k, r = self.expr(a)
                nm = f"__fc{self.e.nl}_{i}"
                self.e.bind(nm, r, k)
                names.append(nm)
            tmp = f"__fcout{self.e.nl}"
            self.emit_assign(Assign(name=tmp, op=ElementwiseImpl(expr.name), args=tuple(names)))
            return "f32", self.e.env[tmp]
        if isinstance(expr, BinaryExpr):
            return self.binary(expr)
        print(f"unsupported expr {type(expr).__name__}", file=sys.stderr)
        raise SystemExit(2)

    def binary(self, expr: BinaryExpr) -> tuple[str, str]:
        op = expr.op
        li, ri = lit_int(expr.left), lit_int(expr.right)
        if op == "-" and isinstance(expr.left, BinaryExpr) and expr.left.op == "+":
            lp, rp = expr.left.left.pretty(), expr.left.right.pretty()
            rr = expr.right.pretty()
            if lp == rr:
                return self.expr(expr.left.right)
            if rp == rr:
                return self.expr(expr.left.left)
        if li is not None and ri is not None and op not in CMP_OPS and op not in ("&&", "||"):
            if op == "+":
                return "s32", str(li + ri)
            if op == "-":
                return "s32", str(li - ri)
            if op == "*":
                return "s32", str(li * ri)
            if op in ("/", "//") and ri:
                return "s32", str(li // ri)
            if op == "%" and ri:
                return "s32", str(li % ri)
            if op == "^":
                return "s32", str(li ^ ri)
        if op in CMP_OPS or op in ("&&", "||"):
            if op in ("&&", "||"):
                a = self.expr_pred(expr.left)
                b = self.expr_pred(expr.right)
                d = self.e.new_p()
                inst = "and.pred" if op == "&&" else "or.pred"
                self.e.emit(f"        {inst:<16}{d}, {a}, {b};")
                return "pred", d
            lk, lr = self.expr(expr.left)
            p = self.e.new_p()
            cmp = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}[op]
            if lk == "f32" or (lit_int(expr.right) is None and not isinstance(expr.right, Literal)):
                rk, rr = self.expr(expr.right)
                if lk == "f32" or rk == "f32":
                    a = lr if lk == "f32" else self.expr_f32(expr.left)
                    b = rr if rk == "f32" else self.expr_f32(expr.right)
                    self.e.emit(f"        setp.{cmp}.f32    {p}, {a}, {b};")
                    return "pred", p
                self.e.emit(f"        setp.{cmp}.s32    {p}, {lr}, {rr};")
                return "pred", p
            if ri is not None and lk == "s32":
                if parse_imm(lr) is not None:
                    lr = self.e.mov_s32(lr)
                self.e.emit(f"        setp.{cmp}.s32    {p}, {lr}, {ri};")
                return "pred", p
            rk, rr = self.expr(expr.right)
            self.e.emit(f"        setp.{cmp}.s32    {p}, {lr}, {rr};")
            return "pred", p
        if ri is not None:
            left = self.expr_s32(expr.left)
            if op == "*":
                return "s32", self.mul_imm(left, ri)
            if op in ("/", "//"):
                return "s32", self.div_imm(left, ri)
            if op == "%":
                return "s32", self.rem_imm(left, ri)
            if op == "+":
                return "s32", self.add_s32(left, str(ri))
            if op == "-":
                return "s32", self.sub_reg(left, str(ri))
            if op == "^":
                d = self.e.new_r()
                self.e.emit(f"        xor.b32         {d}, {left}, {ri};")
                return "s32", d
            if op == "&":
                d = self.e.new_r()
                self.e.emit(f"        and.b32         {d}, {left}, {ri};")
                return "s32", d
            if op == "|":
                d = self.e.new_r()
                self.e.emit(f"        or.b32          {d}, {left}, {ri};")
                return "s32", d
            if op in ("<<", "shl"):
                d = self.e.new_r()
                self.e.emit(f"        shl.b32         {d}, {left}, {ri};")
                return "s32", d
            if op in (">>", "shr"):
                d = self.e.new_r()
                self.e.emit(f"        shr.s32         {d}, {left}, {ri};")
                return "s32", d
            print(f"unsupported binop {op!r}", file=sys.stderr)
            raise SystemExit(2)
        if li is not None and op in ("+", "*"):
            right = self.expr_s32(expr.right)
            if op == "*":
                return "s32", self.mul_imm(right, li)
            return "s32", self.add_s32(right, str(li))
        lk, lr = self.expr(expr.left)
        rk, rr = self.expr(expr.right)
        if lk == "f32" or rk == "f32":
            a = lr if lk == "f32" else self.expr_f32(expr.left)
            b = rr if rk == "f32" else self.expr_f32(expr.right)
            d = self.e.new_f()
            inst = {"+": "add.f32", "-": "sub.f32", "*": "mul.f32", "/": "div.rn.f32", "//": "div.rn.f32"}[op]
            self.e.emit(f"        {inst:<16}{d}, {a}, {b};")
            return "f32", d
        if op == "+":
            return "s32", self.add_s32(lr, rr)
        if op == "-":
            return "s32", self.sub_reg(lr, rr)
        if op == "*":
            rr_imm = parse_imm(rr)
            if rr_imm is not None:
                return "s32", self.mul_imm(lr, rr_imm)
            lr_imm = parse_imm(lr)
            if lr_imm is not None:
                return "s32", self.mul_imm(rr, lr_imm)
            d = self.e.new_r()
            self.e.emit(f"        mul.lo.s32      {d}, {lr}, {rr};")
            return "s32", d
        d = self.e.new_r()
        if op in ("/", "//"):
            self.e.emit(f"        div.s32         {d}, {lr}, {rr};")
        elif op == "%":
            self.e.emit(f"        rem.s32         {d}, {lr}, {rr};")
        elif op == "^":
            self.e.emit(f"        xor.b32         {d}, {lr}, {rr};")
        elif op == "&":
            self.e.emit(f"        and.b32         {d}, {lr}, {rr};")
        elif op == "|":
            self.e.emit(f"        or.b32          {d}, {lr}, {rr};")
        elif op in ("<<", "shl"):
            self.e.emit(f"        shl.b32         {d}, {lr}, {rr};")
        elif op in (">>", "shr"):
            self.e.emit(f"        shr.s32         {d}, {lr}, {rr};")
        else:
            print(f"unsupported binop {op!r}", file=sys.stderr)
            raise SystemExit(2)
        return "s32", d


def compile_ir(path: Path) -> str:
    return compile_bundle(path)["ptx"]


def compile_bundle(path: Path) -> dict:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    graph = Graph.from_dict(raw)
    kop = find_kernel(graph)
    meta_k = load_meta_kernel(path, kop.name)
    args = arg_order_for(kop, meta_k)
    shapes, dtypes = load_shapes(raw, kop, meta_k)
    compiler = Compiler(kop, shapes, dtypes, args, meta_k)
    ptx = compiler.run()
    txt = path.with_suffix(".txt")
    if not txt.is_file():
        txt = path.parent / (path.stem + ".ir.txt")
        if not txt.is_file():
            matches = list(path.parent.glob("*.ir.txt"))
            txt = matches[0] if matches else None
    ir_txt = txt.read_text(encoding="utf-8") if txt and txt.is_file() else ""
    mix: dict[str, int] = {}
    for line in ptx.splitlines():
        t = line.strip()
        if not t or t.startswith(".") or t.startswith("//") or t.endswith(":"):
            continue
        if t.startswith("@"):
            t = t.split(None, 1)[-1]
        op = t.split()[0]
        mix[op] = mix.get(op, 0) + 1
    return {
        "ptx": ptx,
        "entry": kop.name,
        "args": args,
        "shapes": {k: list(v) for k, v in shapes.items()},
        "dtypes": dtypes,
        "smem_bytes": compiler.smem_total,
        "dynamic_smem": compiler.use_dynamic_smem,
        "block": compiler._launch_block(),
        "virt_regs": {
            "pred": compiler.e.np,
            "f32": compiler.e.nf,
            "b32": compiler.e.nr,
            "b64": compiler.e.nrd,
        },
        "stats": compiler.stats,
        "ops": mix,
        "ir_txt": ir_txt,
        "meta": meta_k or {},
        "bytes": len(ptx.encode("ascii", "replace")),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python compile.py path/to/kernel.ir.json", file=sys.stderr)
        return 2
    sys.stdout.write(compile_ir(Path(sys.argv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
