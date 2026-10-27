#!/usr/bin/env python3
"""Kernel IR -> PTX printer.

Contract:  python compile.py path/to/kernel.ir.json > out.ptx

This is the last stage of an ML compiler, not the whole compiler.
Emmy already chose the schedule (tiles, cp.async, tree reduce). We only
print that schedule as PTX. No problem-id table, no shape switch.

How to read this file (top to bottom):

  helpers     dtype / walk / meta ABI
  Emitter     virtual registers (%f, %r, %rd, %p). ptxas allocates for real.
  Compiler    one KernelOp -> one .entry
    prologue  params, cvta, 3-D tid/cta, lane/warp
    stmt()    dispatch table for every IR node
    emit_*    one function per node, named like the IR
    extra ISA TMA / mbarrier / mma.sync / wgmma  (TASK.md * nodes)
  compile_ir  the argv[1] entry point

Pipeline:

  1. Graph.from_dict. Entry name = KernelOp.name.
  2. Parameter order from sibling meta.json arg_order.
  3. Prologue, then Tile N-guard if the launch has extra threads.
  4. Walk body in order. SSA name -> virtual register.
  5. Local peepholes only: FMA, CSE, same-axis loop fusion,
     bounded unroll, ld.v2/v4 pairing, [ptr+imm].

Smem above 48 KB is `.extern .shared` so meta.smem_bytes sizes it at launch.
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
    FlatIndex,
    FuncCallExpr,
    Literal,
    SimplifyCtx,
    TernaryExpr,
    Var,
)
from emmy.compiler.ir.kernel.ir import (
    BlockScaleLoad,
    CpAsyncCommit,
    CpAsyncCopy,
    CpAsyncWait,
    FragmentApply,
    FragmentPromote,
    FragmentRepack,
    FragmentRowReduce,
    KernelOp,
    LdmatrixLoad,
    MbarrierArrive,
    MbarrierArriveExpectTx,
    MbarrierInit,
    MbarrierWait,
    MmaSyncPtx,
    RegFragment,
    RegStore,
    Select,
    SetMaxNReg,
    Smem,
    Sync,
    Tile,
    TmaDescriptor,
    TmaLoad,
    TreeHalve,
    WarpShuffle,
    WgmmaCommit,
    WgmmaDescriptor,
    WgmmaFence,
    WgmmaMma,
    WgmmaWait,
)
from emmy.compiler.ir.stmt import (
    Accum,
    Assign,
    Carry,
    Cond,
    Init,
    Let,
    Load,
    Loop,
    Pre,
    StridedLoop,
    Write,
    ZeroPrologue,
)

PTX_VERSION = "8.0"
PTX_TARGET = "sm_89"
# sm_89 static shared cap (PTX ISA), not a problem size.
SMEM_STATIC_LIMIT = 48 * 1024
# Bound fully-expanded innermost loops so ptxas / I-cache stay healthy.
# Derived from body size × trip count, never from a named problem.
# 768 keeps a01/a02 as a real loop (occupancy) instead of 200+ live f32s.
UNROLL_EXPANSION_CAP = 768

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
    "bf16": "bf16",
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


# Virtual registers. PTX has no physical file here — ptxas does that later.
class Emitter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.nf = 1
        self.nd = 1
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

    def new_d(self) -> str:
        r = f"%fd{self.nd}"
        self.nd += 1
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


# One KernelOp in, one .visible .entry out.
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
        self.arith: dict[str, str] = {}
        self.addr_cse: dict[tuple, str] = {}
        self.e.on_bind = self.invalidate
        # Threads with gid >= Tile.N. None = every lane is in-range.
        # Never branch those lanes around bar.sync / TreeHalve / cp.async.wait.
        self.active: str | None = None
        self.stats: dict[str, int] = {
            "fma": 0,
            "vec_loads": 0,
            "vec_stores": 0,
            "cse_hits": 0,
            "unrolled_iters": 0,
            "fused_loops": 0,
            "cp_async": 0,
            "warp_shuffles": 0,
            "tree_halves": 0,
            "mma": 0,
            "ldmatrix": 0,
            "tma": 0,
            "mbarrier": 0,
            "wgmma": 0,
        }
        self.frags: dict[str, dict] = {}
        self.tma_descs: set[str] = set()
        self.ptx_target = PTX_TARGET
        self.written: set[str] = set()
        # Read-only global buffers staged into shared memory once per
        # block (name -> shape), see _plan_buffer_cache.
        self.cached_buf: dict[str, tuple[int, ...]] = {}
        # Reduction Loops (id(loop) -> lane-axis name) rewritten to split
        # across warp lanes + shfl.sync.bfly instead of 32x redundant
        # per-lane serial work, see _plan_warp_reduce_loops.
        self.warp_reduce_loops: dict[int, str] = {}
        # Active (var_name, k0_register, k) while emit_loop's factor>1
        # path (see _partial_factor) is lowering unrolled copy `k` of a
        # physical loop trip. addr_of consults this to turn a full
        # address recompute into "reuse copy 0's cached address + a
        # compile-time-constant byte offset" whenever the index is
        # provably affine in this var - see _unroll_addr_coeff.
        self._unroll_subst: tuple[str, str, int] | None = None
        self.carry_vals: dict[tuple[str, str], str] = {}
        self.carry_seed: dict[str, float] = {}
        for s in walk_stmts(kop.body):
            if isinstance(s, Write):
                self.written.add(s.output)
            elif isinstance(s, RegStore):
                self.written.add(s.dst_buffer)
            elif isinstance(s, ZeroPrologue):
                self.written.add(s.dst)

    def _plan_buffer_cache(self) -> None:
        """Find read-only global buffers whose every access anywhere in
        the kernel is driven purely by block-uniform index expressions:
        Tile axes whose stride spans at least one whole block (so every
        thread in a block computes the same axis value), or ordinary
        Loop/StridedLoop induction variables whose own start is itself
        uniform. Every thread in the block then touches the identical
        address at the identical program point inside a real loop - the
        structural reason a02 used to re-read the same K values from
        global memory on every loop trip despite an already-low register
        count: K is shared across every output row in an attention tile,
        so caching it once per block (instead of re-reading it from
        global memory on every loop trip) cuts redundant global traffic
        without touching the per-thread Q/V accesses that must stay as-is.

        This is a general analysis, not an a02-specific special case -
        it naturally also applies to a01 (same kt-is-row-independent
        shape) and naturally yields nothing for the pointwise/reduction/
        matmul/fused kernels, where every read-only buffer's index does
        depend on a thread-varying axis. Falls back to doing nothing
        (the existing, already-verified global-memory path) whenever
        the buffer is too big for the remaining static shared-memory
        budget, is ever written, isn't plain f32, or the kernel already
        needs dynamic shared memory - never a correctness risk.

        Critically, "uniform" here means "built only from Loop /
        StridedLoop induction variables" - NEVER a Tile axis, even a
        block-uniform one (constant stride >= block_threads). A Tile
        axis is constant for a whole block's lifetime but DIFFERENT
        across blocks, so an index built only from Tile axes (e.g.
        a02's q[row, a2] where row = a Tile axis, a2 = a Loop var)
        touches only ONE row per block, not the whole buffer; caching
        the WHOLE buffer for that case wastes shared-memory budget on
        the other N-1 rows this block will never read (measured: caching
        q as well as kt bloated a02 to 32KB smem, roughly halving
        achievable occupancy, and made the kernel ~1.9x SLOWER despite
        fewer registers - occupancy lost from the wasted smem outweighed
        the saved address arithmetic). Only Loop/StridedLoop variables -
        which sweep their full declared range inside EVERY block's own
        execution - make the whole-buffer-per-block copy exact and
        worthwhile.
        """
        if self.use_dynamic_smem:
            return
        cells = self._launch_block()
        if not cells:
            return
        uniform: set[str] = set()
        ok: dict[str, bool] = {}
        depth_seen: dict[str, int] = {}

        def expr_uniform(expr) -> bool:
            if isinstance(expr, Literal):
                return True
            if isinstance(expr, Var):
                return expr.name in uniform
            if isinstance(expr, BinaryExpr):
                return expr_uniform(expr.left) and expr_uniform(expr.right)
            if isinstance(expr, TernaryExpr):
                return (
                    expr_uniform(expr.cond)
                    and expr_uniform(expr.if_true)
                    and expr_uniform(expr.if_false)
                )
            return False

        def visit(body, depth: int) -> None:
            for s in body_stmts(body):
                if isinstance(s, Tile):
                    # Deliberately do NOT add Tile axes to `uniform` -
                    # see the docstring above.
                    visit(s.body, depth)
                elif isinstance(s, Loop):
                    uniform.add(s.axis.name)
                    visit(s.body, depth + 1)
                elif isinstance(s, StridedLoop):
                    if expr_uniform(s.start):
                        uniform.add(s.axis.name)
                    visit(s.body, depth + 1)
                elif isinstance(s, Cond):
                    visit(s.body, depth)
                    visit(s.else_body, depth)
                elif isinstance(s, Load):
                    good = all(expr_uniform(ix) for ix in s.index)
                    ok[s.input] = ok.get(s.input, True) and good
                    depth_seen[s.input] = max(depth_seen.get(s.input, 0), depth)

        visit(self.kop.body, 0)
        budget = SMEM_STATIC_LIMIT - self.smem_total
        for buf, good in ok.items():
            if not good or depth_seen.get(buf, 0) == 0:
                continue
            if buf in self.written or buf not in self.shapes:
                continue
            if self.dtypes.get(buf, "f32") not in ("f32", "float", "float32"):
                continue
            shape = tuple(int(d) for d in self.shapes[buf])
            if not shape:
                continue
            total = 4
            for d in shape:
                total *= d
            if total <= 0 or total > budget:
                continue
            self.cached_buf[buf] = shape
            budget -= total

    def _emit_buffer_cache(self) -> None:
        if not self.cached_buf:
            return
        e = self.e
        cells = self._launch_block() or 1
        tlin = e.env["_tid"]
        for buf, shape in self.cached_buf.items():
            nb = nbytes_of(self.dtypes.get(buf, "f32"))
            total = 1
            for d in shape:
                total *= d
            base = e.new_r()
            e.emit(f"        mov.u32         {base}, _cache_{buf};")
            self.smem_base[buf] = base
            gbase = self.ptrs[buf]
            dt = self.dtypes.get(buf, "f32")
            # Fill the cache with one ld.global.nc.v4.f32 per thread per
            # round instead of a
            # scalar float at a time - 4x fewer loop trips (and address
            # computations) to move the same bytes, with the shared-mem
            # layout and later read pattern untouched (this only changes
            # how the cache gets FILLED, each element still lands at the
            # exact same byte offset it always did - 1:1 with the global
            # layout, no transpose). Safe whenever the buffer is plain f32
            # and its total element count divides evenly by cells*4 (so
            # every thread's 4-wide chunk tiles the buffer with no
            # remainder to special-case); both base pointers are at least
            # 16-byte aligned (the global alloc, and `_cache_*`'s own
            # `.align 16` declared in _pack_smem), so a 16-byte-aligned
            # v4 access is safe. Falls back to the original scalar loop,
            # unchanged, whenever that doesn't divide evenly.
            step = cells * 4 if dt in ("f32", "float", "float32") and total % (cells * 4) == 0 else cells
            i = e.new_r()
            if step != cells:
                tid4 = e.new_r()
                e.emit(f"        mul.lo.s32      {tid4}, {tlin}, 4;")
                e.emit(f"        mov.s32         {i}, {tid4};")
            else:
                e.emit(f"        mov.s32         {i}, {tlin};")
            head = e.label("CACHE")
            end = e.label("CACHEEND")
            e.emit('        .pragma "nounroll";')
            e.emit(f"{head}:")
            p = e.new_p()
            e.emit(f"        setp.ge.s32     {p}, {i}, {total};")
            e.emit(f"        @{p} bra        {end};")
            gaddr = e.new_rd()
            e.emit(f"        mad.wide.s32    {gaddr}, {i}, {nb}, {gbase};")
            saddr = e.new_r()
            e.emit(f"        mad.lo.s32      {saddr}, {i}, {nb}, {base};")
            if step != cells:
                f0, f1, f2, f3 = e.new_f(), e.new_f(), e.new_f(), e.new_f()
                e.emit(f"        ld.global.nc.v4.f32 {{{f0}, {f1}, {f2}, {f3}}}, [{gaddr}];")
                e.emit(f"        st.shared.v4.f32 [{saddr}], {{{f0}, {f1}, {f2}, {f3}}};")
            else:
                val = e.new_f()
                e.emit(f"        ld.global.nc.f32   {val}, [{gaddr}];")
                e.emit(f"        st.shared.f32   [{saddr}], {val};")
            e.emit(f"        add.s32         {i}, {i}, {step};")
            e.emit(f"        bra             {head};")
            e.emit(f"{end}:")
        e.emit("        bar.sync        0;")

    def _lane_axis_for_tile(self, tile: Tile) -> str | None:
        """Name of the Tile axis that `emit_tile` binds to `gid % 32`
        (the hardware lane) for a single-warp block (`block_threads ==
        32`): the axis with extent 32 at the innermost (stride-1)
        position. Returns None whenever the block isn't exactly one
        warp (so there is no risk of a second warp's lanes, or a
        raster sub-grouping that packs more than one axis-group into
        one warp, being mixed into a reduction that assumes full-warp
        uniformity), whenever no axis matches, or whenever `raster`
        would override the plain stride-based binding `emit_tile` uses
        (mirrors the exact gate `emit_tile` itself uses for raster to
        stay a no-op)."""
        if int(tile.block_threads or 0) != 32:
            return None
        extents = [axis_extent(a) for a in tile.axes]
        strides = [1] * len(extents)
        acc = 1
        for i in range(len(extents) - 1, -1, -1):
            strides[i] = acc
            acc *= extents[i]
        if tile.raster_group and tile.raster_axes:
            names = [a.name for a in tile.axes]
            if tile.raster_axes[0] in names and tile.raster_axes[1] in names:
                mi = names.index(tile.raster_axes[0])
                ni = names.index(tile.raster_axes[1])
                if ni == mi + 1 and extents[mi] > 1 and extents[ni] > 1:
                    gi = mi if tile.raster_orient == "m" else ni
                    if min(int(tile.raster_group), extents[gi]) > 1:
                        return None
        candidates = [a.name for i, a in enumerate(tile.axes) if extents[i] == 32 and strides[i] == 1]
        return candidates[0] if len(candidates) == 1 else None

    def _expr_mentions(self, expr, name: str) -> bool:
        """True if `expr` reads `name` anywhere, or is a shape we don't
        specifically recognize (conservative: unknown expression kinds
        are treated as mentioning everything, so the warp-reduce
        rewrite below only ever fires when it can prove the loop body
        is lane-independent, never on a guess)."""
        if isinstance(expr, Literal):
            return False
        if isinstance(expr, Var):
            return expr.name == name
        if isinstance(expr, BinaryExpr):
            return self._expr_mentions(expr.left, name) or self._expr_mentions(expr.right, name)
        if isinstance(expr, TernaryExpr):
            return (
                self._expr_mentions(expr.cond, name)
                or self._expr_mentions(expr.if_true, name)
                or self._expr_mentions(expr.if_false, name)
            )
        if isinstance(expr, CastExpr):
            return self._expr_mentions(expr.value, name)
        return True

    def _is_warp_uniform_dot_reduce(self, loop: Loop, lane_axis: str) -> bool:
        """True iff `loop` (trip count must be exactly 32, the warp
        width) is a pure multiply-add reduction - any number of
        single-output f32 Loads, any number of f32 `multiply` Assigns,
        exactly one `add` Accum with no `base` - and none of its Loads
        index by `lane_axis`. That combination means every lane in the
        (single-warp) block computes the identical accumulated scalar,
        32 times over, redundantly: this is the structural reason a02
        still spent 32 serial iterations per lane even after the
        shared-memory buffer-cache fix (see _plan_buffer_cache). The
        rewrite splits the reduction's 32 iterations one-per-lane and
        combines with shfl.sync.bfly instead of each lane repeating all
        32 serially. See _emit_warp_reduce_loop for the rewrite and
        FEATURES.md for the measured win."""
        if int(axis_extent(loop.axis)) != 32 or loop.axis.name == lane_axis:
            return False
        body = list(body_stmts(loop.body))
        if not body:
            return False
        accum = None
        for s in body:
            if isinstance(s, Load):
                if len(s.names) != 1:
                    return False
                if dtype_name(s.dtype) not in ("f32", "float", "float32"):
                    return False
                if any(self._expr_mentions(ix, lane_axis) for ix in s.index):
                    return False
            elif isinstance(s, Assign):
                if s.op.name != "multiply":
                    return False
                if dtype_name(s.dtype) not in ("f32", "float", "float32"):
                    return False
            elif isinstance(s, Accum):
                if accum is not None or s.op.name != "add" or s.base is not None:
                    return False
                if dtype_name(s.dtype) not in ("f32", "float", "float32"):
                    return False
                accum = s
            else:
                return False
        return accum is not None

    def _plan_warp_reduce_loops(self) -> None:
        """Find every Loop matching _is_warp_uniform_dot_reduce inside a
        single-warp (block_threads == 32) Tile, and record it (by
        object id) in self.warp_reduce_loops so emit_loop rewrites it
        instead of lowering it the ordinary (serial/unrolled) way.
        General: fires for any kernel shaped like a02 (a reduction
        whose trip count happens to equal the warp width and whose
        body never reads by the lane axis), not an a02-specific rule;
        naturally yields nothing for every other kernel in the open
        set, where either the block isn't one warp, no reduction has
        exactly 32 trips, or the reduction's own body is already
        lane-dependent (a genuinely parallel-across-lanes workload,
        not a redundant one)."""

        def visit(body, lane_axis: str | None) -> None:
            for s in body_stmts(body):
                if isinstance(s, Tile):
                    visit(s.body, self._lane_axis_for_tile(s))
                elif isinstance(s, Loop):
                    if lane_axis and self._is_warp_uniform_dot_reduce(s, lane_axis):
                        self.warp_reduce_loops[id(s)] = lane_axis
                    else:
                        visit(s.body, lane_axis)
                elif isinstance(s, Cond):
                    visit(s.body, lane_axis)
                    visit(s.else_body, lane_axis)
                elif isinstance(s, StridedLoop):
                    visit(s.body, lane_axis)

        visit(self.kop.body, None)

    def _emit_warp_reduce_loop(self, loop: Loop, body) -> None:
        """Lower a Loop flagged by _plan_warp_reduce_loops: bind the
        loop's own axis variable directly to this lane's hardware lane
        id (the Tile axis register that _lane_axis_for_tile proved
        equals gid % 32 here), run the body ONCE so each lane computes
        its own single term, then fold the 32 per-lane partials
        together with the standard 5-step shfl.sync.bfly butterfly
        (same instruction shape as emit_warpshuffle) so every lane ends
        up holding the full sum - identical result to the 32-iteration
        serial loop it replaces, in log2(32) shuffles instead of 32
        redundant iterations per lane."""
        e = self.e
        lane_axis = self.warp_reduce_loops[id(loop)]
        var = loop.axis.name
        e.bind(var, e.env[lane_axis], "s32")
        self.stmts(body)
        accum = next(s for s in body_stmts(body) if isinstance(s, Accum))
        reg = e.env[accum.name]
        mask = e.new_r()
        e.emit(f"        activemask.b32  {mask};")
        step = 16
        while step > 0:
            bits = e.new_r()
            e.emit(f"        mov.b32         {bits}, {reg};")
            e.emit(f"        shfl.sync.bfly.b32 {bits}, {bits}, {step}, 31, {mask};")
            shuf = e.new_f()
            e.emit(f"        mov.b32         {shuf}, {bits};")
            nxt = e.new_f()
            e.emit(f"        add.f32         {nxt}, {reg}, {shuf};")
            reg = nxt
            step >>= 1
        e.bind(accum.name, reg, "f32")

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
        self._plan_buffer_cache()
        self._plan_warp_reduce_loops()
        self.ptx_target = "sm_90" if self._uses_sm90() else PTX_TARGET
        e.emit(f".version {PTX_VERSION}")
        e.emit(f".target {self.ptx_target}")
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
            for buf, shape in self.cached_buf.items():
                total = 1
                for d in shape:
                    total *= d
                nb = nbytes_of(self.dtypes.get(buf, "f32"))
                e.emit(f".shared .align 16 .b8 _cache_{buf}[{total * nb}];")
            if smems or self.cached_buf:
                e.emit("")
        param_names = list(self.args)
        for name in self._tma_desc_names():
            if name not in param_names:
                param_names.append(name)
            self.tma_descs.add(name)
        params = ",\n".join(
            f"        .param .{self._param_ty(name)} param_{name}" for name in param_names
        )
        e.emit(f".visible .entry {self.kop.name}(")
        e.emit(params)
        e.emit(")")
        req = self._reqntid_clause()
        if req:
            e.emit(f".reqntid {req}")
        e.emit("{")
        body_at = len(e.lines)
        self.prologue()
        self.stmts(self.kop.body)
        e.emit("        ret;")
        e.emit("}")
        decls = [
            f"        .reg .pred      %p<{max(self.e.np, 2)}>;",
            f"        .reg .f32       %f<{max(self.e.nf, 2)}>;",
        ]
        if self.e.nd > 1:
            decls.append(f"        .reg .f64       %fd<{self.e.nd}>;")
        decls += [
            f"        .reg .b32       %r<{max(self.e.nr, 2)}>;",
            f"        .reg .b64       %rd<{max(self.e.nrd, 2)}>;",
            "",
        ]
        e.lines[body_at:body_at] = decls
        # Do not emit .maxnreg: ptxas already packs (a01: 48 regs without it, 64 with .maxnreg 168).
        return "\n".join(e.lines) + "\n"

    def _launch_block(self) -> int | None:
        for s in walk_stmts(self.kop.body):
            if isinstance(s, Tile) and s.block_threads:
                return int(s.block_threads)
        block = self.meta.get("block")
        if block:
            prod = 1
            for x in block:
                prod *= int(x)
            return prod
        return None

    def _reqntid_clause(self) -> str | None:
        block = self.meta.get("block")
        if block and len(block) >= 2 and any(int(x) > 1 for x in block[1:]):
            vals = [int(x) for x in list(block) + [1, 1]][:3]
            return ", ".join(str(v) for v in vals)
        bt = self._launch_block()
        # 1-warp .reqntid makes ptxas spend 80 regs on ILP (a02). Occupancy
        # is already block-limited, so skip the clause and let it pack.
        if bt == 32:
            return None
        return str(bt) if bt else None

    def _is_scalar(self, name: str) -> bool:
        for arg in self.meta.get("args") or []:
            if arg.get("name") == name and arg.get("scalar"):
                return True
        return False

    def _param_ty(self, name: str) -> str:
        if self._is_scalar(name):
            dt = self.dtypes.get(name, "f32")
            return ptx_type(dt)
        return "u64"

    def _load_op(self, space: str, pty: str, buf: str, vec: int = 1) -> str:
        cache = ".nc" if space == "global" and buf not in self.written else ""
        v = f".v{vec}" if vec > 1 else ""
        return f"ld.{space}{cache}{v}.{pty}"

    def _uses_sm90(self) -> bool:
        hopper = (
            TmaDescriptor, TmaLoad, MbarrierInit, MbarrierArrive, MbarrierArriveExpectTx,
            MbarrierWait, SetMaxNReg, WgmmaDescriptor, WgmmaMma, WgmmaFence, WgmmaCommit, WgmmaWait,
        )
        return any(isinstance(s, hopper) for s in walk_stmts(self.kop.body))

    def _tma_desc_names(self) -> list[str]:
        return [s.name for s in walk_stmts(self.kop.body) if isinstance(s, TmaDescriptor)]

    def _dim3(self, key: str) -> list[int]:
        v = self.meta.get(key) or [1, 1, 1]
        vals = [int(x) for x in list(v) + [1, 1]][:3]
        return vals

    def _launch_1d(self) -> bool:
        return all(x == 1 for x in self._dim3("block")[1:]) and all(x == 1 for x in self._dim3("grid")[1:])

    def _needs_lane(self) -> bool:
        return any(isinstance(s, (WarpShuffle, TreeHalve, LdmatrixLoad, MmaSyncPtx, RegStore)) for s in walk_stmts(self.kop.body))

    def prologue(self) -> None:
        e = self.e
        for name in self.args:
            if self._is_scalar(name):
                dt = self.dtypes.get(name, "f32")
                pty = ptx_type(dt)
                if dt in ("f32", "float", "float32"):
                    r = e.new_f()
                    e.emit(f"        ld.param.f32    {r}, [param_{name}];")
                    e.bind(name, r, "f32")
                elif dt in ("f64", "double"):
                    r = e.new_d()
                    e.emit(f"        ld.param.f64    {r}, [param_{name}];")
                    e.bind(name, r, "f64")
                else:
                    r = e.new_r()
                    e.emit(f"        ld.param.{pty}    {r}, [param_{name}];")
                    e.bind(name, r, "s32")
                continue
            raw = e.new_rd()
            e.emit(f"        ld.param.u64    {raw}, [param_{name}];")
            if name in self.smem:
                e.emit(f"        cvta.to.shared.u64     {raw}, {raw};")
            else:
                e.emit(f"        cvta.to.global.u64     {raw}, {raw};")
            self.ptrs[name] = raw
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
        if self._launch_1d():
            e.emit(f"        mad.lo.s32      {gid}, {cta}, {ntid}, {tid};")
            ncta = e.new_r()
            e.emit(f"        mov.u32         {ncta}, %nctaid.x;")
            tlin, nprod, clin = tid, ntid, cta
            z = e.imm_s32(0)
            one = e.imm_s32(1)
            e.bind("threadIdx.y", z, "s32")
            e.bind("threadIdx.z", z, "s32")
            e.bind("blockDim.y", one, "s32")
            e.bind("blockDim.z", one, "s32")
            e.bind("blockIdx.y", z, "s32")
            e.bind("blockIdx.z", z, "s32")
        else:
            tidy = e.new_r()
            tidz = e.new_r()
            ntidy = e.new_r()
            ntidz = e.new_r()
            nctax = e.new_r()
            nctay = e.new_r()
            ctay = e.new_r()
            ctaz = e.new_r()
            tlin = e.new_r()
            clin = e.new_r()
            nprod = e.new_r()
            e.emit(f"        mov.u32         {tidy}, %tid.y;")
            e.emit(f"        mov.u32         {tidz}, %tid.z;")
            e.emit(f"        mov.u32         {ntidy}, %ntid.y;")
            e.emit(f"        mov.u32         {ntidz}, %ntid.z;")
            e.emit(f"        mov.u32         {ctay}, %ctaid.y;")
            e.emit(f"        mov.u32         {ctaz}, %ctaid.z;")
            e.emit(f"        mov.u32         {nctax}, %nctaid.x;")
            e.emit(f"        mov.u32         {nctay}, %nctaid.y;")
            tmp = e.new_r()
            e.emit(f"        mad.lo.s32      {tmp}, {tidz}, {ntidy}, {tidy};")
            e.emit(f"        mad.lo.s32      {tlin}, {tmp}, {ntid}, {tid};")
            e.emit(f"        mad.lo.s32      {tmp}, {ctaz}, {nctay}, {ctay};")
            e.emit(f"        mad.lo.s32      {clin}, {tmp}, {nctax}, {cta};")
            e.emit(f"        mul.lo.s32      {tmp}, {ntid}, {ntidy};")
            e.emit(f"        mul.lo.s32      {nprod}, {tmp}, {ntidz};")
            e.emit(f"        mad.lo.s32      {gid}, {clin}, {nprod}, {tlin};")
            nctaz = e.new_r()
            ncta = e.new_r()
            e.emit(f"        mov.u32         {nctaz}, %nctaid.z;")
            e.emit(f"        mul.lo.s32      {ncta}, {nctax}, {nctay};")
            e.emit(f"        mul.lo.s32      {ncta}, {ncta}, {nctaz};")
            e.bind("threadIdx.y", tidy, "s32")
            e.bind("threadIdx.z", tidz, "s32")
            e.bind("blockDim.y", ntidy, "s32")
            e.bind("blockDim.z", ntidz, "s32")
            e.bind("blockIdx.y", ctay, "s32")
            e.bind("blockIdx.z", ctaz, "s32")
        e.bind("_tid", tlin, "s32")
        e.bind("_ntid", nprod, "s32")
        e.bind("_cta", clin, "s32")
        e.bind("_ncta", ncta, "s32")
        e.bind("_gid", gid, "s32")
        e.bind("threadIdx.x", tid, "s32")
        e.bind("blockDim.x", ntid, "s32")
        e.bind("blockIdx.x", cta, "s32")
        for name in self.tma_descs:
            raw = e.new_rd()
            e.emit(f"        ld.param.u64    {raw}, [param_{name}];")
            self.ptrs[name] = raw
        if self._needs_lane():
            # Lane/warp must come from the linearized LOCAL thread index
            # (tlin), not raw %tid.x: CUDA packs threads into warps of 32
            # by threadIdx.z*ntid.y*ntid.x + threadIdx.y*ntid.x + threadIdx.x.
            # tlin == tid for every 1D launch (the baseline + reduction/
            # matmul/attention kernels), so this changes nothing there; it
            # only matters for multi-row tensor-core blocks (ntid.y/z > 1).
            lane = e.new_r()
            warp = e.new_r()
            e.emit(f"        and.b32         {lane}, {tlin}, 31;")
            e.emit(f"        shr.u32         {warp}, {tlin}, 5;")
            e.bind("lane", lane, "s32")
            e.bind("warp", warp, "s32")
        self._emit_buffer_cache()

    def invalidate(self, name: str) -> None:
        pat = re.compile(rf"(^|[^A-Za-z0-9_]){re.escape(name)}([^A-Za-z0-9_]|$)")
        if self.cse:
            self.cse = {k: v for k, v in self.cse.items() if not pat.search(k)}
        if self.mem:
            self.mem = {k: v for k, v in self.mem.items() if not pat.search(k)}

    def _push_flow(self) -> tuple[dict, dict, dict, dict]:
        return dict(self.cse), dict(self.mem), dict(self.arith), dict(self.addr_cse)

    def _pop_flow(self, saved: tuple[dict, dict, dict, dict]) -> None:
        self.cse, self.mem, self.arith, self.addr_cse = saved

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
            if isinstance(s, Write) and self._try_vec_store(s, sl[i + 1 :], done):
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

        def base_off(e):
            # `base + k` for a literal k decomposes to (base text, k); a
            # bare expression (no "+ literal" at the top) is its own base
            # at offset 0. This makes x[k] vs x[k+1] match for ANY k, not
            # just k == 0 — the IR emits "(idx + 2)", "(idx + 3)", ... for
            # every unrolled copy, not only the first pair.
            if isinstance(e, BinaryExpr) and e.op == "+" and lit_int(e.right) is not None:
                return e.left.pretty(), lit_int(e.right)
            return e.pretty(), 0

        base_a, off_a = base_off(a)
        base_b, off_b = base_off(b)
        return base_a == base_b and off_b == off_a + 1

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
        space = "shared" if s.input in self.smem or s.input in self.cached_buf else "global"
        addr = self.addr_of(s.input, s.index, space)
        regs = [self.e.new_f() for _ in chain]
        packed = ", ".join(regs)
        if space == "global" and self.active:
            z = self.e.imm_f32(0.0)
            for r in regs:
                self.e.emit(f"        mov.f32         {r}, {z};")
        self.e.emit(f"{self._at(space)}{self._load_op(space, 'f32', s.input, vec)} {{{packed}}}, [{addr}];")
        for load, reg in zip(chain, regs):
            name = list(load.names)[0]
            self.e.bind(name, reg, "f32")
            idx = ", ".join(ix.pretty() for ix in load.index)
            self.mem[f"{load.input}[{idx}]|1"] = (reg,)
            done.add(id(load))
        self.stats["vec_loads"] += 1
        return True

    def _try_vec_store(self, s: Write, rest, done: set[int]) -> bool:
        if s.atomic or len(list(s.values)) != 1:
            return False
        dt = dtype_name(s.value_dtype) or self.dtypes.get(s.output, "f32")
        if dt not in ("f32", "float", "float32"):
            return False
        if s.output in self.smem:
            return False
        chain = [s]
        last = s
        # A value-producing Assign/Accum between two Writes (e.g. the
        # reciprocal+multiply that rescales each attention output column)
        # must not block vectorizing those Writes: run it now so the next
        # Write's value is bound, then keep scanning for the vector match.
        pending: list = []
        for t in rest:
            if isinstance(t, (Assign, Accum)) and id(t) not in done:
                pending.append(t)
                continue
            if not isinstance(t, Write) or t.atomic or t.output != s.output:
                break
            if len(list(t.values)) != 1:
                break
            if not self._consecutive_index(last.index, t.index):
                break
            for p in pending:
                self.stmt(p)
                done.add(id(p))
            pending = []
            chain.append(t)
            last = t
            if len(chain) == 4:
                break
        vec = 4 if len(chain) >= 4 else 2 if len(chain) >= 2 else 0
        if vec == 0:
            return False
        chain = chain[:vec]
        addr = self.addr_of(s.output, s.index, "global")
        regs = [self.as_f32(list(w.values)[0]) for w in chain]
        packed = ", ".join(regs)
        self.e.emit(f"{self._at('global')}st.global.v{vec}.f32 [{addr}], {{{packed}}};")
        for w in chain:
            done.add(id(w))
        self._kill_mem(s.output)
        self.stats["vec_stores"] = self.stats.get("vec_stores", 0) + 1
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

    def _body_touches_global(self, body) -> bool:
        for s in walk_stmts(body):
            if isinstance(s, Load) and s.input not in self.smem and s.input not in self.cached_buf:
                return True
            if isinstance(s, Write) and s.output not in self.smem:
                return True
            if isinstance(s, (CpAsyncCopy, TmaLoad, RegStore)):
                return True
        return False

    def _body_global_bufs(self, body) -> set[str]:
        bufs: set[str] = set()
        for s in walk_stmts(body):
            if isinstance(s, Load) and s.input not in self.smem and s.input not in self.cached_buf:
                bufs.add(s.input)
            if isinstance(s, Write) and s.output not in self.smem:
                bufs.add(s.output)
        return bufs

    def _should_unroll(self, loop, end: int, body) -> bool:
        # Measured on 64 compilers: full-unroll of global bodies blows
        # attention (a02: 80 regs vs 48). Full-unroll of smem bodies is how
        # the fastest matmul entries get their FMA count.
        if end <= 0 or self._has_nested_loop(body):
            return False
        if not self._body_is_straight(body):
            return False
        ops = max(self._stmt_count(body), 1)
        if ops * end > UNROLL_EXPANSION_CAP:
            return False
        if self._body_touches_global(body):
            # A loop with a SINGLE remaining global buffer (the rest
            # already cached in shared memory, e.g. attention's Q once K
            # is staged) only keeps one live address register per
            # unrolled copy - the register blow-up that made wide global
            # unrolling a loss (measured: a02 at 80 regs) came from many
            # DISTINCT buffers needing independent address arithmetic
            # live at once, not from unrolling itself. Measured on the
            # real GPU runner: unrolling a02's 32-trip inner loop once Q
            # is the only global read left is faster (22us vs 37us), not
            # a register/occupancy regression.
            if len(self._body_global_bufs(body)) <= 1:
                return end <= 32
            return end <= 4
        return end <= 64

    def _partial_factor(self, end: int, body) -> int:
        # Global partial-unroll of a body with NO nested loop kept a02 at
        # 80 regs (8 live addr pairs) before the warp-reduce pass existed -
        # occupancy from a rolled inner product beat the extra ILP, so this
        # used to unconditionally return 1. First attempt at reviving it
        # after the warp-reduce pass shrank a02's body (30 vs 24 regs
        # looked promising) silently corrupted results (max abs err ~3.9 on
        # the real GPU, not ~1e-7): emit_loop's factor>1 path calls
        # self.stmts(body) more than once per *Python* codegen pass while
        # the result still lives inside ONE physical PTX loop, which
        # re-enters any nested reduce Loop's own emit_loop/seed_accums more
        # than once - seed_accums's "already bound -> don't reseed" guard
        # (correct for the OUTER carried softmax state, which must persist
        # across real loop trips) then also wrongly skipped reseeding the
        # INNER reduction's own accumulator between copies. Fixed via
        # `_reseed_nested_accums` (emit_loop explicitly un-binds the
        # nested accumulator between copies so it genuinely restarts), but
        # kept this factor narrowly gated: only a nested loop already
        # proven lane-independent and reseed-safe by the warp-reduce pass
        # (`self.warp_reduce_loops`) is eligible, with at most 2 live
        # global buffers per copy (mirrors _should_unroll's own measured
        # global-unroll gate) and `end` dividing evenly so there is no
        # ragged tail to special-case. Measured on the real GPU runner:
        # factor=8 here (a02's 128-trip outer loop) got the numerics right
        # (err ~1e-7) and took a02 from 38us to ~25us post-warp-reduce,
        # then ~20-22us with address reuse. Byte-identical PTX
        # on every other open kernel - none of their nested loops are in
        # `self.warp_reduce_loops`, so this stays a no-op for them.
        if not self._has_nested_loop(body):
            return 1
        nested = [s for s in body_stmts(body) if isinstance(s, Loop)]
        if len(nested) != 1 or id(nested[0]) not in self.warp_reduce_loops:
            return 1
        if len(self._body_global_bufs(body)) > 2:
            return 1
        ops = max(self._stmt_count(body), 1)
        for factor in (8, 4, 2):
            if end % factor == 0 and ops * factor <= UNROLL_EXPANSION_CAP:
                return factor
        return 1

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
        # One IR node kind -> one emit_*. Unknown kinds exit 2 (never silent).
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
        elif isinstance(s, Select):
            self.emit_select(s)
        elif isinstance(s, Pre):
            self.emit_pre(s)
        elif isinstance(s, Carry):
            self.emit_carry(s)
        elif isinstance(s, ZeroPrologue):
            self.emit_zero_prologue(s)
        elif isinstance(s, TmaDescriptor):
            return
        elif isinstance(s, TmaLoad):
            self.emit_tma_load(s)
        elif isinstance(s, MbarrierInit):
            self.emit_mbar_init(s)
        elif isinstance(s, MbarrierArriveExpectTx):
            self.emit_mbar_expect(s)
        elif isinstance(s, MbarrierArrive):
            self.emit_mbar_arrive(s)
        elif isinstance(s, MbarrierWait):
            self.emit_mbar_wait(s)
        elif isinstance(s, SetMaxNReg):
            self.emit_setmaxnreg(s)
        elif isinstance(s, RegFragment):
            self.emit_reg_fragment(s)
        elif isinstance(s, LdmatrixLoad):
            self.emit_ldmatrix(s)
        elif isinstance(s, BlockScaleLoad):
            self.emit_block_scale(s)
        elif isinstance(s, MmaSyncPtx):
            self.emit_mma_sync(s)
        elif isinstance(s, FragmentPromote):
            self.emit_frag_promote(s)
        elif isinstance(s, FragmentApply):
            self.emit_frag_apply(s)
        elif isinstance(s, FragmentRowReduce):
            self.emit_frag_row_reduce(s)
        elif isinstance(s, FragmentRepack):
            self.emit_frag_repack(s)
        elif isinstance(s, RegStore):
            self.emit_reg_store(s)
        elif isinstance(s, WgmmaDescriptor):
            self.emit_wgmma_desc(s)
        elif isinstance(s, WgmmaMma):
            self.emit_wgmma_mma(s)
        elif isinstance(s, WgmmaFence):
            self.e.emit("        wgmma.fence.sync.aligned;")
        elif isinstance(s, WgmmaCommit):
            self.e.emit("        wgmma.commit_group.sync.aligned;")
        elif isinstance(s, WgmmaWait):
            self.e.emit(f"        wgmma.wait_group.sync.aligned {int(s.group)};")
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
        prev_active = self.active
        # Collective ops need every non-exited thread. A bra-skip around
        # bar.sync is a Volta+ deadlock. Predicate global ld/st instead.
        collective = any(
            isinstance(s, (Sync, TreeHalve, WarpShuffle, CpAsyncCopy, CpAsyncCommit, CpAsyncWait,
                           MbarrierInit, MbarrierArrive, MbarrierArriveExpectTx, MbarrierWait,
                           TmaLoad, SetMaxNReg, WgmmaFence, WgmmaCommit, WgmmaWait,
                           LdmatrixLoad, MmaSyncPtx))
            for s in walk_stmts(tile.body)
        )
        if guard and collective:
            p = e.new_p()
            e.emit(f"        setp.lt.s32     {p}, {gid}, {n};")
            self.active = p
        elif guard:
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
                    if extents[gi] > 1 and extents[oi] > 1 and g_eff > 1:
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
            if ext == 1:
                e.bind(axis.name, "0", "s32")
                continue
            term = gid if s == 1 else self.div_imm(gid, s)
            if i == 0:
                e.bind(axis.name, term, "s32")
            else:
                e.bind(axis.name, self.rem_imm(term, ext), "s32")
        self.stmts(tile.body)
        self.active = prev_active
        if skip:
            e.emit(f"{skip}:")
            if saved is not None:
                self._pop_flow(saved)

    def _arith(self, key: str, emit) -> str:
        hit = self.arith.get(key)
        if hit is not None:
            self.stats["cse_hits"] += 1
            return hit
        val = emit()
        self.arith[key] = val
        return val

    def mul_imm(self, src: str, imm: int) -> str:
        si = parse_imm(src)
        if si is not None:
            return str(si * imm)
        if imm == 0:
            return "0"
        if imm == 1:
            return src
        if imm == -1:
            return self._arith(f"neg:{src}", lambda: self._neg_s32(src))
        return self._arith(f"{src}*{imm}", lambda: self._mul_imm_emit(src, imm))

    def _neg_s32(self, src: str) -> str:
        d = self.e.new_r()
        self.e.emit(f"        neg.s32         {d}, {src};")
        return d

    def _mul_imm_emit(self, src: str, imm: int) -> str:
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
        return self._arith(f"{src}/{imm}", lambda: self._div_imm_emit(src, imm))

    def _div_imm_emit(self, src: str, imm: int) -> str:
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
        if imm == 1:
            return "0"
        return self._arith(f"{src}%{imm}", lambda: self._rem_imm_emit(src, imm))

    def _rem_imm_emit(self, src: str, imm: int) -> str:
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
        return self._arith(f"{a}*{imm}+{c}", lambda: self._mad_imm_emit(a, imm, c))

    def _mad_imm_emit(self, a: str, imm: int, c: str) -> str:
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
        key = f"{a}+{b}" if str(a) <= str(b) else f"{b}+{a}"
        return self._arith(key, lambda: self._add_s32_emit(a, b, ia, ib))

    def _add_s32_emit(self, a: str, b: str, ia, ib) -> str:
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

    def _reseed_nested_accums(self, body) -> list[str]:
        """Names of Accum statements that live directly inside a nested
        Loop/StridedLoop marked `seed=True` one level under `body`
        (body -> inner.body -> Accum). `emit_loop`'s factor>1 path (see
        `_partial_factor`) calls `self.stmts(body)` more than once per
        *Python* codegen pass while the result still lives inside ONE
        physical PTX loop; each of those calls re-enters the nested
        Loop's own `emit_loop`, which re-runs `seed_accums` - but that
        guard is "skip if already bound", which is correct for an OUTER
        accumulator that must persist across real loop trips (seeded
        once, before this factor-loop even starts, by the ordinary
        `self.seed_accums(body, ...)` call above - that call only reads
        body's own top-level statements, so it's untouched by this), and
        WRONG for an inner reduction that must restart at its identity
        value on every one of these replicated copies. Used by
        `emit_loop` to explicitly un-bind those names between copies so
        the next one reseeds for real instead of inheriting the
        previous copy's final sum (the exact bug that made an earlier
        attempt at this measure ~3.9 max-abs-err on the real GPU - see
        the comment on `_partial_factor`)."""
        names: list[str] = []
        for s in body_stmts(body):
            if isinstance(s, (Loop, StridedLoop)) and getattr(s, "seed", False):
                for inner in body_stmts(s.body):
                    if isinstance(inner, Accum) and inner.name not in names:
                        names.append(inner.name)
        return names

    def emit_loop(self, loop: Loop, body_override=None, seed_any: bool = False) -> None:
        e = self.e
        body = body_override if body_override is not None else loop.body
        self.seed_accums(body, True if seed_any else loop.seed)
        var = loop.axis.name
        end = axis_extent(loop.axis)
        if end <= 0:
            return
        if body_override is None and id(loop) in self.warp_reduce_loops:
            self._emit_warp_reduce_loop(loop, body)
            return
        if self._should_unroll(loop, end, body):
            self.stats["unrolled_iters"] += end
            for k in range(end):
                e.bind(var, str(k), "s32")
                self.stmts(body)
            return
        factor = self._partial_factor(end, body)
        i = e.mov_s32(0)
        saved = self._push_flow()
        head = e.label("LOOP")
        if self._body_touches_global(body):
            e.emit('        .pragma "nounroll";')
        e.emit(f"{head}:")
        if factor > 1:
            self.stats["unrolled_iters"] += factor
            reseed = self._reseed_nested_accums(body)
            saved_subst = self._unroll_subst
            for k in range(factor):
                if k > 0:
                    for name in reseed:
                        e.env.pop(name, None)
                        self.invalidate(name)
                if k == 0:
                    e.bind(var, i, "s32")
                else:
                    iv = e.new_r()
                    e.emit(f"        add.s32         {iv}, {i}, {k};")
                    e.bind(var, iv, "s32")
                self._unroll_subst = (var, i, k)
                self.stmts(body)
            self._unroll_subst = saved_subst
            e.emit(f"        add.s32         {i}, {i}, {factor};")
        else:
            e.bind(var, i, "s32")
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
        if self._body_touches_global(loop.body):
            e.emit('        .pragma "nounroll";')
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

    def _at(self, space: str) -> str:
        if space == "global" and self.active:
            return f"        @{self.active} "
        return "        "

    def emit_load(self, s: Load) -> None:
        dt = dtype_name(s.dtype) or self.dtypes.get(s.input, "f32")
        pty = ptx_type(dt)
        space = "shared" if s.input in self.smem or s.input in self.cached_buf else "global"
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
            if space == "global" and self.active:
                z = self.e.imm_f32(0.0)
                for r in regs:
                    self.e.emit(f"        mov.f32         {r}, {z};")
            self.e.emit(f"{self._at(space)}{self._load_op(space, pty, s.input, vec)} {{{packed}}}, [{addr}];")
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
                if space == "global" and self.active:
                    self.e.emit(f"        mov.f32         {reg}, {self.e.imm_f32(0.0)};")
                self.e.emit(f"{self._at(space)}{self._load_op(space, pty, s.input)}   {reg}, [{addr}];")
                self.e.bind(name, reg, "f32")
                regs_out.append(reg)
            elif dt in ("f16", "half"):
                tmp = self.e.new_r()
                dest = self.e.new_f()
                self.e.emit(f"{self._at(space)}{self._load_op(space, 'f16', s.input)}    {tmp}, [{addr}];")
                self.e.emit(f"        cvt.f32.f16      {dest}, {tmp};")
                self.e.bind(name, dest, "f32")
                regs_out.append(dest)
            elif dt in ("bf16",):
                tmp = self.e.new_r()
                dest = self.e.new_f()
                self.e.emit(f"{self._at(space)}{self._load_op(space, 'bf16', s.input)}   {tmp}, [{addr}];")
                self.e.emit(f"        cvt.f32.bf16     {dest}, {tmp};")
                self.e.bind(name, dest, "f32")
                regs_out.append(dest)
            elif dt in ("f64", "double"):
                dest = self.e.new_d()
                self.e.emit(f"{self._at(space)}{self._load_op(space, 'f64', s.input)}    {dest}, [{addr}];")
                self.e.bind(name, dest, "f64")
                regs_out.append(dest)
            else:
                kind = "s32"
                reg = self.e.new_r()
                self.e.emit(f"{self._at(space)}{self._load_op(space, pty, s.input)}   {reg}, [{addr}];")
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
            self.e.emit(f"{self._at(space)}red.{space}.add.f32 [{addr}], {src};")
            return
        vec = n if n in (2, 4) and dt in ("f32", "float", "float32") and not s.atomic else 1
        if vec > 1:
            addr = self.addr_of(s.output, s.index, space)
            regs = [self.as_f32(v) for v in vals]
            packed = ", ".join(regs)
            self.e.emit(f"{self._at(space)}st.{space}.v{vec}.{pty} [{addr}], {{{packed}}};")
            return
        for k, name in enumerate(vals):
            idx = s.index
            if k:
                last = BinaryExpr("+", s.index[-1], Literal(k, "int"))
                idx = tuple(s.index[:-1]) + (last,)
            addr = self.addr_of(s.output, idx, space)
            if dt in ("f32", "float", "float32"):
                self.e.emit(f"{self._at(space)}st.{space}.{pty}   [{addr}], {self.as_f32(name)};")
            elif dt in ("f16", "half"):
                bits = self.e.new_r()
                self.e.emit(f"        cvt.rn.f16.f32   {bits}, {self.as_f32(name)};")
                self.e.emit(f"{self._at(space)}st.{space}.f16   [{addr}], {bits};")
            elif dt in ("bf16",):
                bits = self.e.new_r()
                self.e.emit(f"        cvt.rn.bf16.f32  {bits}, {self.as_f32(name)};")
                self.e.emit(f"{self._at(space)}st.{space}.bf16  [{addr}], {bits};")
            elif dt in ("f64", "double"):
                self.e.emit(f"{self._at(space)}st.{space}.f64   [{addr}], {self.as_f64(name)};")
            else:
                self.e.emit(f"{self._at(space)}st.{space}.{pty}   [{addr}], {self.as_s32(name)};")

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
        pred = f"@{self.active} " if self.active else ""
        e.emit(f"        {pred}cp.async.{q}.shared.global [{dst}], [{src}], {n};")
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

    def _linear_terms(self, expr) -> tuple[dict, int] | None:
        """Flatten a +/-/neg tree of index-arithmetic into {leaf: coeff} +
        const, so `(base + off) - base` cancels instead of round-tripping
        through registers. Never descends into `*`, `/`, casts, or
        comparisons — those stay opaque leaves, so this cannot change
        multiply/divide semantics. Returns None when nothing would change
        (a single leaf, coeff 1, const 0) so the caller's normal path and
        CSE keys are untouched for every expression this cannot help.
        """
        terms: dict[str, tuple[int, object]] = {}
        const = 0

        def walk(e, sign: int) -> bool:
            nonlocal const
            if isinstance(e, Literal) and not isinstance(e.value, bool):
                if e.dtype in ("float", "f32") or isinstance(e.value, float) and e.value != int(e.value):
                    return False
                const += sign * int(e.value)
                return True
            if isinstance(e, BinaryExpr) and e.op == "+":
                return walk(e.left, sign) and walk(e.right, sign)
            if isinstance(e, BinaryExpr) and e.op == "-":
                return walk(e.left, sign) and walk(e.right, -sign)
            try:
                key = e.pretty()
            except Exception:
                return False
            coeff, _ = terms.get(key, (0, e))
            terms[key] = (coeff + sign, e)
            return True

        if not (isinstance(expr, BinaryExpr) and expr.op in ("+", "-")):
            return None
        if not walk(expr, 1):
            return None
        live = {k: v for k, v in terms.items() if v[0] != 0}
        if len(live) == 1 and const == 0 and next(iter(live.values()))[0] == 1:
            return None
        return live, const

    def _index_s32(self, expr) -> str:
        lin = self._linear_terms(expr)
        if lin is None:
            return self.expr_s32(expr)
        terms, const = lin
        acc = None
        for coeff, rep in terms.values():
            reg = self.expr_s32(rep)
            term = self.mul_imm(reg, coeff)
            acc = term if acc is None else self.add_s32(acc, term)
        if acc is None:
            return str(const)
        return self.add_s32(acc, str(const)) if const else acc

    def _as_var_times_const(self, expr, uvar: str) -> int | None:
        """`int` if `expr` is exactly `Var(uvar)*CONST` or `CONST*Var(uvar)`,
        else None. Used only to recognize the one extra multiplicative
        shape `_linear_terms` deliberately leaves opaque (it never
        descends into `*`), so this still can't change multiply/divide
        semantics - it only ever *reads* a coefficient, never rewrites one."""
        if isinstance(expr, BinaryExpr) and expr.op == "*":
            li, ri = lit_int(expr.left), lit_int(expr.right)
            if isinstance(expr.left, Var) and expr.left.name == uvar and ri is not None:
                return ri
            if isinstance(expr.right, Var) and expr.right.name == uvar and li is not None:
                return li
        return None

    def _scalar_var_coeff(self, expr, uvar: str) -> int | None:
        """Coefficient of `Var(uvar)` within one index dimension's
        expression - 1 for a bare `Var(uvar)`, `coeff` for
        `Var(uvar)*coeff` (or the reverse), the sum of such across a
        +/- tree via `_linear_terms`, or 0 if `uvar` provably doesn't
        appear at all. None means `uvar` appears somewhere this can't
        prove a constant per-step stride for (e.g. inside a `*` with a
        non-constant, or more than once with different multipliers) -
        the caller (`_unroll_addr_coeff`) always treats None as "give
        up, fall back to the ordinary full address recompute", never as
        a 0."""
        if isinstance(expr, Var):
            return 1 if expr.name == uvar else 0
        mult = self._as_var_times_const(expr, uvar)
        if mult is not None:
            return mult
        lin = self._linear_terms(expr)
        if lin is None:
            return 0 if not self._expr_mentions(expr, uvar) else None
        terms, _const = lin
        found = 0
        for coeff, rep in terms.values():
            if isinstance(rep, Var) and rep.name == uvar:
                found += coeff
                continue
            m = self._as_var_times_const(rep, uvar)
            if m is not None:
                found += coeff * m
            elif self._expr_mentions(rep, uvar):
                return None
        return found

    def _unroll_addr_coeff(self, buf: str, indices, uvar: str) -> int | None:
        """Element-unit stride that incrementing `uvar` by 1 adds to
        `buf`'s flattened index, mirroring `flatten`'s own row-major
        (last-index-fastest) stride arithmetic but purely symbolically -
        no registers, no code emitted - so `addr_of` can decide, before
        computing anything, whether copy `k`'s address for this exact
        buf+indices is simply copy 0's address plus `k*stride*nbytes`.
        None (give up - safe, just means no shortcut) whenever `uvar`
        appears in more than one index dimension, in a dimension with no
        provable constant coefficient, or `buf`'s shape doesn't line up
        with `indices` the same way `flatten` itself requires."""
        if not indices:
            return None
        shape = self.shapes.get(buf)
        if not (shape and len(shape) == len(indices)) and len(indices) != 1:
            return None
        total = None
        for d, ix in enumerate(indices):
            c = self._scalar_var_coeff(ix, uvar)
            if c is None:
                return None
            if c == 0:
                continue
            if len(indices) == 1:
                stride = 1
            else:
                stride = 1
                for k in range(d + 1, len(shape)):
                    stride *= int(shape[k])
            if total is not None:
                return None
            total = c * stride
        return total

    def flatten(self, buf: str, indices) -> str:
        if not indices:
            return "0"
        shape = self.shapes.get(buf)
        if shape and len(indices) == 2 and len(shape) == 2:
            rec = self._divmod_recompose(indices[0], indices[1], int(shape[1]))
            if rec is not None:
                return self._index_s32(rec)
        if len(indices) == 1:
            return self._index_s32(indices[0])
        if not shape or len(shape) != len(indices):
            acc = self._index_s32(indices[0])
            for ix in indices[1:]:
                acc = self.add_s32(acc, self._index_s32(ix))
            return acc
        acc = None
        for d, ix in enumerate(indices):
            stride = 1
            for k in range(d + 1, len(shape)):
                stride *= int(shape[k])
            term = self._index_s32(ix)
            if acc is None:
                acc = term if stride == 1 else self.mul_imm(term, stride)
            elif stride == 1:
                acc = self.add_s32(acc, term)
            else:
                acc = self.mad_imm(term, stride, acc)
        return acc

    def addr_of(self, buf: str, indices, space: str) -> str:
        subst = self._unroll_subst
        if subst is not None:
            uvar, k0_reg, k = subst
            if k > 0:
                coeff = self._unroll_addr_coeff(buf, indices, uvar)
                if coeff:
                    nb_elem = nbytes_of(self.dtypes.get(buf, "f32"))
                    off = k * coeff * nb_elem
                    saved_env = self.e.env.get(uvar)
                    self.e.env[uvar] = k0_reg
                    self._unroll_subst = None
                    try:
                        base_addr = self.addr_of(buf, indices, space)
                    finally:
                        self._unroll_subst = subst
                        if saved_env is not None:
                            self.e.env[uvar] = saved_env
                        else:
                            del self.e.env[uvar]
                    return f"{base_addr}+{off}" if off else base_addr
        idx = self.flatten(buf, indices)
        nb = nbytes_of(self.dtypes.get(buf, "f32"))
        base = self.smem_base[buf] if space == "shared" else self.ptrs[buf]
        imm = parse_imm(idx)
        if imm is not None:
            off = imm * nb
            if off == 0:
                return base
            return f"{base} + {off}"
        key = (space, base, idx, nb)
        hit = self.addr_cse.get(key)
        if hit is not None:
            self.stats["cse_hits"] += 1
            return hit
        if space == "shared":
            dst = self.e.new_r()
            self.e.emit(f"        mad.lo.s32      {dst}, {idx}, {nb}, {base};")
            self.addr_cse[key] = dst
            return dst
        addr = self.e.new_rd()
        self.e.emit(f"        mad.wide.s32    {addr}, {idx}, {nb}, {base};")
        self.addr_cse[key] = addr
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
        addr = self.mad_imm(idx_reg, nb, base)
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
        addr = self.mad_imm(idx_reg, nb, base)
        self.e.emit(f"        st.shared.f32   [{addr}], {src};")

    def as_f64(self, name: str) -> str:
        if name in self.e.env and self.e.kind.get(name) == "f64":
            return self.e.env[name]
        src = self.as_f32(name)
        d = self.e.new_d()
        self.e.emit(f"        cvt.f64.f32      {d}, {src};")
        return d

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
                "threadIdx.x": "threadIdx.x",
                "threadIdx.y": "threadIdx.y",
                "threadIdx.z": "threadIdx.z",
                "blockDim.x": "blockDim.x",
                "blockDim.y": "blockDim.y",
                "blockDim.z": "blockDim.z",
                "blockIdx.x": "blockIdx.x",
                "blockIdx.y": "blockIdx.y",
                "blockIdx.z": "blockIdx.z",
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
        if isinstance(expr, FlatIndex):
            return "s32", self.flatten(expr.buffer, expr.index)
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

    def _carry_slot(self, name: str, index) -> str:
        if not index:
            return "0"
        imm = lit_int(index[0]) if len(index) == 1 else None
        if imm is not None and all(lit_int(x) == 0 for x in index[1:]):
            return str(imm)
        return "_"

    def emit_pre(self, s: Pre) -> None:
        slot = self._carry_slot(s.carrier, s.index)
        key = (s.carrier, slot)
        if key in self.carry_vals:
            src = self.carry_vals[key]
        elif (s.carrier, "_") in self.carry_vals:
            src = self.carry_vals[(s.carrier, "_")]
        else:
            src = self.e.imm_f32(float(self.carry_seed.get(s.carrier, 0.0)))
        dest = self.e.new_f()
        self.e.emit(f"        mov.f32         {dest}, {src};")
        self.e.bind(s.name, dest, "f32")

    def emit_carry(self, s: Carry) -> None:
        self.carry_seed[s.name] = float(s.seed)
        slot = self._carry_slot(s.name, s.index)
        val = self.as_f32(s.value)
        dest = self.e.new_f()
        self.e.emit(f"        mov.f32         {dest}, {val};")
        self.carry_vals[(s.name, slot)] = dest
        self.e.bind(s.name, dest, "f32")

    def emit_zero_prologue(self, s: ZeroPrologue) -> None:
        words = int(s.words)
        if words <= 0 or s.dst not in self.ptrs:
            return
        e = self.e
        gid = e.env.get("_gid") or e.mov_s32(0)
        ntid = e.env.get("_ntid") or e.mov_s32(1)
        ncta = e.env.get("_ncta") or e.mov_s32(1)
        stride = e.new_r()
        e.emit(f"        mul.lo.s32      {stride}, {ntid}, {ncta};")
        i = e.new_r()
        e.emit(f"        mov.s32         {i}, {gid};")
        head, done = e.label("ZPRO"), e.label("ZPEND")
        e.emit(f"{head}:")
        p = e.new_p()
        e.emit(f"        setp.ge.s32     {p}, {i}, {words};")
        e.emit(f"        @{p} bra        {done};")
        addr = e.new_rd()
        e.emit(f"        mad.wide.s32    {addr}, {i}, 4, {self.ptrs[s.dst]};")
        e.emit(f"        st.global.u32   [{addr}], 0;")
        e.emit(f"        add.s32         {i}, {i}, {stride};")
        e.emit(f"        bra             {head};")
        e.emit(f"{done}:")
        self._kill_mem(s.dst)

    def emit_select(self, s: Select) -> None:
        dest = self.e.new_f()
        taken = self.e.new_p()
        z32 = self.e.imm_s32(0)
        self.e.emit(f"        setp.eq.s32     {taken}, {z32}, 1;")
        first = True
        for br in s.branches:
            p = self.expr_pred(br.select)
            val = self.as_f32(br.value)
            use = self.e.new_p()
            self.e.emit(f"        not.pred        {use}, {taken};")
            self.e.emit(f"        and.pred        {use}, {use}, {p};")
            if first:
                z = self.e.imm_f32(0.0)
                self.e.emit(f"        selp.f32        {dest}, {val}, {z}, {use};")
                first = False
            else:
                self.e.emit(f"        selp.f32        {dest}, {val}, {dest}, {use};")
            nxt = self.e.new_p()
            self.e.emit(f"        or.pred         {nxt}, {taken}, {p};")
            taken = nxt
        self.e.bind(s.name, dest, "f32")

    def _mbar_addr(self, name: str, slot) -> str:
        base = self.smem_base.get(name) or self.e.imm_s32(0)
        if slot is None:
            return str(base)
        off = self.expr_s32(slot)
        addr = self.e.new_r()
        self.e.emit(f"        mad.lo.s32      {addr}, {off}, 8, {base};")
        return addr

    def emit_mbar_init(self, s: MbarrierInit) -> None:
        self.stats["mbarrier"] += 1
        addr = self._mbar_addr(s.mbar, s.slot)
        self.e.emit(f"        mbarrier.init.shared.b64 [{addr}], {int(s.count)};")

    def emit_mbar_expect(self, s: MbarrierArriveExpectTx) -> None:
        self.stats["mbarrier"] += 1
        addr = self._mbar_addr(s.mbar, s.slot)
        self.e.emit(f"        mbarrier.arrive.expect_tx.shared.b64 [{addr}], {int(s.bytes_)};")

    def emit_mbar_arrive(self, s: MbarrierArrive) -> None:
        self.stats["mbarrier"] += 1
        addr = self._mbar_addr(s.mbar, s.slot)
        self.e.emit(f"        mbarrier.arrive.shared.b64 [{addr}];")

    def emit_mbar_wait(self, s: MbarrierWait) -> None:
        self.stats["mbarrier"] += 1
        addr = self._mbar_addr(s.mbar, s.slot)
        phase = self.expr_s32(s.phase)
        p = self.e.new_p()
        spin = self.e.label("MBAR")
        self.e.emit(f"{spin}:")
        self.e.emit(f"        mbarrier.try_wait.parity.shared.b64 {p}, [{addr}], {phase};")
        skip = self.e.new_p()
        self.e.emit(f"        not.pred        {skip}, {p};")
        self.e.emit(f"        @{skip} bra      {spin};")

    def emit_tma_load(self, s: TmaLoad) -> None:
        self.stats["tma"] += 1
        dst = self.addr_of(s.smem, s.smem_index, "shared")
        desc = self.ptrs.get(s.desc, s.desc)
        coords = ", ".join(self.expr_s32(c) for c in s.coords)
        mbar = self._mbar_addr(s.mbar, s.mbar_slot)
        rank = max(len(s.coords), 1)
        self.e.emit(
            f"        cp.async.bulk.tensor.{rank}d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
            f"[{dst}], [{desc}], {{{coords}}}, [{mbar}];"
        )
        self._kill_mem(s.smem)

    def emit_setmaxnreg(self, s: SetMaxNReg) -> None:
        d = "inc" if str(s.direction).lower() in ("inc", "increase", "up") else "dec"
        self.e.emit(f"        setmaxnreg.{d}.sync.aligned.u32 {int(s.count)};")

    def _frag_nregs(self, role: str, shape, dtype: str, nregs) -> int:
        if nregs:
            return int(nregs)
        m, n, k = (int(x) for x in shape[:3]) if shape else (16, 8, 16)
        dt = dtype_name(dtype) if not isinstance(dtype, str) else dtype
        if role.lower() in ("c", "d", "acc"):
            return 4 if (m, n) in ((16, 8), (16, 16)) else 8
        if role.lower() == "a":
            return 4 if dt in ("f16", "bf16", "half") else 8
        return 2 if dt in ("f16", "bf16", "half") else 4

    def emit_reg_fragment(self, s: RegFragment) -> None:
        dt = dtype_name(s.dtype)
        n = self._frag_nregs(s.role, s.shape, dt, s.nregs)
        kind = "f32" if str(s.role).lower() in ("c", "d", "acc") or dt in ("f32", "float") else "b32"
        regs = [self.e.new_f() if kind == "f32" else self.e.new_r() for _ in range(n)]
        self.frags[s.name] = {"role": s.role, "dtype": dt, "regs": regs, "shape": tuple(s.shape), "kind": kind}
        if str(s.role).lower() in ("c", "d", "acc"):
            z = self.e.imm_f32(0.0)
            for r in regs:
                self.e.emit(f"        mov.f32         {r}, {z};")

    def emit_ldmatrix(self, s: LdmatrixLoad) -> None:
        self.stats["ldmatrix"] += 1
        frag = self.frags.get(s.frag)
        if frag is None:
            regs = [self.e.new_r() for _ in range(4)]
            self.frags[s.frag] = {"role": s.role, "dtype": "f16", "regs": regs, "shape": (16, 8, 16), "kind": "b32"}
            frag = self.frags[s.frag]
        space = "shared" if s.src_buffer in self.smem else "global"
        addr = self.addr_of(s.src_buffer, s.src_index, space)
        regs = frag["regs"]
        x = {2: "x2", 4: "x4"}.get(len(regs), "x4")
        packed = ", ".join(regs[: 4 if x == "x4" else 2])
        trans = ".trans" if s.b_trans else ""
        self.e.emit(f"        ldmatrix.sync.aligned.{x}.m8n8{trans}.shared.b16 {{{packed}}}, [{addr}];")

    def emit_block_scale(self, s: BlockScaleLoad) -> None:
        addr = self.addr_of(s.src_buffer, s.src_index, "shared" if s.src_buffer in self.smem else "global")
        dest = self.e.new_r()
        self.e.emit(f"        ld.shared.b8    {dest}, [{addr}];")
        self.frags[s.frag] = {"role": s.role, "dtype": "u8", "regs": [dest], "shape": (1, 1, 1), "kind": "b32"}

    def emit_mma_sync(self, s: MmaSyncPtx) -> None:
        self.stats["mma"] += 1
        a = self.frags.get(s.a_frag, {}).get("regs") or [self.e.new_r() for _ in range(4)]
        b = self.frags.get(s.b_frag, {}).get("regs") or [self.e.new_r() for _ in range(2)]
        c = self.frags.get(s.c_frag, {}).get("regs") or [self.e.new_f() for _ in range(4)]
        m, n, k = (int(x) for x in s.shape[:3])
        layout = "row.col" if not s.b_row_major else "row.row"
        ab, cd = s.ab_dtype, s.c_dtype
        al = ", ".join(a[:4])
        bl = ", ".join(b[:2])
        cl = ", ".join(c[:4])
        self.e.emit(
            f"        mma.sync.aligned.m{m}n{n}k{k}.{layout}.{cd}.{ab}.{ab}.{cd} "
            f"{{{cl}}}, {{{al}}}, {{{bl}}}, {{{cl}}};"
        )

    def emit_frag_promote(self, s: FragmentPromote) -> None:
        src = self.frags.get(s.src, {}).get("regs") or []
        dst_regs = []
        for r in src:
            d = self.e.new_f()
            self.e.emit(f"        cvt.f32.f16      {d}, {r};")
            dst_regs.append(d)
        self.frags[s.dst] = {"role": "c", "dtype": "f32", "regs": dst_regs or [self.e.new_f()], "shape": (16, 8, 16), "kind": "f32"}

    def emit_frag_apply(self, s: FragmentApply) -> None:
        out = self.frags.get(s.out) or self.frags.get(s.args[0] if s.args else "", {})
        regs = list((out or {}).get("regs") or [])
        op = s.op.name if hasattr(s.op, "name") else str(s.op)
        inst = F32_OPS.get(op)
        for i, r in enumerate(regs):
            extras = []
            for a, kind in zip(s.args, s.kinds or ()):
                if kind == "frag" and isinstance(a, str) and a in self.frags:
                    extras.append(self.frags[a]["regs"][min(i, len(self.frags[a]["regs"]) - 1)])
                elif kind in ("uniform", "scalar") and isinstance(a, str):
                    extras.append(self.as_f32(a))
            if inst and extras:
                self.e.emit(f"        {inst:<16}{r}, {r}, {extras[0]};")
            elif op == "relu":
                self.e.emit(f"        max.f32         {r}, {r}, {f32_hex(0.0)};")

    def emit_frag_row_reduce(self, s: FragmentRowReduce) -> None:
        op = s.op.name if hasattr(s.op, "name") else str(s.op)
        inst = F32_OPS.get(op, "add.f32")
        acc_t, acc_b = self.e.new_f(), self.e.new_f()
        z = self.e.imm_f32(0.0)
        self.e.emit(f"        mov.f32         {acc_t}, {z};")
        self.e.emit(f"        mov.f32         {acc_b}, {z};")
        for name in s.frags:
            regs = self.frags.get(name, {}).get("regs") or []
            if len(regs) >= 2:
                self.e.emit(f"        {inst:<16}{acc_t}, {acc_t}, {regs[0]};")
                self.e.emit(f"        {inst:<16}{acc_b}, {acc_b}, {regs[2] if len(regs) > 2 else regs[1]};")
        self.e.bind(s.top, acc_t, "f32")
        self.e.bind(s.bot, acc_b, "f32")

    def emit_frag_repack(self, s: FragmentRepack) -> None:
        regs = []
        for src in s.srcs:
            for r in self.frags.get(src, {}).get("regs") or []:
                bits = self.e.new_r()
                self.e.emit(f"        cvt.rn.f16.f32   {bits}, {r};")
                regs.append(bits)
        self.frags[s.frag] = {"role": "a", "dtype": s.ab_dtype, "regs": regs[:4] or [self.e.new_r()], "shape": (16, 8, 16), "kind": "b32"}

    def emit_reg_store(self, s: RegStore) -> None:
        frag = self.frags.get(s.frag, {})
        regs = frag.get("regs") or []
        space = "shared" if s.dst_buffer in self.smem else "global"
        lane = self.e.env.get("lane") or self.as_s32("lane")
        group = self.e.new_r()
        t = self.e.new_r()
        self.e.emit(f"        shr.u32         {group}, {lane}, 2;")
        self.e.emit(f"        and.b32         {t}, {lane}, 3;")
        # m16n8 C layout: rows group / group+8, cols 2*t / 2*t+1
        pairs = [(0, 0), (0, 1), (8, 0), (8, 1)]
        for i, r in enumerate(regs[:4]):
            ro, co = pairs[i]
            row = self.e.new_r()
            col = self.e.new_r()
            self.e.emit(f"        add.s32         {row}, {group}, {ro};")
            self.e.emit(f"        mad.lo.s32      {col}, {t}, 2, {co};")
            if s.row_base is not None:
                row = self.add_s32(row, self.expr_s32(s.row_base))
            if s.col_base is not None:
                col = self.add_s32(col, self.expr_s32(s.col_base))
            idx = list(s.dst_index) + [None, None]
            # last two coords replaced by computed row/col if 2D buffer
            if len(s.dst_index) >= 2:
                addr = self.addr_of(s.dst_buffer, tuple(list(s.dst_index[:-2]) + [Var("__row"), Var("__col")]) if False else s.dst_index, space)
            nb = nbytes_of(self.dtypes.get(s.dst_buffer, "f32"))
            base = self.ptrs.get(s.dst_buffer) or self.smem_base.get(s.dst_buffer)
            flat = self.e.new_r()
            ld = int(s.ldm) if s.ldm else int((self.shapes.get(s.dst_buffer) or [0, 8])[-1])
            self.e.emit(f"        mad.lo.s32      {flat}, {row}, {ld}, {col};")
            ptr = self.e.new_rd()
            self.e.emit(f"        mad.wide.s32    {ptr}, {flat}, {nb}, {base};")
            if s.atomic:
                self.e.emit(f"        red.{space}.add.f32 [{ptr}], {r};")
            else:
                self.e.emit(f"        st.{space}.f32   [{ptr}], {r};")

    def emit_wgmma_desc(self, s: WgmmaDescriptor) -> None:
        smem_idx = self.flatten(s.smem, (s.smem_index,) if s.smem_index is not None else (Literal(0, "int"),))
        base = self.smem_base[s.smem]
        desc = self.e.new_rd()
        self.e.emit(f"        cvt.u64.u32      {desc}, {base};")
        self.e.emit(f"        // wgmma desc {s.name} swizzle={s.swizzle} lbo={s.lbo_bytes} sbo={s.sbo_bytes}")
        self.e.bind(s.name, desc, "s32")
        self.ptrs[s.name] = desc
        _ = smem_idx

    def emit_wgmma_mma(self, s: WgmmaMma) -> None:
        self.stats["wgmma"] += 1
        m, n, k = (int(x) for x in s.shape[:3])
        acc = []
        for name in s.c_frags:
            acc.extend(self.frags.get(name, {}).get("regs") or [])
        if not acc:
            acc = [self.e.new_f() for _ in range(4)]
        cl = ", ".join(acc[:4])
        b = self.ptrs.get(s.b_desc, s.b_desc)
        a = self.ptrs.get(s.a_desc or "", "") or (self.frags.get(s.a_frag or "", {}) or {}).get("regs", [self.e.new_r()])[0]
        self.e.emit(
            f"        wgmma.mma_async.sync.aligned.m{m}n{n}k{k}.f32.{s.ab_dtype}.{s.ab_dtype} "
            f"{{{cl}}}, {a}, {b}, {int(s.scale_d)}, {int(s.trans_a)}, {int(s.trans_b)};"
        )


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
            "f64": compiler.e.nd,
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
