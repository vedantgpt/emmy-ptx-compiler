#!/usr/bin/env python3
"""IR problems for every remaining unmarked op mix the compiler claims to lower.

These are not the official hidden set. They exist to prove compile.py keys off
IR structure (Tile/Load/Assign/...) and never a problem id or shape table.

    .venv\\Scripts\\python.exe tools/make_world_set.py
    .venv\\Scripts\\python.exe tools/run_all_irs.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Axis, DataType, Tile
from emmy.compiler.ir.stmt import Assign, Cond, Init, Let, Load, Write
from emmy.compiler.ir.stmt.leaves import Select, SelectBranch
from emmy.compiler.dim import Dim

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "problems_world"
F32 = DataType("f32", np.dtype("float32"), 4)
BT = 128


def _input(name, shape):
    return {"op": "InputOp", "op_fields": {}, "inputs": [],
            "outputs": [{"name": name, "shape": shape, "dtype": "f32"}]}


def _const(name, value):
    return {
        "op": "ConstantOp",
        "op_fields": {
            "name": name, "value": value, "context_value": None,
            "load_ops": [], "source_path": None, "source_parts": [],
            "source_shape": None, "source_dtype": None, "source_graph": None,
        },
        "inputs": [],
        "outputs": [{"name": name, "shape": [1], "dtype": "f32"}],
    }


def _kernel(entry, body, inputs, out, shape):
    return {
        "op": "KernelOp",
        "op_fields": {"body": [repr(body)], "name": entry, "zero_delegated": [], "serial": []},
        "inputs": inputs,
        "outputs": [{"name": out, "shape": shape, "dtype": "f32"}],
    }


def _seed(names, shapes):
    return {n: np.random.default_rng(i).standard_normal(shapes[n] or [1]).astype(np.float32)
            for i, n in enumerate(names)}


def write(pid, entry, graph, kernels, expected, rtol, atol):
    d = OUT / pid
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{entry}.ir.json").write_text(json.dumps(graph, indent=2))
    (d / "meta.json").write_text(json.dumps({
        "id": pid, "rtol": rtol, "atol": atol,
        "graph_inputs": graph["inputs"], "graph_outputs": graph["outputs"],
        "kernels": kernels,
    }, indent=1))
    np.savez(d / "expected.npz", **expected)


def unary(pid, op, n, ref, rtol=1e-3, atol=1e-4, extra_args=()):
    entry = f"k_{op}"
    args = ("in0",) + extra_args
    lets = ()
    if extra_args:
        lets = (Let(extra_args[0], Literal(0.01, "float"), F32),)
    body = Tile(axes=(Axis("a0", Dim(n)),), body=lets + (
        Load(names=("in0",), input="x", index=(Var("a0"),), dtype=F32),
        Assign("v0", ElementwiseImpl(op), args, F32),
        Write(output=op, index=(Var("a0"),), values=("v0",), value_dtype=F32),
    ))
    graph = {"inputs": ["x"], "outputs": [op], "nodes": {
        "x": _input("x", [n]),
        op: _kernel(entry, body, ["x"], op, [n]),
    }}
    x = _seed(["x"], {"x": [n]})["x"]
    g, b = [math.ceil(n / BT), 1, 1], [BT, 1, 1]
    write(pid, entry, graph, [{
        "node": op, "entry": entry, "arg_order": ["x", op],
        "grid": g, "block": b, "smem_bytes": 0, "zero_outputs": [],
        "args": [
            {"name": "x", "dtype": "f32", "shape": [n], "seed": 0},
            {"name": op, "dtype": "f32", "shape": [n], "output": True},
        ], "final": True,
    }], {op: np.asarray(ref(x), np.float32)}, rtol, atol)


def vec4_add(pid, n):
    assert n % 4 == 0
    tiles = n // 4
    entry = "k_add_v4"
    idx = lambda k: BinaryExpr("+", BinaryExpr("*", Var("a0"), Literal(4, "int")), Literal(k, "int"))
    loads = []
    for k in range(4):
        loads += [
            Load(names=(f"lhs{k}",), input="x0", index=(idx(k),), dtype=F32),
            Load(names=(f"rhs{k}",), input="x1", index=(idx(k),), dtype=F32),
            Assign(f"v{k}", ElementwiseImpl("add"), (f"lhs{k}", f"rhs{k}"), F32),
        ]
    body = Tile(axes=(Axis("a0", Dim(tiles)),), body=tuple(loads) + (
        Write(output="add", index=(idx(0),), values=("v0", "v1", "v2", "v3"), value_dtype=F32),
    ))
    graph = {"inputs": ["x0", "x1"], "outputs": ["add"], "nodes": {
        "x0": _input("x0", [n]), "x1": _input("x1", [n]),
        "add": _kernel(entry, body, ["x0", "x1"], "add", [n]),
    }}
    s = _seed(["x0", "x1"], {"x0": [n], "x1": [n]})
    g, b = [math.ceil(tiles / BT), 1, 1], [BT, 1, 1]
    write(pid, entry, graph, [{
        "node": "add", "entry": entry, "arg_order": ["x0", "x1", "add"],
        "grid": g, "block": b, "smem_bytes": 0, "zero_outputs": [],
        "args": [
            {"name": "x0", "dtype": "f32", "shape": [n], "seed": 0},
            {"name": "x1", "dtype": "f32", "shape": [n], "seed": 1},
            {"name": "add", "dtype": "f32", "shape": [n], "output": True},
        ], "final": True,
    }], {"add": s["x0"] + s["x1"]}, 0.0, 0.0)


def const_scale(pid, n, scale=3.0, shift=-0.5):
    entry = "k_const_scale"
    body = Tile(axes=(Axis("a0", Dim(n)),), body=(
        Load(names=("c0",), input="mul_c", index=(Literal(0, "int"),), dtype=F32),
        Load(names=("c1",), input="add_c", index=(Literal(0, "int"),), dtype=F32),
        Load(names=("in0",), input="x", index=(Var("a0"),), dtype=F32),
        Assign("v0", ElementwiseImpl("multiply"), ("in0", "c0"), F32),
        Assign("v1", ElementwiseImpl("add"), ("v0", "c1"), F32),
        Write(output="add", index=(Var("a0"),), values=("v1",), value_dtype=F32),
    ))
    graph = {"inputs": ["x"], "outputs": ["add"], "nodes": {
        "mul_c": _const("mul_c", scale),
        "add_c": _const("add_c", shift),
        "x": _input("x", [n]),
        "add": _kernel(entry, body, ["mul_c", "add_c", "x"], "add", [n]),
    }}
    x = _seed(["x"], {"x": [n]})["x"]
    g, b = [math.ceil(n / BT), 1, 1], [BT, 1, 1]
    write(pid, entry, graph, [{
        "node": "add", "entry": entry, "arg_order": ["mul_c", "add_c", "x", "add"],
        "grid": g, "block": b, "smem_bytes": 0, "zero_outputs": [],
        "args": [
            {"name": "mul_c", "dtype": "f32", "shape": [1], "data": [scale]},
            {"name": "add_c", "dtype": "f32", "shape": [1], "data": [shift]},
            {"name": "x", "dtype": "f32", "shape": [n], "seed": 0},
            {"name": "add", "dtype": "f32", "shape": [n], "output": True},
        ], "final": True,
    }], {"add": x * np.float32(scale) + np.float32(shift)}, 1e-6, 1e-6)


def cond_else(pid, n):
    entry = "k_cond_else"
    body = Tile(axes=(Axis("a0", Dim(n)),), body=(
        Load(names=("in0",), input="x", index=(Var("a0"),), dtype=F32),
        Init("v0", 0.0, dtype=F32),
        Cond(
            BinaryExpr(">", Var("a0"), Literal(n // 2, "int")),
            body=(Assign("v0", ElementwiseImpl("relu"), ("in0",), F32),),
            else_body=(Assign("v0", ElementwiseImpl("negative"), ("in0",), F32),),
        ),
        Write(output="out", index=(Var("a0"),), values=("v0",), value_dtype=F32),
    ))
    graph = {"inputs": ["x"], "outputs": ["out"], "nodes": {
        "x": _input("x", [n]),
        "out": _kernel(entry, body, ["x"], "out", [n]),
    }}
    x = _seed(["x"], {"x": [n]})["x"]
    idx = np.arange(n)
    ref = np.where(idx > n // 2, np.maximum(x, 0), -x).astype(np.float32)
    g, b = [math.ceil(n / BT), 1, 1], [BT, 1, 1]
    write(pid, entry, graph, [{
        "node": "out", "entry": entry, "arg_order": ["x", "out"],
        "grid": g, "block": b, "smem_bytes": 0, "zero_outputs": [],
        "args": [
            {"name": "x", "dtype": "f32", "shape": [n], "seed": 0},
            {"name": "out", "dtype": "f32", "shape": [n], "output": True},
        ], "final": True,
    }], {"out": ref}, 0.0, 0.0)


def select_relu(pid, n):
    entry = "k_select"
    body = Tile(axes=(Axis("a0", Dim(n)),), body=(
        Load(names=("in0",), input="x", index=(Var("a0"),), dtype=F32),
        Let("z", Literal(0.0, "float"), F32),
        Select("v0", (
            SelectBranch("in0", BinaryExpr(">", Var("in0"), Var("z"))),
            SelectBranch("z", Literal(1, "int")),
        )),
        Write(output="out", index=(Var("a0"),), values=("v0",), value_dtype=F32),
    ))
    graph = {"inputs": ["x"], "outputs": ["out"], "nodes": {
        "x": _input("x", [n]),
        "out": _kernel(entry, body, ["x"], "out", [n]),
    }}
    x = _seed(["x"], {"x": [n]})["x"]
    g, b = [math.ceil(n / BT), 1, 1], [BT, 1, 1]
    write(pid, entry, graph, [{
        "node": "out", "entry": entry, "arg_order": ["x", "out"],
        "grid": g, "block": b, "smem_bytes": 0, "zero_outputs": [],
        "args": [
            {"name": "x", "dtype": "f32", "shape": [n], "seed": 0},
            {"name": "out", "dtype": "f32", "shape": [n], "output": True},
        ], "final": True,
    }], {"out": np.maximum(x, 0)}, 0.0, 0.0)


def where_gt(pid, n):
    entry = "k_where"
    body = Tile(axes=(Axis("a0", Dim(n)),), body=(
        Load(names=("a",), input="x0", index=(Var("a0"),), dtype=F32),
        Load(names=("b",), input="x1", index=(Var("a0"),), dtype=F32),
        Assign("p", ElementwiseImpl("greater"), ("a", "b"), F32),
        Assign("v0", ElementwiseImpl("where"), ("p", "a", "b"), F32),
        Write(output="out", index=(Var("a0"),), values=("v0",), value_dtype=F32),
    ))
    graph = {"inputs": ["x0", "x1"], "outputs": ["out"], "nodes": {
        "x0": _input("x0", [n]), "x1": _input("x1", [n]),
        "out": _kernel(entry, body, ["x0", "x1"], "out", [n]),
    }}
    s = _seed(["x0", "x1"], {"x0": [n], "x1": [n]})
    g, b = [math.ceil(n / BT), 1, 1], [BT, 1, 1]
    write(pid, entry, graph, [{
        "node": "out", "entry": entry, "arg_order": ["x0", "x1", "out"],
        "grid": g, "block": b, "smem_bytes": 0, "zero_outputs": [],
        "args": [
            {"name": "x0", "dtype": "f32", "shape": [n], "seed": 0},
            {"name": "x1", "dtype": "f32", "shape": [n], "seed": 1},
            {"name": "out", "dtype": "f32", "shape": [n], "output": True},
        ], "final": True,
    }], {"out": np.where(s["x0"] > s["x1"], s["x0"], s["x1"])}, 0.0, 0.0)


def atomic_sum(pid, n):
    entry = "k_atomic_sum"
    body = Tile(axes=(Axis("a0", Dim(n)),), body=(
        Load(names=("in0",), input="x", index=(Var("a0"),), dtype=F32),
        Write(output="sum_1", index=(), values=("in0",), value_dtype=F32, atomic=True),
    ))
    graph = {"inputs": ["x"], "outputs": ["sum_1"], "nodes": {
        "x": _input("x", [n]),
        "sum_1": _kernel(entry, body, ["x"], "sum_1", []),
    }}
    x = _seed(["x"], {"x": [n]})["x"]
    g, b = [math.ceil(n / BT), 1, 1], [BT, 1, 1]
    write(pid, entry, graph, [{
        "node": "sum_1", "entry": entry, "arg_order": ["x", "sum_1"],
        "grid": g, "block": b, "smem_bytes": 0, "zero_outputs": ["sum_1"],
        "args": [
            {"name": "x", "dtype": "f32", "shape": [n], "seed": 0},
            {"name": "sum_1", "dtype": "f32", "shape": [1], "output": True},
        ], "final": True,
    }], {"sum_1": np.asarray(x.astype(np.float64).sum(), np.float32)}, 1e-3, 1e-3)


def launch_3d(pid, n=64):
    entry = "k_add_3d"
    body = Tile(axes=(Axis("a0", Dim(n)),), body=(
        Load(names=("in0",), input="x0", index=(Var("a0"),), dtype=F32),
        Load(names=("in1",), input="x1", index=(Var("a0"),), dtype=F32),
        Assign("v0", ElementwiseImpl("add"), ("in0", "in1"), F32),
        Write(output="add", index=(Var("a0"),), values=("v0",), value_dtype=F32),
    ), block_threads=n)
    graph = {"inputs": ["x0", "x1"], "outputs": ["add"], "nodes": {
        "x0": _input("x0", [n]), "x1": _input("x1", [n]),
        "add": _kernel(entry, body, ["x0", "x1"], "add", [n]),
    }}
    s = _seed(["x0", "x1"], {"x0": [n], "x1": [n]})
    write(pid, entry, graph, [{
        "node": "add", "entry": entry, "arg_order": ["x0", "x1", "add"],
        "grid": [1, 1, 1], "block": [8, 4, 2], "smem_bytes": 0, "zero_outputs": [],
        "args": [
            {"name": "x0", "dtype": "f32", "shape": [n], "seed": 0},
            {"name": "x1", "dtype": "f32", "shape": [n], "seed": 1},
            {"name": "add", "dtype": "f32", "shape": [n], "output": True},
        ], "final": True,
    }], {"add": s["x0"] + s["x1"]}, 0.0, 0.0)


def leaky(pid, n, slope=0.01):
    entry = "k_leaky"
    body = Tile(axes=(Axis("a0", Dim(n)),), body=(
        Load(names=("in0",), input="x", index=(Var("a0"),), dtype=F32),
        Let("z", Literal(0.0, "float"), F32),
        Let("s", Literal(slope, "float"), F32),
        Assign("p", ElementwiseImpl("greater"), ("in0", "z"), F32),
        Assign("sc", ElementwiseImpl("multiply"), ("in0", "s"), F32),
        Assign("v0", ElementwiseImpl("where"), ("p", "in0", "sc"), F32),
        Write(output="out", index=(Var("a0"),), values=("v0",), value_dtype=F32),
    ))
    graph = {"inputs": ["x"], "outputs": ["out"], "nodes": {
        "x": _input("x", [n]),
        "out": _kernel(entry, body, ["x"], "out", [n]),
    }}
    x = _seed(["x"], {"x": [n]})["x"]
    g, b = [math.ceil(n / BT), 1, 1], [BT, 1, 1]
    write(pid, entry, graph, [{
        "node": "out", "entry": entry, "arg_order": ["x", "out"],
        "grid": g, "block": b, "smem_bytes": 0, "zero_outputs": [],
        "args": [
            {"name": "x", "dtype": "f32", "shape": [n], "seed": 0},
            {"name": "out", "dtype": "f32", "shape": [n], "output": True},
        ], "final": True,
    }], {"out": np.where(x > 0, x, x * np.float32(slope))}, 1e-5, 1e-5)


def clamp_op(pid, n):
    entry = "k_clamp"
    body = Tile(axes=(Axis("a0", Dim(n)),), body=(
        Load(names=("in0",), input="x", index=(Var("a0"),), dtype=F32),
        Let("lo", Literal(-0.5, "float"), F32),
        Let("hi", Literal(0.5, "float"), F32),
        Assign("t", ElementwiseImpl("maximum"), ("in0", "lo"), F32),
        Assign("v0", ElementwiseImpl("minimum"), ("t", "hi"), F32),
        Write(output="out", index=(Var("a0"),), values=("v0",), value_dtype=F32),
    ))
    graph = {"inputs": ["x"], "outputs": ["out"], "nodes": {
        "x": _input("x", [n]),
        "out": _kernel(entry, body, ["x"], "out", [n]),
    }}
    x = _seed(["x"], {"x": [n]})["x"]
    g, b = [math.ceil(n / BT), 1, 1], [BT, 1, 1]
    write(pid, entry, graph, [{
        "node": "out", "entry": entry, "arg_order": ["x", "out"],
        "grid": g, "block": b, "smem_bytes": 0, "zero_outputs": [],
        "args": [
            {"name": "x", "dtype": "f32", "shape": [n], "seed": 0},
            {"name": "out", "dtype": "f32", "shape": [n], "output": True},
        ], "final": True,
    }], {"out": np.clip(x, -0.5, 0.5)}, 0.0, 0.0)


def build():
    if OUT.exists():
        for old in OUT.iterdir():
            if old.is_dir():
                for f in old.iterdir():
                    f.unlink()
                old.rmdir()
            else:
                old.unlink()
    OUT.mkdir(parents=True, exist_ok=True)

    unary("w01_abs_222", "abs", 222, np.abs, 0, 0)
    unary("w02_neg_111", "negative", 111, lambda x: -x, 0, 0)
    unary("w03_sqrt_256", "sqrt", 256, lambda x: np.sqrt(np.abs(x) + 1e-6), 2e-3, 2e-3)
    unary("w04_rsqrt_192", "rsqrt", 192, lambda x: 1 / np.sqrt(np.abs(x) + 1e-3), 2e-2, 2e-2)
    unary("w05_log_160", "log", 160, lambda x: np.log(np.abs(x) + 1e-3), 3e-2, 3e-2)
    unary("w06_silu_200", "silu", 200, lambda x: x / (1 + np.exp(-np.clip(x, -20, 20))), 2e-2, 2e-2)
    unary("w07_copy_96", "copy", 96, lambda x: x, 0, 0)
    leaky("w08_leaky_180", 180)
    clamp_op("w09_clamp_240", 240)
    vec4_add("w10_vec4_1024", 1024)
    const_scale("w11_const_777", 777)
    cond_else("w12_cond_300", 300)
    select_relu("w13_select_256", 256)
    where_gt("w14_where_384", 384)
    atomic_sum("w15_atomic_512", 512)
    launch_3d("w16_launch3d_64", 64)

    # sqrt/rsqrt/log need non-negative inputs — rewrite those refs to match IR (raw x)
    # IR applies sqrt to raw x; only abs/neg/copy/silu/leaky are safe as written.
    # Fix sqrt/rsqrt/log: apply abs+eps in the IR.
    _fix_safe_unaries()

    ids = sorted(p.name for p in OUT.iterdir() if p.is_dir())
    (OUT / "INDEX.json").write_text(json.dumps({"count": len(ids), "problems": ids}, indent=2))
    print(f"wrote {len(ids)} world problems under {OUT}")
    return ids


def _fix_safe_unaries():
    """Rebuild sqrt/rsqrt/log so the IR matches a defined numpy ref."""
    for pid, op, n, prep, ref, rtol in (
        ("w03_sqrt_256", "sqrt", 256, lambda x: np.abs(x) + 1e-6, np.sqrt, 2e-3),
        ("w04_rsqrt_192", "rsqrt", 192, lambda x: np.abs(x) + 1e-3, lambda z: 1 / np.sqrt(z), 2e-2),
        ("w05_log_160", "log", 160, lambda x: np.abs(x) + 1e-3, np.log, 3e-2),
    ):
        entry = f"k_{op}"
        body = Tile(axes=(Axis("a0", Dim(n)),), body=(
            Load(names=("in0",), input="x", index=(Var("a0"),), dtype=F32),
            Assign("a", ElementwiseImpl("abs"), ("in0",), F32),
            Let("eps", Literal(1e-3 if op != "sqrt" else 1e-6, "float"), F32),
            Assign("p", ElementwiseImpl("add"), ("a", "eps"), F32),
            Assign("v0", ElementwiseImpl(op), ("p",), F32),
            Write(output=op, index=(Var("a0"),), values=("v0",), value_dtype=F32),
        ))
        graph = {"inputs": ["x"], "outputs": [op], "nodes": {
            "x": _input("x", [n]),
            op: _kernel(entry, body, ["x"], op, [n]),
        }}
        x = _seed(["x"], {"x": [n]})["x"]
        g, b = [math.ceil(n / BT), 1, 1], [BT, 1, 1]
        write(pid, entry, graph, [{
            "node": op, "entry": entry, "arg_order": ["x", op],
            "grid": g, "block": b, "smem_bytes": 0, "zero_outputs": [],
            "args": [
                {"name": "x", "dtype": "f32", "shape": [n], "seed": 0},
                {"name": op, "dtype": "f32", "shape": [n], "output": True},
            ], "final": True,
        }], {op: np.asarray(ref(prep(x)), np.float32)}, rtol, rtol)


if __name__ == "__main__":
    build()
