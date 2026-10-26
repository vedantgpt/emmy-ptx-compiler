#!/usr/bin/env python3
"""Build a *proxy* hidden-like problem set from public operator lists.

Kaggle / Hugging Face CUDA dumps cannot feed compile.py (they are not Kernel IR).
This maps KernelBench L1 ops that sit inside the five official categories onto
fresh Emmy IRs with different shapes and op mixes, then writes expected.npz
with the same seed scheme as tools/generate.py.

    .venv\\Scripts\\python.exe tools/make_proxy_set.py
    .venv\\Scripts\\python.exe tools/run_proxy.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import (
    Axis,
    DataType,
    KernelOp,
    Smem,
    Sync,
    Tile,
    TreeHalve,
    WarpShuffle,
)
from emmy.compiler.ir.stmt import Accum, Assign, Cond, Init, Let, Load, Loop, StridedLoop, Write
from emmy.compiler.tensor import Tensor
from emmy.compiler.dim import Dim

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "problems_proxy"
OPEN = ROOT / "problems"
F32 = DataType("f32", np.dtype("float32"), 4)
BT = 128


def _input(name: str, shape: list[int]) -> dict:
    return {"op": "InputOp", "op_fields": {}, "inputs": [],
            "outputs": [{"name": name, "shape": shape, "dtype": "f32"}]}


def _kernel(entry: str, body, inputs: list[str], out_name: str, out_shape: list[int]) -> dict:
    return {
        "op": "KernelOp",
        "op_fields": {"body": [repr(body)], "name": entry, "zero_delegated": [], "serial": []},
        "inputs": inputs,
        "outputs": [{"name": out_name, "shape": out_shape, "dtype": "f32"}],
    }


def _graph(inputs: list[str], outputs: list[str], nodes: dict) -> dict:
    return {"inputs": inputs, "outputs": outputs, "nodes": nodes}


def _seeded(names: list[str], shapes: dict[str, list[int]]) -> dict[str, np.ndarray]:
    out = {}
    for i, name in enumerate(names):
        rng = np.random.default_rng(i)
        out[name] = rng.standard_normal(shapes[name] or [1]).astype(np.float32)
    return out


def _launch(n: int, block: int = BT) -> tuple[list[int], list[int]]:
    grid = [max(1, math.ceil(n / block)), 1, 1]
    return grid, [block, 1, 1]


def write_problem(pid: str, entry: str, graph: dict, kernels: list[dict],
                  expected: dict[str, np.ndarray], rtol: float, atol: float) -> None:
    d = OUT / pid
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{entry}.ir.json").write_text(json.dumps(graph, indent=2))
    meta = {
        "id": pid, "rtol": rtol, "atol": atol,
        "graph_inputs": graph["inputs"], "graph_outputs": graph["outputs"],
        "kernels": kernels,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    np.savez(d / "expected.npz", **expected)


def pointwise_binary(pid: str, op: str, n: int, ref, rtol=0.0, atol=0.0) -> None:
    entry = f"k_{op}_pointwise"
    out = op if op != "subtract" else "sub"
    body = Tile(axes=(Axis("a0", Dim(n)),), body=(
        Load(names=("in0",), input="x0", index=(Var("a0"),), dtype=F32),
        Load(names=("in1",), input="x1", index=(Var("a0"),), dtype=F32),
        Assign("v0", ElementwiseImpl(op), ("in0", "in1"), F32),
        Write(output=out, index=(Var("a0"),), values=("v0",), value_dtype=F32),
    ))
    graph = _graph(["x0", "x1"], [out], {
        "x0": _input("x0", [n]), "x1": _input("x1", [n]),
        out: _kernel(entry, body, ["x0", "x1"], out, [n]),
    })
    seeded = _seeded(["x0", "x1"], {"x0": [n], "x1": [n]})
    grid, block = _launch(n)
    write_problem(pid, entry, graph, [{
        "node": out, "entry": entry, "arg_order": ["x0", "x1", out],
        "grid": grid, "block": block, "smem_bytes": 0, "zero_outputs": [],
        "args": [
            {"name": "x0", "dtype": "f32", "shape": [n], "seed": 0},
            {"name": "x1", "dtype": "f32", "shape": [n], "seed": 1},
            {"name": out, "dtype": "f32", "shape": [n], "output": True},
        ], "final": True,
    }], {out: np.asarray(ref(seeded["x0"], seeded["x1"]), np.float32)}, rtol, atol)


def pointwise_unary(pid: str, op: str, n: int, ref, rtol=1e-3, atol=1e-4) -> None:
    entry = f"k_{op}_pointwise"
    body = Tile(axes=(Axis("a0", Dim(n)),), body=(
        Load(names=("in0",), input="x", index=(Var("a0"),), dtype=F32),
        Assign("v0", ElementwiseImpl(op), ("in0",), F32),
        Write(output=op, index=(Var("a0"),), values=("v0",), value_dtype=F32),
    ))
    graph = _graph(["x"], [op], {
        "x": _input("x", [n]),
        op: _kernel(entry, body, ["x"], op, [n]),
    })
    seeded = _seeded(["x"], {"x": [n]})
    grid, block = _launch(n)
    write_problem(pid, entry, graph, [{
        "node": op, "entry": entry, "arg_order": ["x", op],
        "grid": grid, "block": block, "smem_bytes": 0, "zero_outputs": [],
        "args": [
            {"name": "x", "dtype": "f32", "shape": [n], "seed": 0},
            {"name": op, "dtype": "f32", "shape": [n], "output": True},
        ], "final": True,
    }], {op: np.asarray(ref(seeded["x"]), np.float32)}, rtol, atol)


def pointwise_relu_mul(pid: str, n: int) -> None:
    entry = "k_mul_pointwise"
    body = Tile(axes=(Axis("a0", Dim(n)),), body=(
        Load(names=("in0",), input="x", index=(Var("a0"),), dtype=F32),
        Assign("v0", ElementwiseImpl("relu"), ("in0",), F32),
        Load(names=("in1",), input="y", index=(Var("a0"),), dtype=F32),
        Assign("v1", ElementwiseImpl("multiply"), ("v0", "in1"), F32),
        Write(output="mul", index=(Var("a0"),), values=("v1",), value_dtype=F32),
    ))
    graph = _graph(["x", "y"], ["mul"], {
        "x": _input("x", [n]), "y": _input("y", [n]),
        "mul": _kernel(entry, body, ["x", "y"], "mul", [n]),
    })
    seeded = _seeded(["x", "y"], {"x": [n], "y": [n]})
    grid, block = _launch(n)
    write_problem(pid, entry, graph, [{
        "node": "mul", "entry": entry, "arg_order": ["x", "y", "mul"],
        "grid": grid, "block": block, "smem_bytes": 0, "zero_outputs": [],
        "args": [
            {"name": "x", "dtype": "f32", "shape": [n], "seed": 0},
            {"name": "y", "dtype": "f32", "shape": [n], "seed": 1},
            {"name": "mul", "dtype": "f32", "shape": [n], "output": True},
        ], "final": True,
    }], {"mul": np.maximum(seeded["x"], 0) * seeded["y"]}, 0.0, 0.0)


def pointwise_scale_shift(pid: str, n: int, scale=2.0, shift=1.0) -> None:
    entry = "k_add_pointwise"
    body = Tile(axes=(Axis("a0", Dim(n)),), body=(
        Load(names=("in0",), input="x", index=(Var("a0"),), dtype=F32),
        Let("s", Literal(scale, "float"), F32),
        Let("b", Literal(shift, "float"), F32),
        Assign("v0", ElementwiseImpl("multiply"), ("in0", "s"), F32),
        Assign("v1", ElementwiseImpl("add"), ("v0", "b"), F32),
        Write(output="add", index=(Var("a0"),), values=("v1",), value_dtype=F32),
    ))
    graph = _graph(["x"], ["add"], {
        "x": _input("x", [n]),
        "add": _kernel(entry, body, ["x"], "add", [n]),
    })
    seeded = _seeded(["x"], {"x": [n]})
    grid, block = _launch(n)
    write_problem(pid, entry, graph, [{
        "node": "add", "entry": entry, "arg_order": ["x", "add"],
        "grid": grid, "block": block, "smem_bytes": 0, "zero_outputs": [],
        "args": [
            {"name": "x", "dtype": "f32", "shape": [n], "seed": 0},
            {"name": "add", "dtype": "f32", "shape": [n], "output": True},
        ], "final": True,
    }], {"add": seeded["x"] * np.float32(scale) + np.float32(shift)}, 1e-6, 1e-6)


def pointwise_2d(pid: str, h: int, w: int) -> None:
    entry = "k_add_pointwise"
    body = Tile(axes=(Axis("a0", Dim(h)), Axis("a1", Dim(w))), body=(
        Load(names=("in0",), input="x0", index=(Var("a0"), Var("a1")), dtype=F32),
        Load(names=("in1",), input="x1", index=(Var("a0"), Var("a1")), dtype=F32),
        Assign("v0", ElementwiseImpl("add"), ("in0", "in1"), F32),
        Write(output="add", index=(Var("a0"), Var("a1")), values=("v0",), value_dtype=F32),
    ))
    graph = _graph(["x0", "x1"], ["add"], {
        "x0": _input("x0", [h, w]), "x1": _input("x1", [h, w]),
        "add": _kernel(entry, body, ["x0", "x1"], "add", [h, w]),
    })
    seeded = _seeded(["x0", "x1"], {"x0": [h, w], "x1": [h, w]})
    grid, block = _launch(h * w)
    write_problem(pid, entry, graph, [{
        "node": "add", "entry": entry, "arg_order": ["x0", "x1", "add"],
        "grid": grid, "block": block, "smem_bytes": 0, "zero_outputs": [],
        "args": [
            {"name": "x0", "dtype": "f32", "shape": [h, w], "seed": 0},
            {"name": "x1", "dtype": "f32", "shape": [h, w], "seed": 1},
            {"name": "add", "dtype": "f32", "shape": [h, w], "output": True},
        ], "final": True,
    }], {"add": seeded["x0"] + seeded["x1"]}, 0.0, 0.0)


def _reduce_tile(n: int, acc_op: str, out_name: str, extra_before_write=()) -> Tile:
    combine = Assign("acc0", ElementwiseImpl(acc_op), ("acc0", "acc0__o"), None)
    return Tile(axes=(Axis("a0_co", Dim(BT)),), body=(
        StridedLoop(
            axis=Axis("a0", Dim(n)), start=Var("a0_co"), step=Literal(128, "int"),
            body=(
                Load(names=("in0",), input="x", index=(Var("a0"),), dtype=F32),
                Accum("acc0", "in0", ElementwiseImpl(acc_op), F32, axes=("a0",)),
            ),
        ),
        WarpShuffle(state=("acc0",), state_b=("acc0__o",), combine_states=(combine,), length=32, dtype=F32),
        Smem("acc0_smem", (4,), "float", 0),
        Cond(BinaryExpr("==", Var("lane"), Literal(0, "int")), body=(
            Write(output="acc0_smem", index=(Var("warp"),), values=("acc0",), value_dtype=F32),
        )),
        Sync(),
        TreeHalve(bufs=("acc0_smem",), state=("acc0",), state_b=("acc0__o",),
                  combine_states=(Assign("acc0", ElementwiseImpl(acc_op), ("acc0", "acc0__o"), None),),
                  length=4, tid_var="warp", dtype=F32),
        *extra_before_write,
        Cond(BinaryExpr("==", Var("a0_co"), Literal(0, "int")), body=(
            Write(output=out_name, index=(), values=("acc0",), value_dtype=F32),
        )),
    ), block_threads=BT)


def reduce_sum(pid: str, n: int) -> None:
    entry = "k_sum_1_reduce"
    body = _reduce_tile(n, "add", "sum_1")
    graph = _graph(["x"], ["sum_1"], {
        "x": _input("x", [n]),
        "sum_1": _kernel(entry, body, ["x"], "sum_1", []),
    })
    seeded = _seeded(["x"], {"x": [n]})
    write_problem(pid, entry, graph, [{
        "node": "sum_1", "entry": entry, "arg_order": ["x", "sum_1"],
        "grid": [1, 1, 1], "block": [BT, 1, 1], "smem_bytes": 0, "zero_outputs": [],
        "args": [
            {"name": "x", "dtype": "f32", "shape": [n], "seed": 0},
            {"name": "sum_1", "dtype": "f32", "shape": [1], "output": True},
        ], "final": True,
    }], {"sum_1": np.asarray(seeded["x"].astype(np.float64).sum(), np.float32)}, 1e-3, 1e-3)


def reduce_max(pid: str, n: int) -> None:
    entry = "k_max_1_reduce"
    body = _reduce_tile(n, "maximum", "max_1")
    graph = _graph(["x"], ["max_1"], {
        "x": _input("x", [n]),
        "max_1": _kernel(entry, body, ["x"], "max_1", []),
    })
    seeded = _seeded(["x"], {"x": [n]})
    write_problem(pid, entry, graph, [{
        "node": "max_1", "entry": entry, "arg_order": ["x", "max_1"],
        "grid": [1, 1, 1], "block": [BT, 1, 1], "smem_bytes": 0, "zero_outputs": [],
        "args": [
            {"name": "x", "dtype": "f32", "shape": [n], "seed": 0},
            {"name": "max_1", "dtype": "f32", "shape": [1], "output": True},
        ], "final": True,
    }], {"max_1": np.asarray(seeded["x"].max(), np.float32)}, 0.0, 0.0)


def reduce_mean(pid: str, n: int) -> None:
    entry = "k_mean_reduce"
    extra = (
        Let("invn", Literal(1.0 / n, "float"), F32),
        Assign("acc0", ElementwiseImpl("multiply"), ("acc0", "invn"), F32),
    )
    body = _reduce_tile(n, "add", "mean", extra)
    graph = _graph(["x"], ["mean"], {
        "x": _input("x", [n]),
        "mean": _kernel(entry, body, ["x"], "mean", []),
    })
    seeded = _seeded(["x"], {"x": [n]})
    write_problem(pid, entry, graph, [{
        "node": "mean", "entry": entry, "arg_order": ["x", "mean"],
        "grid": [1, 1, 1], "block": [BT, 1, 1], "smem_bytes": 0, "zero_outputs": [],
        "args": [
            {"name": "x", "dtype": "f32", "shape": [n], "seed": 0},
            {"name": "mean", "dtype": "f32", "shape": [1], "output": True},
        ], "final": True,
    }], {"mean": np.asarray(seeded["x"].astype(np.float64).mean(), np.float32)}, 1e-3, 1e-3)


def row_sum(pid: str, rows: int, cols: int) -> None:
    entry = "k_sum_1_reduce"
    combine = Assign("acc0", ElementwiseImpl("add"), ("acc0", "acc0__o"), None)
    body = Tile(axes=(Axis("a0", Dim(rows)), Axis("a1_co", Dim(BT))), body=(
        StridedLoop(
            axis=Axis("a1", Dim(cols)), start=Var("a1_co"), step=Literal(128, "int"),
            body=(
                Load(names=("in0",), input="x", index=(Var("a0"), Var("a1")), dtype=F32),
                Accum("acc0", "in0", ElementwiseImpl("add"), F32, axes=("a1",)),
            ),
        ),
        WarpShuffle(state=("acc0",), state_b=("acc0__o",), combine_states=(combine,), length=32, dtype=F32),
        Smem("acc0_smem", (4,), "float", 0),
        Cond(BinaryExpr("==", Var("lane"), Literal(0, "int")), body=(
            Write(output="acc0_smem", index=(Var("warp"),), values=("acc0",), value_dtype=F32),
        )),
        Sync(),
        TreeHalve(bufs=("acc0_smem",), state=("acc0",), state_b=("acc0__o",),
                  combine_states=(Assign("acc0", ElementwiseImpl("add"), ("acc0", "acc0__o"), None),),
                  length=4, tid_var="warp", dtype=F32),
        Cond(BinaryExpr("==", Var("a1_co"), Literal(0, "int")), body=(
            Write(output="sum_1", index=(Var("a0"),), values=("acc0",), value_dtype=F32),
        )),
    ), block_threads=BT)
    graph = _graph(["x"], ["sum_1"], {
        "x": _input("x", [rows, cols]),
        "sum_1": _kernel(entry, body, ["x"], "sum_1", [rows]),
    })
    seeded = _seeded(["x"], {"x": [rows, cols]})
    write_problem(pid, entry, graph, [{
        "node": "sum_1", "entry": entry, "arg_order": ["x", "sum_1"],
        "grid": [rows, 1, 1], "block": [BT, 1, 1], "smem_bytes": 0, "zero_outputs": [],
        "args": [
            {"name": "x", "dtype": "f32", "shape": [rows, cols], "seed": 0},
            {"name": "sum_1", "dtype": "f32", "shape": [rows], "output": True},
        ], "final": True,
    }], {"sum_1": seeded["x"].astype(np.float64).sum(-1).astype(np.float32)}, 1e-3, 1e-3)


def naive_matmul(pid: str, m: int, k: int, n: int) -> None:
    entry = "k_matmul"
    body = Tile(axes=(Axis("i", Dim(m)), Axis("j", Dim(n))), body=(
        Init("acc0", 0.0, dtype=F32),
        Loop(axis=Axis("kk", Dim(k)), body=(
            Load(names=("a",), input="x0", index=(Var("i"), Var("kk")), dtype=F32),
            Load(names=("b",), input="x1", index=(Var("kk"), Var("j")), dtype=F32),
            Assign("p", ElementwiseImpl("multiply"), ("a", "b"), F32),
            Accum("acc0", "p", ElementwiseImpl("add"), F32, axes=("kk",)),
        )),
        Write(output="matmul", index=(Var("i"), Var("j")), values=("acc0",), value_dtype=F32),
    ))
    graph = _graph(["x0", "x1"], ["matmul"], {
        "x0": _input("x0", [m, k]), "x1": _input("x1", [k, n]),
        "matmul": _kernel(entry, body, ["x0", "x1"], "matmul", [m, n]),
    })
    seeded = _seeded(["x0", "x1"], {"x0": [m, k], "x1": [k, n]})
    grid, block = _launch(m * n)
    ref = (seeded["x0"].astype(np.float64) @ seeded["x1"].astype(np.float64)).astype(np.float32)
    write_problem(pid, entry, graph, [{
        "node": "matmul", "entry": entry, "arg_order": ["x0", "x1", "matmul"],
        "grid": grid, "block": block, "smem_bytes": 0, "zero_outputs": [],
        "args": [
            {"name": "x0", "dtype": "f32", "shape": [m, k], "seed": 0},
            {"name": "x1", "dtype": "f32", "shape": [k, n], "seed": 1},
            {"name": "matmul", "dtype": "f32", "shape": [m, n], "output": True},
        ], "final": True,
    }], {"matmul": ref}, 1e-3, 1e-3)


def fused_square_scale(pid: str, n: int) -> None:
    entry = "k_square_scale"
    body = Tile(axes=(Axis("a0", Dim(n)),), body=(
        Load(names=("in0",), input="x", index=(Var("a0"),), dtype=F32),
        Assign("v0", ElementwiseImpl("square"), ("in0",), F32),
        Let("s", Literal(0.5, "float"), F32),
        Assign("v1", ElementwiseImpl("multiply"), ("v0", "s"), F32),
        Write(output="out", index=(Var("a0"),), values=("v1",), value_dtype=F32),
    ))
    graph = _graph(["x"], ["out"], {
        "x": _input("x", [n]),
        "out": _kernel(entry, body, ["x"], "out", [n]),
    })
    seeded = _seeded(["x"], {"x": [n]})
    grid, block = _launch(n)
    write_problem(pid, entry, graph, [{
        "node": "out", "entry": entry, "arg_order": ["x", "out"],
        "grid": grid, "block": block, "smem_bytes": 0, "zero_outputs": [],
        "args": [
            {"name": "x", "dtype": "f32", "shape": [n], "seed": 0},
            {"name": "out", "dtype": "f32", "shape": [n], "output": True},
        ], "final": True,
    }], {"out": (seeded["x"] * seeded["x"]) * np.float32(0.5)}, 1e-6, 1e-6)


def clone_softmax(pid: str, rows: int, cols: int) -> None:
    src = json.loads((OPEN / "f02_row_softmax" / "k_softmax_05263d.ir.json").read_text())
    src["nodes"]["x"]["outputs"][0]["shape"] = [rows, cols]
    src["nodes"]["softmax"]["outputs"][0]["shape"] = [rows, cols]
    body = src["nodes"]["softmax"]["op_fields"]["body"][0]
    body = body.replace("Dim(64)", f"Dim({rows})", 1)
    body = body.replace("Dim(256)", f"Dim({cols})")
    src["nodes"]["softmax"]["op_fields"]["body"][0] = body
    entry = src["nodes"]["softmax"]["op_fields"]["name"]
    seeded = _seeded(["x"], {"x": [rows, cols]})
    x = seeded["x"].astype(np.float64)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    ref = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
    write_problem(pid, entry, src, [{
        "node": "softmax", "entry": entry, "arg_order": ["x", "softmax"],
        "grid": [rows, 1, 1], "block": [BT, 1, 1], "smem_bytes": 0, "zero_outputs": [],
        "args": [
            {"name": "x", "dtype": "f32", "shape": [rows, cols], "seed": 0},
            {"name": "softmax", "dtype": "f32", "shape": [rows, cols], "output": True},
        ], "final": True,
    }], {"softmax": ref}, 1e-3, 1e-4)


def naive_attention(pid: str, seq: int, dim: int, scale: float) -> None:
    """Recompute softmax(Q @ KT) @ V per output cell — small shapes only."""
    entry = "k_attn_naive"
    body = Tile(axes=(Axis("i", Dim(seq)), Axis("j", Dim(dim))), body=(
        Init("m", -1e30, dtype=F32),
        Loop(axis=Axis("k0", Dim(seq)), body=(
            Init("sc0", 0.0, dtype=F32),
            Loop(axis=Axis("d0", Dim(dim)), body=(
                Load(names=("q0",), input="q", index=(Var("i"), Var("d0")), dtype=F32),
                Load(names=("kt0",), input="kt", index=(Var("d0"), Var("k0")), dtype=F32),
                Assign("p0", ElementwiseImpl("multiply"), ("q0", "kt0"), F32),
                Accum("sc0", "p0", ElementwiseImpl("add"), F32, axes=("d0",)),
            )),
            Let("inv", Literal(scale, "float"), F32),
            Assign("sc0", ElementwiseImpl("multiply"), ("sc0", "inv"), F32),
            Accum("m", "sc0", ElementwiseImpl("maximum"), F32, axes=("k0",)),
        )),
        Init("den", 0.0, dtype=F32),
        Loop(axis=Axis("k1", Dim(seq)), body=(
            Init("sc1", 0.0, dtype=F32),
            Loop(axis=Axis("d1", Dim(dim)), body=(
                Load(names=("q1",), input="q", index=(Var("i"), Var("d1")), dtype=F32),
                Load(names=("kt1",), input="kt", index=(Var("d1"), Var("k1")), dtype=F32),
                Assign("p1", ElementwiseImpl("multiply"), ("q1", "kt1"), F32),
                Accum("sc1", "p1", ElementwiseImpl("add"), F32, axes=("d1",)),
            )),
            Let("inv1", Literal(scale, "float"), F32),
            Assign("sc1", ElementwiseImpl("multiply"), ("sc1", "inv1"), F32),
            Assign("dlt", ElementwiseImpl("subtract"), ("sc1", "m"), F32),
            Assign("ex", ElementwiseImpl("exp"), ("dlt",), F32),
            Accum("den", "ex", ElementwiseImpl("add"), F32, axes=("k1",)),
        )),
        Assign("invd", ElementwiseImpl("reciprocal"), ("den",), F32),
        Init("acc", 0.0, dtype=F32),
        Loop(axis=Axis("k2", Dim(seq)), body=(
            Init("sc2", 0.0, dtype=F32),
            Loop(axis=Axis("d2", Dim(dim)), body=(
                Load(names=("q2",), input="q", index=(Var("i"), Var("d2")), dtype=F32),
                Load(names=("kt2",), input="kt", index=(Var("d2"), Var("k2")), dtype=F32),
                Assign("p2", ElementwiseImpl("multiply"), ("q2", "kt2"), F32),
                Accum("sc2", "p2", ElementwiseImpl("add"), F32, axes=("d2",)),
            )),
            Let("inv2", Literal(scale, "float"), F32),
            Assign("sc2", ElementwiseImpl("multiply"), ("sc2", "inv2"), F32),
            Assign("dlt2", ElementwiseImpl("subtract"), ("sc2", "m"), F32),
            Assign("ex2", ElementwiseImpl("exp"), ("dlt2",), F32),
            Assign("w", ElementwiseImpl("multiply"), ("ex2", "invd"), F32),
            Load(names=("vv",), input="v", index=(Var("k2"), Var("j")), dtype=F32),
            Assign("wv", ElementwiseImpl("multiply"), ("w", "vv"), F32),
            Accum("acc", "wv", ElementwiseImpl("add"), F32, axes=("k2",)),
        )),
        Write(output="attn", index=(Var("i"), Var("j")), values=("acc",), value_dtype=F32),
    ))
    graph = _graph(["q", "kt", "v"], ["attn"], {
        "q": _input("q", [seq, dim]),
        "kt": _input("kt", [dim, seq]),
        "v": _input("v", [seq, dim]),
        "attn": _kernel(entry, body, ["q", "kt", "v"], "attn", [seq, dim]),
    })
    seeded = _seeded(["q", "kt", "v"], {"q": [seq, dim], "kt": [dim, seq], "v": [seq, dim]})
    q, kt, v = (seeded[k].astype(np.float64) for k in ("q", "kt", "v"))
    scores = (q @ kt) * scale
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    ref = ((e / e.sum(axis=-1, keepdims=True)) @ v).astype(np.float32)
    grid, block = _launch(seq * dim)
    write_problem(pid, entry, graph, [{
        "node": "attn", "entry": entry, "arg_order": ["q", "kt", "v", "attn"],
        "grid": grid, "block": block, "smem_bytes": 0, "zero_outputs": [],
        "args": [
            {"name": "q", "dtype": "f32", "shape": [seq, dim], "seed": 0},
            {"name": "kt", "dtype": "f32", "shape": [dim, seq], "seed": 1},
            {"name": "v", "dtype": "f32", "shape": [seq, dim], "seed": 2},
            {"name": "attn", "dtype": "f32", "shape": [seq, dim], "output": True},
        ], "final": True,
    }], {"attn": ref}, 2e-3, 2e-3)


def build() -> list[str]:
    if OUT.exists():
        for old in OUT.iterdir():
            if old.is_dir():
                for f in old.iterdir():
                    f.unlink()
                old.rmdir()
    OUT.mkdir(parents=True, exist_ok=True)

    # category 1 — KernelBench-style pointwise, hidden-like sizes (incl. not multiple of block)
    pointwise_binary("p10_add_96", "add", 96, lambda a, b: a + b)
    pointwise_binary("p11_add_200", "add", 200, lambda a, b: a + b)
    pointwise_binary("p12_add_333", "add", 333, lambda a, b: a + b)
    pointwise_binary("p13_add_1000", "add", 1000, lambda a, b: a + b)
    pointwise_binary("p14_add_8192", "add", 8192, lambda a, b: a + b)
    pointwise_binary("p15_sub_512", "subtract", 512, lambda a, b: a - b)
    pointwise_binary("p16_mul_768", "multiply", 768, lambda a, b: a * b)
    pointwise_binary("p17_max_4096", "maximum", 4096, lambda a, b: np.maximum(a, b))
    pointwise_binary("p18_min_256", "minimum", 256, lambda a, b: np.minimum(a, b))
    pointwise_unary("p20_relu_1000", "relu", 1000, lambda x: np.maximum(x, 0), 0, 0)
    pointwise_unary("p21_exp_256", "exp", 256, lambda x: np.exp(np.clip(x, -20, 20)), 2e-2, 2e-2)
    pointwise_unary("p22_tanh_512", "tanh", 512, np.tanh, 2e-2, 2e-2)
    pointwise_unary("p23_sigmoid_384", "sigmoid", 384, lambda x: 1 / (1 + np.exp(-np.clip(x, -20, 20))), 2e-2, 2e-2)
    pointwise_relu_mul("p24_relu_mul_300", 300)
    pointwise_scale_shift("p25_scale_1536", 1536)
    pointwise_2d("p26_add_32x64", 32, 64)
    pointwise_2d("p27_add_7x13", 7, 13)

    # category 2 — reductions, KernelBench 47/48 style
    reduce_sum("r10_sum_256", 256)
    reduce_sum("r11_sum_768", 768)
    reduce_sum("r12_sum_1000", 1000)
    reduce_sum("r13_sum_4096", 4096)
    reduce_sum("r14_sum_16384", 16384)
    reduce_max("r15_max_1024", 1024)
    reduce_mean("r16_mean_777", 777)
    row_sum("r17_row_32x256", 32, 256)
    row_sum("r18_row_16x1024", 16, 1024)
    row_sum("r19_row_128x128", 128, 128)

    # category 3 — KernelBench 1/2/8 irregular GEMM (naive schedule)
    naive_matmul("m10_mm_32", 32, 32, 32)
    naive_matmul("m11_mm_64", 64, 64, 64)
    naive_matmul("m12_mm_rect", 16, 48, 24)
    naive_matmul("m13_mm_irregular", 17, 33, 19)

    # category 4 — fused / KernelBench 23 softmax + square
    fused_square_scale("f10_square_half_512", 512)
    clone_softmax("f11_softmax_32x128", 32, 128)
    clone_softmax("f12_softmax_16x512", 16, 512)
    clone_softmax("f13_softmax_8x256", 8, 256)

    # category 5 — attention-like, smaller than open a01/a02
    naive_attention("a10_attn_8", 8, 8, 1.0 / math.sqrt(8))
    naive_attention("a11_attn_16", 16, 16, 1.0 / math.sqrt(16))

    ids = sorted(p.name for p in OUT.iterdir() if p.is_dir())
    (OUT / "INDEX.json").write_text(json.dumps({
        "note": "Proxy set. Not the official hidden evaluation IRs.",
        "source": ["official 5 categories", "KernelBench L1 names mapped to torch-like ops"],
        "problems": ids,
        "count": len(ids),
    }, indent=2))
    print(f"wrote {len(ids)} proxy problems under {OUT}")
    return ids


if __name__ == "__main__":
    build()
