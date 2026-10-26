"""Leftover threads must not skip bar.sync.

Emmy's CUDA sketch is `if (gid < N) body`. That hangs on Volta+ when
body contains Sync / TreeHalve / cp.async.wait. We keep every lane in
the collective and predicate only global ld/st.
"""

import unittest

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import Axis, DataType, KernelOp, Sync, Tile
from emmy.compiler.ir.stmt import Load, Write

import compile as C

F32 = DataType("f32", np.dtype("float32"), 4)


def _ptx(tile, block=128):
    kop = KernelOp(name="k_guard", body=(tile,))
    return C.Compiler(
        kop,
        shapes={"x": (100,), "y": (100,)},
        dtypes={"x": "f32", "y": "f32"},
        args=["x", "y"],
        meta={"block": [block, 1, 1], "grid": [1, 1, 1]},
    ).run()


class CollectiveGuard(unittest.TestCase):
    def test_leftover_plus_sync_does_not_bra_skip(self):
        tile = Tile(
            axes=(Axis("a0", Dim(100)),),
            block_threads=128,
            body=(
                Load(names=("in0",), input="x", index=(Var("a0"),), dtype=F32),
                Sync(barrier_id=0),
                Write(output="y", index=(Var("a0"),), values=("in0",), value_dtype=F32),
            ),
        )
        ptx = _ptx(tile)
        self.assertIn("bar.sync", ptx)
        self.assertIn("setp.lt.s32", ptx)
        self.assertNotIn("bra        SKIP", ptx)
        self.assertRegex(ptx, r"@%\w+\s+ld\.global")
        self.assertRegex(ptx, r"@%\w+\s+st\.global")
        for line in ptx.splitlines():
            if "bar.sync" in line:
                self.assertFalse(line.strip().startswith("@"), line)

    def test_pointwise_leftover_still_bra_skips(self):
        tile = Tile(
            axes=(Axis("a0", Dim(100)),),
            block_threads=128,
            body=(
                Load(names=("in0",), input="x", index=(Var("a0"),), dtype=F32),
                Write(output="y", index=(Var("a0"),), values=("in0",), value_dtype=F32),
            ),
        )
        ptx = _ptx(tile)
        self.assertIn("bra        SKIP", ptx)
        self.assertNotIn("bar.sync", ptx)


if __name__ == "__main__":
    unittest.main()
