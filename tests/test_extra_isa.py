import unittest

import numpy as np
from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.kernel.ir import (
    DataType,
    KernelOp,
    LdmatrixLoad,
    MbarrierInit,
    MmaSyncPtx,
    RegFragment,
    Smem,
    TmaDescriptor,
    TmaLoad,
)

import compile as C

F16 = DataType("f16", np.dtype("float16"), 2)
F32 = DataType("f32", np.dtype("float32"), 4)


def extra_kernel() -> str:
    sm = Smem("smem_a", (128,), "f32", 16)
    mb = Smem("mbar0", (1,), "u64", 8)
    fa = RegFragment("fa", "a", (16, 8, 16), F16, nregs=4)
    fb = RegFragment("fb", "b", (16, 8, 16), F16, nregs=2)
    fc = RegFragment("fc", "c", (16, 8, 16), F32, nregs=4)
    ld = LdmatrixLoad("fa", "smem_a", (Literal(0, "int"),), "a")
    mma = MmaSyncPtx("fc", "fa", "fb", (16, 8, 16), "f16", "f32")
    desc = TmaDescriptor("tensormap", "x", (128, 128), (32, 32))
    init = MbarrierInit("mbar0", 1)
    tma = TmaLoad(
        "smem_a",
        (Literal(0, "int"),),
        "tensormap",
        (Literal(0, "int"), Literal(0, "int")),
        "mbar0",
    )
    kop = KernelOp(name="k_extra_isa", body=(sm, mb, fa, fb, fc, ld, mma, desc, init, tma))
    compiler = C.Compiler(
        kop,
        shapes={"smem_a": (128,), "mbar0": (1,), "x": (128, 128)},
        dtypes={"smem_a": "f32", "mbar0": "u64", "x": "f32"},
        args=["x"],
        meta={"block": [32, 2, 1], "grid": [4, 2, 1]},
    )
    return compiler.run()


class ExtraIsa(unittest.TestCase):
    def test_tensor_core_and_tma_lower(self):
        ptx = extra_kernel()
        self.assertIn(".target sm_90", ptx)
        self.assertIn("ldmatrix.sync.aligned", ptx)
        self.assertIn("mma.sync.aligned.m16n8k16", ptx)
        self.assertIn("cp.async.bulk.tensor", ptx)
        self.assertIn("mbarrier.init.shared.b64", ptx)
        self.assertIn(".reqntid 32, 2, 1", ptx)
        self.assertIn("%tid.y", ptx)
        self.assertIn("%ctaid.y", ptx)
        self.assertIn("param_tensormap", ptx)
        self.assertNotIn("unsupported", ptx)


if __name__ == "__main__":
    unittest.main()
