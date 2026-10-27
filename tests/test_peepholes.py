"""Regression tests for peepholes added after benchmarking ~60 public
compilers: vector-store hoisting across a value-producing Assign, and
shared-memory address reuse inside TreeHalve (load/store same slot).

Both are measured wins (a01_attention: 8 scalar stores -> 4 st.global.v2;
f01/r03 TreeHalve: one mad.lo.s32 per slot instead of two) kept honest by
asserting the exact PTX shape, not just "it still compiles".
"""

import re
import unittest

import numpy as np
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.dim import Dim
from emmy.compiler.ir.kernel.ir import Axis, DataType, KernelOp, Smem, Tile, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt import Accum, Assign, Load, Loop, Write

import compile as C

F32 = DataType("f32", np.dtype("float32"), 4)


def _compile(kop: KernelOp, **shapes_dtypes) -> str:
    shapes = shapes_dtypes.get("shapes", {})
    dtypes = shapes_dtypes.get("dtypes", {})
    args = shapes_dtypes.get("args", [])
    meta = shapes_dtypes.get("meta", {"block": [32, 1, 1], "grid": [1, 1, 1]})
    return C.Compiler(kop, shapes=shapes, dtypes=dtypes, args=args, meta=meta).run()


class VecStoreAcrossAssign(unittest.TestCase):
    def test_reciprocal_between_writes_still_vectorizes(self):
        body = (
            Load(names=("in0",), input="x", index=(Literal(0, "int"),), dtype=F32),
            Load(names=("in1",), input="x", index=(Literal(1, "int"),), dtype=F32),
            Assign(name="a0", op=ElementwiseImpl("reciprocal"), args=("in0",), dtype=F32),
            Write(output="out", index=(Literal(0, "int"),), values=("a0",), value_dtype=F32),
            Assign(name="a1", op=ElementwiseImpl("reciprocal"), args=("in1",), dtype=F32),
            Write(output="out", index=(Literal(1, "int"),), values=("a1",), value_dtype=F32),
        )
        kop = KernelOp(name="k_vecstore", body=body)
        ptx = _compile(
            kop,
            shapes={"x": (2,), "out": (2,)},
            dtypes={"x": "f32", "out": "f32"},
            args=["x", "out"],
        )
        self.assertIn("st.global.v2.f32", ptx)
        self.assertEqual(ptx.count("st.global.f32"), 0)
        # Both reciprocals must still run (nothing got dropped by the hoist).
        self.assertEqual(ptx.count("rcp.rn.f32"), 2)

    def test_unrelated_write_between_still_blocks_vectorization(self):
        body = (
            Load(names=("in0",), input="x", index=(Literal(0, "int"),), dtype=F32),
            Load(names=("in1",), input="x", index=(Literal(1, "int"),), dtype=F32),
            Write(output="out", index=(Literal(0, "int"),), values=("in0",), value_dtype=F32),
            Write(output="other", index=(Literal(0, "int"),), values=("in0",), value_dtype=F32),
            Write(output="out", index=(Literal(1, "int"),), values=("in1",), value_dtype=F32),
        )
        kop = KernelOp(name="k_novecstore", body=body)
        ptx = _compile(
            kop,
            shapes={"x": (2,), "out": (2,), "other": (1,)},
            dtypes={"x": "f32", "out": "f32", "other": "f32"},
            args=["x", "out", "other"],
        )
        self.assertNotIn("st.global.v2.f32", ptx)


class TreeHalveAddressReuse(unittest.TestCase):
    def test_load_store_same_slot_reuses_address_register(self):
        sm = Smem("acc0_smem", (4,), "f32", 16)
        th = TreeHalve(
            bufs=("acc0_smem",),
            state=("acc0",),
            state_b=("acc0_b",),
            combine_states=(Assign(name="acc0", op=ElementwiseImpl("add"), args=("acc0", "acc0_b"), dtype=F32),),
            length=4,
            tid_var="warp",
            inner=None,
            barrier_id=0,
            barrier_count=0,
        )
        tile = Tile(
            axes=(Axis("a0", Dim(32)),),
            block_threads=32,
            body=(
                sm,
                Accum(name="acc0", value="warp", op=ElementwiseImpl("add"), dtype=F32),
                th,
            ),
        )
        kop = KernelOp(name="k_treehalve", body=(tile,))
        ptx = _compile(
            kop,
            shapes={"y": (32,)},
            dtypes={"y": "f32"},
            args=["y"],
            meta={"block": [32, 1, 1], "grid": [1, 1, 1]},
        )
        # Every TSTEP block loads [slot] and later stores back to [slot]:
        # the store must reuse the load's address register, not recompute it.
        steps = [blk for blk in ptx.split("TSTEP_") if "ld.shared.f32" in blk and "st.shared.f32" in blk]
        self.assertTrue(steps, "expected at least one TreeHalve step in the PTX")
        for blk in steps:
            ld_line = [l for l in blk.splitlines() if "ld.shared.f32" in l][0]
            ld_addr = ld_line.split("[")[1].split("]")[0]
            st_line = [l for l in blk.splitlines() if "st.shared.f32" in l][0]
            st_addr = st_line.split("[")[1].split("]")[0]
            self.assertEqual(ld_addr, st_addr, blk)


class LaneWarpUseLinearizedTid(unittest.TestCase):
    def test_warp_index_depends_on_tid_y_for_multirow_block(self):
        # block = [32, 2, 1]: two rows of a warp each. lane/warp must come
        # from the LOCAL linearized thread index (tid.y*32 + tid.x), not
        # raw tid.x, or every row would alias onto warp 0.
        body = (
            Load(names=("in0",), input="x", index=(Literal(0, "int"),), dtype=F32),
            WarpShuffle(
                state=("in0",),
                state_b=("in0_b",),
                combine_states=(Assign(name="in0", op=ElementwiseImpl("add"), args=("in0", "in0_b"), dtype=F32),),
                length=32,
            ),
            Write(output="out", index=(Literal(0, "int"),), values=("in0",), value_dtype=F32),
        )
        kop = KernelOp(name="k_warprow", body=body)
        ptx = _compile(
            kop,
            shapes={"x": (64,), "out": (64,)},
            dtypes={"x": "f32", "out": "f32"},
            args=["x", "out"],
            meta={"block": [32, 2, 1], "grid": [1, 1, 1]},
        )
        self.assertIn("%tid.y", ptx)
        tidx_reg = re.search(r"mov\.u32\s+(%r\d+), %tid\.x;", ptx).group(1)
        lane_line = re.search(r"and\.b32\s+%r\d+, (%r\d+), 31;", ptx)
        self.assertIsNotNone(lane_line, ptx)
        self.assertNotEqual(
            lane_line.group(1), tidx_reg,
            "lane must be derived from the linearized local tid, not bare %tid.x, "
            "once the block has more than one row",
        )


class UniformBufferSharedMemoryCache(unittest.TestCase):
    """a02_attention_128 used to re-read K from global memory on every
    one of the 4096 (a1, a2) loop trips despite a low register count
    (48). The gap was a structural difference, not a register-allocation
    artifact: K's index never depends on which thread owns the row, so
    the whole matrix can be staged into shared memory once per block.
    _plan_buffer_cache/_emit_buffer_cache generalize that: a read-only
    buffer whose every index is built ONLY from Loop/StridedLoop
    induction variables (never a Tile axis) is copied into a `.shared`
    mirror once, then every Load of it is redirected to `ld.shared`.
    Measured on the real GPU runner: a02 98us -> 38us, a01 217us -> 120us,
    with all 15 open-set kernels still passing.
    """

    def test_buffer_indexed_only_by_loop_var_is_staged_into_shared_memory(self):
        body = (
            Loop(
                axis=Axis("a1", Dim(8)),
                body=(
                    Load(names=("kv",), input="kt", index=(Var("a1"),), dtype=F32),
                    Accum(name="acc", value="kv", dtype=F32),
                ),
            ),
            Write(output="out", index=(Var("row"),), values=("acc",), value_dtype=F32),
        )
        tile = Tile(axes=(Axis("row", Dim(4)),), block_threads=4, body=body)
        kop = KernelOp(name="k_cacheable", body=(tile,))
        ptx = _compile(
            kop,
            shapes={"kt": (8,), "out": (4,)},
            dtypes={"kt": "f32", "out": "f32"},
            args=["kt", "out"],
            meta={"block": [4, 1, 1], "grid": [1, 1, 1]},
        )
        self.assertIn(".shared .align 16 .b8 _cache_kt[32];", ptx)
        self.assertIn("bar.sync        0;", ptx)
        # Inside the reduce loop, kt must now come from shared memory.
        loop_body = ptx.split("LOOP_")[1] if "LOOP_" in ptx else ptx
        self.assertIn("ld.shared.f32", loop_body)

    def test_buffer_indexed_by_tile_axis_is_not_cached(self):
        # Regression guard: a02's Q buffer is read with an index built ONLY
        # from Tile axes (q[row, a2], row = a Tile axis) - constant for a
        # whole block's lifetime, but a DIFFERENT row per block. An early
        # version of this pass treated "stride >= block_threads" Tile axes
        # as uniform and cached the WHOLE Q buffer per block, wasting 31/32
        # of the staged data and roughly halving occupancy (measured: a02
        # regressed from 38us to 43us versus caching only K). Whole-buffer
        # caching is only correct/worthwhile when EVERY dimension is swept
        # by a real Loop inside this block's own execution, never a Tile
        # axis - even one with a block-spanning stride.
        body = (
            Loop(
                axis=Axis("a1", Dim(8)),
                body=(
                    Load(names=("kv",), input="kt", index=(Var("blk"),), dtype=F32),
                    Accum(name="acc", value="kv", dtype=F32),
                ),
            ),
            Write(output="out", index=(Var("col"),), values=("acc",), value_dtype=F32),
        )
        tile = Tile(
            axes=(Axis("blk", Dim(2)), Axis("col", Dim(4))),
            block_threads=4,
            body=body,
        )
        kop = KernelOp(name="k_notcacheable", body=(tile,))
        ptx = _compile(
            kop,
            shapes={"kt": (2,), "out": (8,)},
            dtypes={"kt": "f32", "out": "f32"},
            args=["kt", "out"],
            meta={"block": [4, 1, 1], "grid": [2, 1, 1]},
        )
        self.assertNotIn("_cache_kt", ptx)
        self.assertIn("ld.global", ptx)


class WarpUniformReductionShuffle(unittest.TestCase):
    """After _plan_buffer_cache closed the shared-memory gap, a02 still
    spent 38us on a 32x redundant reduction: a02's Tile maps one 32-wide
    warp to one output row, with the per-row Q.K dot product's reduction
    axis (head_dim) ALSO exactly 32 wide and never indexed by the lane -
    meaning every one of the 32 lanes in that warp computed the identical
    32-term sum serially. _plan_warp_reduce_loops/_emit_warp_reduce_loop
    rewrite that: a reduce Loop of extent 32 inside a single-warp
    (block_threads == 32) Tile, whose body is a pure multiply-add
    accumulation that never indexes by the Tile's lane-mapped axis, gets
    rewritten from a 32-iteration (often fully-unrolled) serial loop into
    one lane-indexed term plus a 5-step butterfly reduce. Measured on the
    real GPU runner: a02 38us -> ~25us (48 -> 24 registers locally, 39 on
    the runner's own ptxas), with the other 14 open-set kernels
    byte-for-byte unchanged (none of them has a block_threads==32 Tile
    with a reduce Loop of extent exactly 32 that is lane-independent).
    """

    def test_lane_independent_32_reduction_becomes_shuffle(self):
        body = (
            Loop(
                axis=Axis("k", Dim(32)),
                body=(
                    Load(names=("a",), input="kt", index=(Var("k"),), dtype=F32),
                    Load(names=("b",), input="q", index=(Var("k"),), dtype=F32),
                    Assign(name="p", op=ElementwiseImpl("multiply"), args=("a", "b"), dtype=F32),
                    Accum(name="acc", value="p", op=ElementwiseImpl("add"), dtype=F32),
                ),
            ),
            Write(output="out", index=(Var("ch"),), values=("acc",), value_dtype=F32),
        )
        tile = Tile(axes=(Axis("row", Dim(1)), Axis("ch", Dim(32))), block_threads=32, body=body)
        kop = KernelOp(name="k_warpreduce", body=(tile,))
        ptx = _compile(
            kop,
            shapes={"kt": (32,), "q": (32,), "out": (32,)},
            dtypes={"kt": "f32", "q": "f32", "out": "f32"},
            args=["kt", "q", "out"],
            meta={"block": [32, 1, 1], "grid": [1, 1, 1]},
        )
        self.assertIn("activemask.b32", ptx)
        self.assertEqual(ptx.count("shfl.sync.bfly.b32"), 5)
        # One lane-indexed term, not 32 unrolled serial ones.
        self.assertEqual(ptx.count("fma.rn.f32"), 1)

    def test_lane_dependent_reduction_is_not_rewritten(self):
        # Regression guard: if the reduction's own index depends on the
        # lane-mapped axis (a genuinely parallel-across-lanes workload,
        # not a redundant one - e.g. a per-channel running sum), lanes
        # are NOT computing the same value and must stay independent.
        body = (
            Loop(
                axis=Axis("k", Dim(32)),
                body=(
                    Load(names=("a",), input="kt", index=(Var("ch"),), dtype=F32),
                    Accum(name="acc", value="a", op=ElementwiseImpl("add"), dtype=F32),
                ),
            ),
            Write(output="out", index=(Var("ch"),), values=("acc",), value_dtype=F32),
        )
        tile = Tile(axes=(Axis("row", Dim(1)), Axis("ch", Dim(32))), block_threads=32, body=body)
        kop = KernelOp(name="k_nowarpreduce", body=(tile,))
        ptx = _compile(
            kop,
            shapes={"kt": (32,), "out": (32,)},
            dtypes={"kt": "f32", "out": "f32"},
            args=["kt", "out"],
            meta={"block": [32, 1, 1], "grid": [1, 1, 1]},
        )
        self.assertNotIn("shfl.sync.bfly", ptx)


class UnrolledCopyAddressReuse(unittest.TestCase):
    """The warp-reduce fix above closed most, but not all, of a02's
    leftover serial work (measured: 38us -> ~25us on the real GPU
    runner). The remaining difference in our own lowering: the outer
    K-row loop can be unrolled 8x (an ordinary partial unroll,
    `_partial_factor`), and each of the 8 copies' V/out addresses can
    reuse copy 0's base register with a cheap immediate byte offset
    (`[%rd+128]`, `[%rd+256]`, ...) instead of recomputing the full
    `shl+add+mad.wide.s32` index arithmetic 8 times over. Reviving
    `_partial_factor` (previously hardcoded to 1 - see its own docstring
    for the correctness bug that discovery surfaced and how it's now
    fixed) plus this address-reuse trick in `addr_of`
    (`_unroll_addr_coeff`/`self._unroll_subst`): whenever an unrolled
    copy's index is provably affine in the loop variable, its address is
    copy 0's cached address plus a compile-time-constant byte offset.
    Measured on the real GPU runner: a02 ~25us -> ~20-21us, 40 registers,
    and byte-identical PTX on every other open kernel (none has a nested
    loop already proven safe by the warp-reduce pass, which is what
    gates this factor on at all).
    """

    def test_unrolled_copies_reuse_base_address_with_immediate_offset(self):
        inner = Loop(
            axis=Axis("k", Dim(32)),
            body=(
                Load(names=("a",), input="kt", index=(Var("k"), Var("row")), dtype=F32),
                Load(names=("b",), input="q", index=(Var("k"),), dtype=F32),
                Assign(name="p", op=ElementwiseImpl("multiply"), args=("a", "b"), dtype=F32),
                Accum(name="acc", value="p", op=ElementwiseImpl("add"), dtype=F32),
            ),
            seed=True,
        )
        outer_body = (
            inner,
            Load(names=("vv",), input="v", index=(Var("row"), Var("ch")), dtype=F32),
            Assign(name="out_v", op=ElementwiseImpl("add"), args=("acc", "vv"), dtype=F32),
            Write(output="out", index=(Var("row"), Var("ch")), values=("out_v",), value_dtype=F32),
        )
        outer = Loop(axis=Axis("row", Dim(16)), body=outer_body)
        tile = Tile(axes=(Axis("b0", Dim(1)), Axis("ch", Dim(32))), block_threads=32, body=(outer,))
        kop = KernelOp(name="k_addr_reuse", body=(tile,))
        ptx = _compile(
            kop,
            shapes={"kt": (32, 16), "q": (32,), "v": (16, 32), "out": (16, 32)},
            dtypes={"kt": "f32", "q": "f32", "v": "f32", "out": "f32"},
            args=["kt", "q", "v", "out"],
            meta={"block": [32, 1, 1], "grid": [1, 1, 1]},
        )
        # 8-way unrolled (16 trips / factor 8 -> 2 physical trips). Inside
        # the real loop body (not the one-time kt/q cache-fill prologue,
        # which has its own address arithmetic), only ONE mad.wide.s32
        # per global buffer (v, out) - the other 7 copies of each reuse
        # that same address plus an immediate byte offset instead of
        # recomputing it.
        loop_body = ptx.split("LOOP_")[1]
        self.assertEqual(loop_body.count("mad.wide.s32"), 2)
        for off in (128, 256, 384, 512, 640, 768, 896):
            self.assertIn(f"+{off}]", loop_body)
        # kt (shared) gets the same treatment: one mad.lo.s32 for its
        # address, then +4, +8, ... +28 (one f32 element apart) for the
        # other 7 copies.
        for off in (4, 8, 12, 16, 20, 24, 28):
            self.assertIn(f"+{off}];", loop_body)

    def test_single_trip_loop_has_no_address_reuse_offsets(self):
        # Regression guard: a loop with NO nested warp-reduce loop must
        # never get a factor>1 treatment (and therefore no immediate
        # address-reuse offsets) - _partial_factor only fires when the
        # nested loop is already in self.warp_reduce_loops.
        body = (
            Load(names=("vv",), input="v", index=(Var("row"),), dtype=F32),
            Write(output="out", index=(Var("row"),), values=("vv",), value_dtype=F32),
        )
        outer = Loop(axis=Axis("row", Dim(16)), body=body)
        tile = Tile(axes=(Axis("b0", Dim(1)), Axis("ch", Dim(32))), block_threads=32, body=(outer,))
        kop = KernelOp(name="k_no_addr_reuse", body=(tile,))
        ptx = _compile(
            kop,
            shapes={"v": (16,), "out": (16,)},
            dtypes={"v": "f32", "out": "f32"},
            args=["v", "out"],
            meta={"block": [32, 1, 1], "grid": [1, 1, 1]},
        )
        self.assertNotIn("+128]", ptx)


class VectorizedBufferCacheFill(unittest.TestCase):
    """The cache-fill loop that _plan_buffer_cache/_emit_buffer_cache
    stages a read-only buffer through (see UniformBufferSharedMemoryCache)
    now fills it with one `ld.global.nc.v4.f32`/`st.shared.v4.f32` pair
    per thread per round instead of a scalar float at a time - 4x fewer
    loop trips to move the same bytes, with the shared-memory layout and
    every later read of it completely unchanged (each element still lands
    at the exact byte offset it always did). Safe whenever the buffer is
    plain f32 and its element count divides evenly by block_threads*4 -
    falls back to the original scalar loop otherwise (see
    UniformBufferSharedMemoryCache's own test, whose buffer size
    deliberately doesn't divide evenly and must keep compiling to a
    scalar fill). Measured on the real GPU runner: a02 ~25us (post
    address-reuse) -> ~20us. a01's cache-fill also gets vectorized (its
    K buffer divides evenly too), with the same register count (82) and
    PASS at the same ~120us as before - a real structural change with a
    measured-neutral result there, not a regression.
    """

    def test_fillable_by_four_uses_vectorized_load_and_store(self):
        body = (
            Loop(
                axis=Axis("a1", Dim(8)),
                body=(
                    Load(names=("kv",), input="kt", index=(Var("a1"),), dtype=F32),
                    Accum(name="acc", value="kv", dtype=F32),
                ),
            ),
            Write(output="out", index=(Var("row"),), values=("acc",), value_dtype=F32),
        )
        tile = Tile(axes=(Axis("row", Dim(4)),), block_threads=4, body=body)
        kop = KernelOp(name="k_veccache", body=(tile,))
        ptx = _compile(
            kop,
            shapes={"kt": (16,), "out": (4,)},
            dtypes={"kt": "f32", "out": "f32"},
            args=["kt", "out"],
            meta={"block": [4, 1, 1], "grid": [1, 1, 1]},
        )
        # 16 elements / (4 threads * 4-wide) == 1 round exactly.
        self.assertIn("ld.global.nc.v4.f32", ptx)
        self.assertIn("st.shared.v4.f32", ptx)

    def test_not_fillable_by_four_stays_scalar(self):
        # Mirrors UniformBufferSharedMemoryCache's own fixture size
        # (8 elements / (4 threads * 4) does not divide evenly) - must
        # keep compiling to the original scalar fill, not silently drop
        # a tail of elements.
        body = (
            Loop(
                axis=Axis("a1", Dim(8)),
                body=(
                    Load(names=("kv",), input="kt", index=(Var("a1"),), dtype=F32),
                    Accum(name="acc", value="kv", dtype=F32),
                ),
            ),
            Write(output="out", index=(Var("row"),), values=("acc",), value_dtype=F32),
        )
        tile = Tile(axes=(Axis("row", Dim(4)),), block_threads=4, body=body)
        kop = KernelOp(name="k_veccache_tail", body=(tile,))
        ptx = _compile(
            kop,
            shapes={"kt": (8,), "out": (4,)},
            dtypes={"kt": "f32", "out": "f32"},
            args=["kt", "out"],
            meta={"block": [4, 1, 1], "grid": [1, 1, 1]},
        )
        self.assertNotIn("ld.global.nc.v4.f32", ptx)
        self.assertIn("ld.global.nc.f32", ptx)


if __name__ == "__main__":
    unittest.main()
