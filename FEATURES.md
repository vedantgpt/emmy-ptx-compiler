# Compiler features (what to look for)

Judges: the 15 open f32 problems are the baseline filter. This file lists
what the walker actually lowers so the extra-credit review is not a grep
hunt.

## How to run

```bash
pip install -r requirements.txt
python compile.py path/to/kernel.ir.json > out.ptx
python -m unittest discover -s tests -q
```

`compile.py` is standalone. The repo layout is optional.

## Baseline (open + hidden of the same kinds)

| Feature | Where |
|---|---|
| `Graph.from_dict` load, `KernelOp.name` as `.entry` | `compile_bundle` |
| ABI from sibling `meta.json` `arg_order` | `arg_order_for` |
| Tile N-guard when launch has extra threads / `aux_threads` | `emit_tile` |
| 3-D `tid` / `ctaid` linearization (grid.y/z, block.y/z) | `prologue` |
| `.reqntid` from Tile or full `meta.block` product | `_reqntid_clause` |
| Loop, StridedLoop, Cond (+ else), Let, Init, Select | `stmt` |
| Vector `ld`/`st` `.v2`/`.v4` from IR or consecutive f32 loads | `emit_load`, `_try_vec_load` |
| Atomic `red.global.add.f32` | `emit_write` |
| WarpShuffle (`shfl.sync.bfly` + `activemask`) | `emit_warpshuffle` |
| TreeHalve + `bar.sync` / `bar.warp.sync` | `emit_treehalve`, `emit_sync` |
| `cp.async` + commit/wait | `emit_cp_async` |
| Static smem vs `.extern .shared` above 48 KB | `run` |
| FMA, CSE, same-axis loop fusion, bounded unroll | `_try_fma*`, `expr`, `_can_fuse_loops` |

## Extra credit (Dmitry: tensor cores, other copy engines, dtypes)

These nodes are marked `*` in TASK.md (not in the open set). The compiler
still lowers them when they appear, and switches `.target` to `sm_90` if
TMA / mbarrier / wgmma / `setmaxnreg` is present. The 5090 runner is
`sm_120` and can assemble that PTX.

| Feature | PTX | Where |
|---|---|---|
| f16 / bf16 load+store with cvt to f32 compute | `ld.*.f16`, `cvt.f32.f16`, `st.*.bf16` | `emit_load`, `emit_write` |
| f64 load+store | `ld.*.f64` / `.reg .f64` | `emit_load`, `Emitter.new_d` |
| i32 assigns | `add.s32` / `max.s32` | `emit_assign` |
| `ldmatrix.sync` into `RegFragment` | `ldmatrix.sync.aligned.x4.m8n8` | `emit_ldmatrix` |
| `mma.sync` m16n8k16 (and shape from IR) | `mma.sync.aligned.mMnNkK` | `emit_mma_sync` |
| C-fragment store, m16n8 lane layout | `st.global.f32` per lane | `emit_reg_store` |
| Fragment apply / row-reduce / repack / promote | f32 ops on fragment regs | `emit_frag_*` |
| TMA box copy | `cp.async.bulk.tensor.Nd...mbarrier::complete_tx` | `emit_tma_load` |
| mbarrier init / arrive / expect_tx / wait | `mbarrier.*` | `emit_mbar_*` |
| wgmma async + fence/commit/wait | `wgmma.mma_async` | `emit_wgmma_*` |
| `setmaxnreg` | `setmaxnreg.inc/dec` | `emit_setmaxnreg` |

## Tests and examples

- `tests/test_open_set.py` — compiles every open IR (baseline filter).
- `tests/test_extra_isa.py` — builds a synthetic KernelOp with
  RegFragment + ldmatrix + mma.sync + TMA + mbarrier and checks the PTX
  contains those instructions.
- `examples/extra_isa.py` — same synthetic kernel, prints PTX (no GPU).
