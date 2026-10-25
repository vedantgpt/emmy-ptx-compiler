# Video script (3–5 minutes, your voice)

Record this in one take. Do not read as if reciting a generated essay.
Have `compile.py` open. Point at the functions you name.

## 0:00–0:25 — what you built

“This is a Kernel-IR to PTX compiler. The contract is
`python compile.py path/to/kernel.ir.json` writing PTX to stdout.

It does not call Emmy’s CUDA renderer or nvcc. Emmy is only used to
deserialize the IR through `Graph.from_dict`. Parameter order comes from
the sibling `meta.json` `arg_order`, because Emmy can reorder buffers
between pipeline stages.”

## 0:25–1:20 — load, ABI, tile guard

Open `compile_bundle`, `arg_order_for`, `prologue`, `emit_tile`.

“Load: find the `KernelOp`, take its name as the `.entry`, take
`arg_order` from meta.

Prologue: load each `.param .u64`, `cvta.to.global`, then
`gid = cta * ntid + tid`.

The launch can have more threads than the Tile has cells — p00 is 64
cells on a 256-thread block. That `if (tid < N)` is not in the IR. I
derive N from the Tile axes and emit `setp.ge` plus a branch when
`N % block_threads != 0` or when `aux_threads` is set. Extra threads
must not load out of bounds.”

## 1:20–2:20 — walking the IR

Open `stmt` and one of `emit_load`, `emit_write`, `emit_strided`,
`emit_warpshuffle`.

“The walker is one method per unmarked Kernel-IR node: Tile, Loop,
StridedLoop, Cond, Load, Write, Assign, Accum, Init, Let, Smem, Sync,
WarpShuffle, TreeHalve, and the three cp.async ops.

Loads and writes honor vector form from the IR — `ld.global.v4.f32`
when one Load binds four names, or `st.global.v2`/`v4` on a grouped
Write. Atomic writes are `red.global.add.f32`.

Reductions: a StridedLoop plus Accum, then `shfl.sync.bfly` with
`activemask` so inactive lanes do not participate, then TreeHalve over
shared memory with `bar.sync` between steps.

Identities are generic: add starts at 0, max at -1e30. I do not
hardcode a problem name.”

## 2:20–3:20 — peepholes that stay general

Open `_try_fma`, `_try_vec_load`, `_should_unroll`, the 48 KB smem cap.

“Peepholes are local and architecture-driven, not per-kernel:

- mul then add, or mul then accum, becomes `fma.rn.f32`
- consecutive same-axis Loops with no nested loop get fused
- unroll only innermost straight-line bodies, or when the IR sets the
  unroll flag, and only while `ops * trips` stays under 1536 so I-cache
  and ptxas stay healthy
- two or four consecutive f32 loads of the same buffer become `ld.v2`
  or `ld.v4`
- static shared memory above 48 KB becomes `.extern .shared` because
  that is the PTX static cap, not a problem size
- `.reqntid` comes from Tile.block_threads or meta.block”

## 3:20–4:10 — what you would say if they ask about speed

“Correctness is the score. Speed is the last tiebreak. a01 sits at 255
registers because the IR unrolls a 64-trip inner product; that is an
occupancy limit on the 5090’s 64K register file, not a bug I papered
over. I will not emit problem-specific schedules.”

## 4:10–end — close

“Hidden problems are the same five categories with different shapes.
The compiler only looks at the IR and the sibling meta.json. That is
the whole submission.”

## If you freeze

Name the function on screen and say what IR node it implements.
Do not apologize. Do not mention the assistant.
