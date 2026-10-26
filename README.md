# Kernel IR → PTX compiler

Emmy Hackathon 2026 entry. A **program**, not a folder of PTX files.

```text
Kernel IR JSON  +  sibling meta.json
        │
        ▼
   compile.py          (this repo)
        │
        ▼
      PTX stdout       (.version 8.0 / .target sm_89)
        │
        ▼
      ptxas  →  GPU binary
```

```bash
python compile.py path/to/kernel.ir.json > out.ptx
```

Emmy is used only to **load** the IR (`Graph.from_dict`). This compiler does **not** call Emmy’s CUDA renderer, `emmy compile`, or `nvcc`. Parameter order comes from the sibling `meta.json` `arg_order`, because Emmy can reorder buffers between pipeline stages.

- Feature map for judges: [FEATURES.md](FEATURES.md)
- Video notes: [WALKTHROUGH.md](WALKTHROUGH.md)

## Requirements

| | |
|---|---|
| Python | **3.12 or newer** (Emmy’s Kernel IR only deserializes through `emmy-ml==0.3.11`) |
| GPU | **Not required** to compile |
| CUDA / `nvcc` | **Not required** |
| `ptxas` | Optional, only if you want to assemble PTX locally |
| Torch | **Not required** to compile. Only needed if you generate new IRs yourself |

A hosted runner at `http://runner.riftstack.ai:48701` can assemble PTX with `ptxas` and run it on an RTX 5090 (`sm_120`). PTX written for `sm_89` runs there unchanged.

## Setup

```bash
git clone https://github.com/vedantgpt/emmy-ptx-compiler.git
cd emmy-ptx-compiler

python -m venv .venv
```

Activate the venv:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Install:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt`:

```text
emmy-ml==0.3.11
numpy
requests
```

That is enough to read every Kernel IR and run this compiler. No Torch, no CUDA toolkit.

### Optional: open-set problems

This repo is the compiler. The 15 practice kernels live in the official starter:

```bash
git clone https://github.com/cloudrift-ai/emmy-hackathon-2026.git
```

Each problem folder has:

| file | role |
|---|---|
| `*.ir.json` | **input** to `compile.py` |
| `*.ir.txt` | same IR, pretty-printed |
| `meta.json` | launch geometry + `arg_order` (read automatically if it sits next to the IR) |
| `*.ref.cu` | Emmy’s CUDA C++ reference (do not chain this) |
| `expected.npz` | golden outputs for the runner |

## Compile a kernel

Stdout must be **PTX only**. Print logs to stderr.

```bash
python compile.py path/to/kernel.ir.json > out.ptx
```

Example, using a problem from the official starter:

```bash
python compile.py ../emmy-hackathon-2026/problems/p00_scalar_add/k_add_pointwise.ir.json > out.ptx
```

If `problems/` is already next to `compile.py`:

```bash
python compile.py problems/r01_array_sum/k_sum_1_reduce.ir.json > out.ptx
```

Wrong usage (compiler exits 2):

```bash
python compile.py
# usage: python compile.py path/to/kernel.ir.json
```

### What the compiler reads

1. `argv[1]` — Kernel IR JSON
2. Sibling `meta.json` — `arg_order`, `block`, `grid` for the matching `.entry`
3. The `KernelOp` in the graph — entry name is `KernelOp.name`

### What it prints

A PTX module that starts with:

```ptx
.version 8.0
.target sm_89
.address_size 64
```

`.target` becomes `sm_90` only if the IR contains Hopper-only nodes (TMA, mbarrier, wgmma, `setmaxnreg`).

## Check the PTX locally

### Unit tests (no GPU)

```bash
python -m unittest discover -s tests -q
```

| test | what it checks |
|---|---|
| `tests/test_open_set.py` | every vendored `problems/*/*.ir.json` emits a `.version` / `sm_89` module (skips if `problems/` is missing) |
| `tests/test_no_hardcode.py` | `compile.py` does not name a problem id, shape table, or kernel entry |
| `tests/test_collective_guard.py` | leftover threads do not `bra` around `bar.sync` |
| `tests/test_extra_isa.py` | synthetic tensor-core + TMA kernel contains `ldmatrix`, `mma.sync`, TMA, mbarrier |

Print PTX for the extra-ISA kernel (no GPU):

```bash
python examples/extra_isa.py
```

### Assemble with `ptxas` (optional)

If CUDA is installed:

```bash
ptxas -arch=sm_89 -o out.cubin out.ptx
```

Windows, typical path:

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\ptxas.exe
```

### Run on the hosted GPU (optional)

The official starter’s `submit.py` compiles with **this** `compile.py`, sends PTX to the runner, launches under `meta.json`, and compares `expected.npz`:

```bash
python submit.py p00_scalar_add --compiler compile.py
python submit.py r01_array_sum --compiler /path/to/emmy-ptx-compiler/compile.py
```

Exit code 0 means `ptxas` accepted the PTX, the entry name matched, and outputs were within `rtol` / `atol`.

## Local studio

```bash
python tools/studio.py
```

Opens `http://127.0.0.1:8765` (override with `EMMY_STUDIO_PORT`). The page compiles a Kernel IR, shows the PTX, and can call the runner if `submit.py` and `problems/` are present.

## How the compiler is structured

`compile.py` is the whole submission. One KernelOp in, one `.visible .entry` out.

| piece | job |
|---|---|
| `compile_bundle` / `compile_ir` | load JSON, find `KernelOp`, take ABI from `meta.json` |
| `arg_order_for` | parameter order from `meta.json`, not from IR `inputs` |
| `Emitter` | virtual registers (`%f`, `%r`, `%rd`, `%p`). `ptxas` allocates for real |
| `Compiler.run` | header, shared memory, signature, prologue, body |
| `prologue` | `ld.param` + `cvta.to.global`, 3-D tid/cta → linear gid |
| `emit_tile` | axis decode + leftover-thread guard |
| `stmt` | one IR node kind → one `emit_*` |
| peepholes | FMA, CSE, same-axis loop fusion, bounded unroll, `ld.v2`/`ld.v4` |

**Thread guard.** The launch can have more threads than the Tile has cells (p00: 64 cells, 256-thread block). That `if (tid < N)` is not in the IR. Extra threads must not load out of bounds. If the body has `bar.sync` / TreeHalve / `cp.async.wait`, leftover lanes stay in the collective and only global `ld`/`st` are predicated — branching around a barrier deadlocks on Volta+.

**Shared memory.** Static `.shared` up to 48 KB. Above that, `.extern .shared` so `meta.smem_bytes` sizes the pool at launch.

**No problem-id table.** Identities are generic (add starts at 0, max at `-1e30`). Hidden kernels are the same five categories with different shapes.

## What it lowers

### Open-set / hidden-set kinds (the score)

Pointwise → reductions → matmul → fused → attention-like.

Tile, Loop, StridedLoop, Cond, Load, Write, Assign, Accum, Init, Let, Select, WarpShuffle, TreeHalve, Smem, Sync, `cp.async` + commit/wait, vector `ld`/`st`, atomic `red.global.add.f32`, 3-D launch, FMA.

### Extra ISA (not in the open set)

f16 / bf16 / f64 load-store, i32 assigns, `ldmatrix.sync`, `mma.sync`, fragment ops, TMA, mbarrier, wgmma, `setmaxnreg`. See [FEATURES.md](FEATURES.md).

## Repository

```text
compile.py              IR → PTX  (the submission)
requirements.txt        emmy-ml==0.3.11, numpy, requests
FEATURES.md             node → PTX map for review
WALKTHROUGH.md          3–5 min video script
examples/extra_isa.py   print tensor-core / TMA PTX
tests/                  open-set, no-hardcode, barrier guard, extra ISA
tools/studio.py         local IR → PTX UI
tools/make_proxy_set.py extra local practice IRs
tools/make_world_set.py extra local practice IRs
tools/run_all_irs.py    compile every IR we have
tools/run_proxy.py      compile + optional runner on proxy problems
```

`compile.py` is standalone. The rest of the layout is optional for evaluation.

## Contract the judges run

```bash
python compile.py path/to/kernel.ir.json > out.ptx
```

- Entry name = `KernelOp.name` in the IR
- Parameters = `meta.json` `arg_order` (buffers as `.u64` pointers)
- Correct under the launch in `meta.json`
- Results within that problem’s `rtol` / `atol`
- PTX must be ASCII; `.version` must be new enough for `sm_89` (this compiler uses **8.0**)
