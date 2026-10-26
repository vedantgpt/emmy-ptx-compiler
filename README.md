# Kernel IR → PTX compiler

Emmy Hackathon 2026 entry.

```bash
pip install -r requirements.txt
python compile.py path/to/kernel.ir.json > out.ptx
```

`compile.py` walks Emmy Kernel IR (`Graph.from_dict`) and emits `.version 8.0` / `.target sm_89` PTX. It does not call Emmy’s CUDA renderer, `emmy compile`, or `nvcc`. Parameter order comes from the sibling `meta.json` `arg_order`.

Optional local studio:

```bash
python tools/studio.py
```

Walkthrough notes for the required video: [WALKTHROUGH.md](WALKTHROUGH.md).

**Features judges asked for after the open-set filter** (tensor cores, TMA/mbarrier, f16/bf16/f64, 2-D/3-D launch): [FEATURES.md](FEATURES.md).

```bash
python -m unittest discover -s tests -q
python examples/extra_isa.py
```
