"""compile.py must not name a problem, shape table, or kernel entry."""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "compile.py").read_text(encoding="utf-8")
BANNED = [
    "p00_scalar_add", "p01_vector_add", "p02_scale_shift", "p03_relu_mul",
    "r01_array_sum", "r02_big_sum", "r03_mean", "r04_row_sum",
    "m01_matmul_512", "m02_matmul_rect",
    "f01_rmsnorm", "f02_row_softmax", "f03_l2norm",
    "a01_attention", "a02_attention_128",
    "k_add_pointwise", "k_sum_1_reduce", "k_matmul_2967be",
    "k_softmax_05263d", "k_matmul_softmax_reduce",
]


class NoHardcode(unittest.TestCase):
    def test_no_problem_ids(self):
        for token in BANNED:
            self.assertNotIn(token, SRC, f"hardcoded {token!r}")

    def test_no_shape_lookup(self):
        self.assertIsNone(re.search(r"PROBLEMS\s*=", SRC))
        self.assertNotIn("if problem ==", SRC)
        self.assertNotIn("if pid ==", SRC)


if __name__ == "__main__":
    unittest.main()
