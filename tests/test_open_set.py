import unittest
from pathlib import Path

import compile as C

ROOT = Path(__file__).resolve().parents[1]


class OpenSetCompile(unittest.TestCase):
    def test_every_open_ir_emits_sm89_ptx(self):
        problems = ROOT / "problems"
        if not problems.is_dir():
            self.skipTest("open problems/ not vendored in this checkout")
        n = 0
        for d in sorted(problems.iterdir()):
            irs = list(d.glob("*.ir.json")) if d.is_dir() else []
            if not irs:
                continue
            ptx = C.compile_ir(irs[0])
            self.assertTrue(ptx.startswith(".version"), d.name)
            self.assertIn(".target sm_89", ptx)
            self.assertIn(".visible .entry", ptx)
            n += 1
        self.assertEqual(n, 15)


if __name__ == "__main__":
    unittest.main()
