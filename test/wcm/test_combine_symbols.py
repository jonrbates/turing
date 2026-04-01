import torch
from test.wcm.helper import TestCase
from turing.wcm.networks import GetV, ArrangeSymbols, CombineSymbols


class TestGetV(TestCase):

    def test_v1_when_scr4_is_1_0(self):
        """scr4 = [1, 0, 0] → v_1=1, v_2=0, v_3=0."""
        tx = self.tx
        h = torch.zeros(1, tx.w)
        h[0, tx.scr4_] = 1
        h[0, tx.scr4_ + 1] = 0
        f = GetV(tx.slices)
        out = f(h)[0]
        self.assertAlmostEqual(out[tx.scr4_].item(), 1.0)
        self.assertAlmostEqual(out[tx.scr4_ + 1].item(), 0.0)
        self.assertAlmostEqual(out[tx.scr4_ + 2].item(), 0.0)

    def test_v2_when_scr4_is_0_1(self):
        """scr4 = [0, 1, 0] → v_1=0, v_2=1, v_3=0."""
        tx = self.tx
        h = torch.zeros(1, tx.w)
        h[0, tx.scr4_] = 0
        h[0, tx.scr4_ + 1] = 1
        f = GetV(tx.slices)
        out = f(h)[0]
        self.assertAlmostEqual(out[tx.scr4_].item(), 0.0)
        self.assertAlmostEqual(out[tx.scr4_ + 1].item(), 1.0)
        self.assertAlmostEqual(out[tx.scr4_ + 2].item(), 0.0)

    def test_v3_when_scr4_is_0_0(self):
        """scr4 = [0, 0, 0] → v_1=0, v_2=0, v_3=1."""
        tx = self.tx
        h = torch.zeros(1, tx.w)
        h[0, tx.scr4_] = 0
        h[0, tx.scr4_ + 1] = 0
        f = GetV(tx.slices)
        out = f(h)[0]
        self.assertAlmostEqual(out[tx.scr4_].item(), 0.0)
        self.assertAlmostEqual(out[tx.scr4_ + 1].item(), 0.0)
        self.assertAlmostEqual(out[tx.scr4_ + 2].item(), 1.0)


class TestArrangeSymbols(TestCase):

    def test_v1_gates_scr1(self):
        """When v_1=1, scr1 should pass through (AND gate)."""
        tx = self.tx
        h = torch.zeros(1, tx.w)
        # Set v_1=1 (scr4[0]=1)
        h[0, tx.scr4_] = 1
        # Set scr1 to some symbol (first alphabet symbol)
        h[0, tx.scr1_] = 1
        empty_symbol = tx.one_alphabet("E")
        f = ArrangeSymbols(tx.slices, empty_symbol)
        out = f(h)[0]
        # scr1 should remain (gated by v_1)
        self.assertAlmostEqual(out[tx.scr1_].item(), 1.0)

    def test_v1_zero_blocks_scr1(self):
        """When v_1=0, scr1 should be zeroed out."""
        tx = self.tx
        h = torch.zeros(1, tx.w)
        h[0, tx.scr4_] = 0
        h[0, tx.scr1_] = 1
        empty_symbol = tx.one_alphabet("E")
        f = ArrangeSymbols(tx.slices, empty_symbol)
        out = f(h)[0]
        self.assertAlmostEqual(out[tx.scr1_].item(), 0.0)

    def test_v3_gates_empty_into_sym1(self):
        """When v_3=1, sym1 should become the empty symbol."""
        tx = self.tx
        h = torch.zeros(1, tx.w)
        # v_3=1 means scr4 = [0, 0, 1] (after GetV)
        h[0, tx.scr4_ + 2] = 1
        empty_symbol = tx.one_alphabet("E")
        f = ArrangeSymbols(tx.slices, empty_symbol)
        out = f(h)[0]
        sym1 = out[tx.sym1_:tx.sym2_]
        self.assertTrue(torch.allclose(sym1, empty_symbol))


class TestCombineSymbols(TestCase):

    def test_sym1_combines_scr1_scr2(self):
        """sym1 = sym1 + scr1 + scr2 after CombineSymbols."""
        tx = self.tx
        h = torch.zeros(1, tx.w)
        h[0, tx.scr1_] = 1.0
        h[0, tx.sym1_] = 0.0
        f = CombineSymbols(tx.slices)
        out = f(h)[0]
        # sym1[0] should be scr1[0] = 1 (sym1 was 0, scr2 was 0)
        self.assertAlmostEqual(out[tx.sym1_].item(), 1.0)

    def test_pos3_copied_to_pos2(self):
        """pos2 should equal pos3 after CombineSymbols."""
        tx = self.tx
        h = torch.zeros(1, tx.w)
        pos3_val = torch.tensor(tx.Bin(5), dtype=torch.float)
        h[0, tx.pos3_:tx.scr1_] = pos3_val
        f = CombineSymbols(tx.slices)
        out = f(h)[0]
        pos2 = out[tx.pos2_:tx.pos3_]
        self.assertTrue(torch.allclose(pos2, pos3_val))

    def test_scratch_zeroed(self):
        """Scratch space should be zeroed after CombineSymbols."""
        tx = self.tx
        h = torch.ones(1, tx.w)
        f = CombineSymbols(tx.slices)
        out = f(h)[0]
        scratch = out[tx.scr1_:]
        self.assertTrue(torch.all(scratch == 0))


if __name__ == '__main__':
    unittest.main()
