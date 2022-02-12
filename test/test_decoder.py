import random
import torch
from test.helper import BaseTestCase
from turing.networks import Transition, FullAdder, PreprocessForAdder

class TestDecoder(BaseTestCase):

    def test_decode_step(self):
        tx = self.tx
        l = 4
        for i in random.choices(range(tx.T), k=20):
            for (z, a), (z_next, u, q) in tx.delta.items():
                o = Transition(tx.slices, tx.delta, tx.states, tx.alphabet, tx.z2i, tx.a2i)(tx.h(z, a, i, l))
                o = PreprocessForAdder(tx.slices)(o)
                for k in range(tx.w_pos-1, -1, -1):
                    o = FullAdder(d_in=tx.w+tx.w_pos, i=tx.pos3_+k, j=tx.w+k, k=tx.scr5_)(o)
                self.assertTensorsEqual(tx.one_states(z_next), o[tx.st_:tx.sym1_], "st")
                self.assertTensorsEqual(torch.zeros_like(o[tx.sym1_:tx.sym2_]), o[tx.sym1_:tx.sym2_], "sym1")
                self.assertTensorsEqual(tx.one_alphabet(u), o[tx.sym2_:tx.pos1_], "sym2")
                self.assertTensorsEqual(torch.Tensor(tx.Bin(i)), o[tx.pos1_:tx.pos2_], "pos1")
                self.assertTensorsEqual(torch.Tensor(tx.Bin(l)), o[tx.pos2_:tx.pos3_], "pos2")
                self.assertTensorsEqual(torch.Tensor(tx.Bin(l+q)), o[tx.pos3_:tx.scr1_], "pos3")
                self.assertTensorsEqual(torch.Tensor([0, 1]) if q == 1 else torch.Tensor([1, 0]), o[tx.scr5_:tx.w], "scr5")