import torch
from test.wcm.helper import TestCase
from turing.wcm.networks import FullAdder

class TestAdder(TestCase):  
 
    def test_adder(self):
        tx = self.tx

        for i in range(10):
            for j in range(10):
                if i + j > tx.T:
                    continue
                o = torch.cat([
                    torch.Tensor(tx.Bin(i)), 
                    torch.Tensor(tx.Bin(j)),
                    torch.Tensor([0]),
                    torch.rand((tx.w-2*tx.w_pos-1))
                ])
                for k in range(tx.w_pos-1, -1, -1):
                    o = FullAdder(tx.w, i=k, j=tx.w_pos+k, k=2*tx.w_pos)(o)
                sum = self.tx.Bin_inverse(o[:tx.w_pos])
                self.assertEqual(i + j, sum)