import random
import torch
from test.helper import BaseTestCase
from turing.networks import BinarySearchStep

class TestBinarySearch(BaseTestCase):

    def Bin_j(self, i: int, j: int):
        pre = self.tx.Bin(i)
        return pre[:j] + [0]*(self.tx.w_pos-j)
 
    def test_single_layer(self):
        """Claim 6.5
        """
        tx = self.tx
        for j in range(tx.w_pos):
            f = BinarySearchStep(tx.slices, j)            
            for iprime in range(tx.T):
                X = self.generate_input(iprime, j)
                o = f(X)
                o = o[-1, :]
                self.assertTensorsEqual(torch.zeros_like(o[tx.scr1_:tx.scr2_]), o[tx.scr1_:tx.scr2_])
                if iprime == 0:
                    self.assertTensorsEqual(torch.zeros_like(o[tx.scr3_:tx.scr4_]), o[tx.scr3_:tx.scr4_])
                    self.assertTensorsEqual(torch.tensor(0.), o[tx.scr4_])
                else:
                    self.assertTensorsEqual(torch.Tensor(self.Bin_j(iprime, j+1)), o[tx.scr3_:tx.scr4_])
                    self.assertTensorsEqual(torch.tensor(1.), o[tx.scr4_])

    def generate_input(self, iprime: int=0, j: int=0):
        tx = self.tx
        X = torch.zeros(tx.T, tx.w)        
        for tau in range(tx.T):
            X[tau, tx.pos1_:tx.pos2_] = torch.Tensor(tx.Bin(tau))
            # use tau for l_{t-1}(x)
            l = tau
            X[tau, tx.pos2_:tx.pos3_] = torch.Tensor(tx.Bin(l))

        if iprime == 0:
            imiss = random.choice(range(1, tx.T))
            X[-1, tx.pos3_:tx.scr1_] = torch.Tensor(tx.Bin(imiss))
            X[imiss, tx.pos2_:tx.pos3_] = torch.Tensor(tx.Bin(0))
        else:
            X[-1, tx.pos3_:tx.scr1_] = X[iprime, tx.pos2_:tx.pos3_]

        X[-1, tx.scr1_:tx.scr2_] = 0
        if iprime == 0:
            X[-1, tx.scr3_:tx.scr4_] = 0
        else:
            X[-1, tx.scr3_:tx.scr4_] = torch.Tensor(self.Bin_j(iprime, j))
        if iprime == 0:
            X[-1, tx.scr4_:tx.scr5_] = 0
        else:
            X[-1, tx.scr4_] = 1

        return X