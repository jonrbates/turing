import random
import torch
from test.helper import BaseTestCase
from turing.networks import IndicateVisitedPosition

class TestVisited(BaseTestCase):     
 
    def test_is_visited(self):
        """Claim 6.4
        """
        tx = self.tx
        f = IndicateVisitedPosition(tx.slices)

        for iprime in range(tx.T):
            X = self.generate_input(iprime)
            o = f(X)
            o = o[-1, :]
            e = torch.zeros_like(o[tx.scr4_:tx.scr5_])
            if iprime > 0:
                e[0] = 1
            self.assertTensorsEqual(torch.zeros_like(o[tx.scr1_:tx.scr4_]), o[tx.scr1_:tx.scr4_], "scr1")
            self.assertTensorsEqual(torch.zeros_like(o[tx.scr3_:tx.scr4_]), o[tx.scr3_:tx.scr4_], "scr3")
            self.assertTensorsEqual(e, o[tx.scr4_:tx.scr5_], "scr4")

    def generate_input(self, iprime: int=0):
        tx = self.tx
        X = torch.zeros(tx.T, tx.w)
        for j in range(tx.T):
            X[j, tx.pos2_:tx.pos3_] = torch.Tensor(tx.Bin(j))
        if iprime == 0:
            imiss = random.choice(range(tx.T))
            X[-1, tx.pos3_:tx.scr1_] = torch.Tensor(tx.Bin(imiss))
            X[imiss, tx.pos2_:tx.pos3_] = torch.Tensor(tx.Bin(0))
        else:
            X[-1, tx.pos3_:tx.scr1_] = X[iprime, tx.pos2_:tx.pos3_]
        return X