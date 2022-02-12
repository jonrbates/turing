import random
import torch
from test.helper import BaseTestCase
from turing.networks import GetLastWrittenSymbol

class TestRetrieve(BaseTestCase):

    def test_single_layer(self):
        """Claim C.6
        """
        tx = self.tx
        f = GetLastWrittenSymbol(tx.slices)
        for iprime in range(tx.T): 
            X = self.generate_input(iprime)
            a = X[iprime, tx.sym2_:tx.pos1_]
            o = f(X)
            o = o[-1, :]
            self.assertTensorsEqual(torch.zeros_like(o[tx.st_:tx.sym1_]), o[tx.st_:tx.sym1_], "st")
            self.assertTensorsEqual(torch.Tensor(tx.Bin(tx.T-1)), o[tx.pos1_:tx.pos2_], "pos1")
            self.assertTensorsEqual(torch.zeros_like(o[tx.scr3_:tx.scr4_]), o[tx.scr3_:tx.scr4_], "scr3")
            if iprime == 0:
                self.assertTensorsEqual(torch.zeros_like(o[tx.scr1_:tx.scr2_]), o[tx.scr1_:tx.scr2_], "scr1")                    
                self.assertTensorsEqual(torch.tensor(0.), o[tx.scr4_], "scr4")
            else:
                self.assertTensorsEqual(a, o[tx.scr1_:tx.scr2_], "scr1")
                self.assertTensorsEqual(torch.tensor(1.), o[tx.scr4_], "scr4")

    def generate_input(self, iprime: int=0):
        tx = self.tx
        X = torch.zeros(tx.T, tx.w)        
        for tau in range(tx.T):
            a = random.choice(tx.alphabet)
            X[tau, tx.sym2_:tx.pos1_] = tx.one_alphabet(a)
            X[tau, tx.pos1_:tx.pos2_] = torch.Tensor(tx.Bin(tau))
            # use tau for l_{t-1}(x)
            l = tau
            X[tau, tx.pos2_:tx.pos3_] = torch.Tensor(tx.Bin(l))

        X[-1, tx.scr1_:tx.scr2_] = 0

        if iprime == 0:
            imiss = random.choice(range(1, tx.T))
            X[-1, tx.pos3_:tx.scr1_] = torch.Tensor(tx.Bin(imiss))
            X[imiss, tx.pos2_:tx.pos3_] = torch.Tensor(tx.Bin(0))
            X[-1, tx.scr3_:tx.scr4_] = 0
            X[-1, tx.scr4_] = 0
        else:
            X[-1, tx.pos3_:tx.scr1_] = X[iprime, tx.pos2_:tx.pos3_]
            X[-1, tx.scr3_:tx.scr4_] = torch.Tensor(tx.Bin(iprime))            
            X[-1, tx.scr4_] = 1                  

        return X