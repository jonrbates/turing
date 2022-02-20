import random
import torch
from test.wcm.helper import TestCase
from turing.wcm.networks import GetInitialSymbol

class TestCrossAttention(TestCase):

    def test_single_layer(self):
        """Claim C.7
        """
        tx = self.tx
        
        tape = "B(())E"
        z = random.choice(tx.states)
        u = random.choice(tx.alphabet)
        i = random.choice(range(tx.T))
        l_minus = random.choice(range(len(tape)))
        l = random.choice(range(len(tape)))
        iprime = 2
        uprime = random.choice(tx.alphabet)

        m = len(tape)
        xl = tape[l]

        f = GetInitialSymbol(tx.slices)

        x, E = self.generate_input(tape, z, u, i, l_minus, l, iprime, uprime)
        o = f(x, E)
        o = o[-1, :]
        self.assertTensorsEqual(tx.one_states(z), o[tx.st_:tx.sym1_], "st")
        self.assertTensorsEqual(tx.one_alphabet(u), o[tx.sym2_:tx.pos1_], "sym2")
        self.assertTensorsEqual(torch.Tensor(tx.Bin(i)), o[tx.pos1_:tx.pos2_], "pos1")
        self.assertTensorsEqual(torch.Tensor(tx.Bin(l_minus)), o[tx.pos2_:tx.pos3_], "pos2")
        self.assertTensorsEqual(torch.Tensor(tx.Bin(l)), o[tx.pos3_:tx.scr1_], "pos3")
        if iprime == 0:
            self.assertTensorsEqual(torch.zeros_like(o[tx.scr1_:tx.scr2_]), o[tx.scr1_:tx.scr2_], "scr1")                    
        else:
            self.assertTensorsEqual(tx.one_alphabet(uprime), o[tx.scr1_:tx.scr2_], "scr1")
        if l > m:
            self.assertTensorsEqual(torch.zeros_like(o[tx.scr2_:tx.scr3_]), o[tx.scr2_:tx.scr3_], "scr2")                    
        else:
            self.assertTensorsEqual(tx.one_alphabet(xl), o[tx.scr2_:tx.scr3_], "scr2")
        if iprime == 0:
            self.assertTensorsEqual(torch.tensor(0.), o[tx.scr4_], "scr4_1")
        else:
            self.assertTensorsEqual(torch.tensor(1.), o[tx.scr4_], "scr4_1")
        if l > m:
            self.assertTensorsEqual(torch.tensor(0.), o[tx.scr4_+1], "scr4_2")
        else:
            self.assertTensorsEqual(torch.tensor(1.), o[tx.scr4_+1], "scr4_2")

    def generate_input(self, tape, z, u, i, l_minus, l, iprime, uprime):
        tx = self.tx

        # Encoder
        E = tx.encode_tape(tape)

        # Decoder input        
        x = torch.zeros(tx.w, )  
        x[tx.st_:tx.sym1_] = tx.one_states(z)
        x[tx.sym2_:tx.pos1_] = tx.one_alphabet(u)
        x[tx.pos1_:tx.pos2_] = torch.Tensor(tx.Bin(i))
        x[tx.pos2_:tx.pos3_] = torch.Tensor(tx.Bin(l_minus))
        x[tx.pos3_:tx.scr1_] = torch.Tensor(tx.Bin(l))        

        if iprime == 0:
            x[tx.scr1_:tx.scr2_] = 0
            x[tx.scr4_] = 0
        else:
            x[tx.scr1_:tx.scr2_] = tx.one_alphabet(uprime)
            x[tx.scr4_] = 1                  

        x = x.unsqueeze(0)
        return x, E