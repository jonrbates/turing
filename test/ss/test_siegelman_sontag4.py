import torch
from test.ss.helper import TestCase
from turing.ss.networks import SiegelmannSontag4

class TestSiegelmannSontag4(TestCase):  
     
    def test_solve_for_beta_and_gamma(self):
        s, p = 3, 2
        d = s*3**p
        tx = SiegelmannSontag4(s, p)
        h = torch.randint(low=-10, high=10, size=(d, d), dtype=torch.float32)
        y_state = torch.randint(low=-10, high=10, size=(d,   s), dtype=torch.float32)
        y_stack = torch.randint(low=-10, high=10, size=(d, 4*p), dtype=torch.float32)
        y = torch.cat([y_state, y_stack], axis=1)
        beta, gamma = tx._solve_for_beta_and_gamma(h, y)
        self.assertTensorsClose(y_state.T, torch.mm(beta, h.T))
        self.assertTensorsClose(y_stack.T, torch.mm(gamma, h.T))    

