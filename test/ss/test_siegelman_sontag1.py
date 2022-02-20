import torch
from test.ss.helper import TestCase
from turing.ss.networks import SiegelmannSontag1

class TestSiegelmannSontag1(TestCase):

    def test_solve_for_beta_and_gamma(self):
        s, p = 2, 1
        tx = SiegelmannSontag1(s, p)
        # low/high values in U_t range of Lemma 6.2
        low_value = (-8*p**2+2)
        high_value = 1

        x = low_value * torch.ones((s*9**p, 8*p+s))
        y = torch.randint(low=-5, high=5, size=(s*9**p, 4*p+s), dtype=torch.float32)
        row = 0
        for i, z in tx.configuration_detector.generate_i_z():
            x[row, 8*p+z] = 1
            slice = tx.configuration_detector.convert_to_tensor_indices(i)
            x[row, slice] = high_value
            row += 1

        cd_out = tx.configuration_detector(x)
        self.assertEqual(x.size(0), torch.linalg.matrix_rank(cd_out))
        beta, gamma = tx._solve_for_beta_and_gamma(cd_out, y)
        self.assertTensorsClose(y[:,4*p:], cd_out @ beta.T)
        self.assertTensorsClose(y[:,:4*p], cd_out @ gamma.T)
