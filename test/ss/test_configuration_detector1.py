import torch
from test.ss.helper import TestCase
from turing.ss.networks import ConfigurationDetector1

class TestConfigurationDetector1(TestCase):

    def test_generate_i(self):
        s, p = 3, 2
        cd = ConfigurationDetector1(s, p)
        l = [_ for _ in cd.generate_i()]
        self.assertEqual(9**p, len(l))
 
    def test_generate_i_top(self):
        s, p = 3, 2
        cd = ConfigurationDetector1(s, p)
        l = [_ for _ in cd._generate_i_top((None, 2))]
        self.assertEqual(2, len(l))

        l = [_ for _ in cd._generate_i_top((3, 2))]
        self.assertEqual(4, len(l))

    def test_forward(self):
        s, p = 3, 2
        cd = ConfigurationDetector1(s, p)
        noisy_sub_nonempty, noisy_sub_top, state = torch.zeros(4*p), torch.zeros(4*p), torch.zeros(s)        
        x = torch.cat([
            noisy_sub_top,
            noisy_sub_nonempty,            
            state
        ])
        
        o = cd(x)
        self.assertEqual((s * 9**p,), o.shape)