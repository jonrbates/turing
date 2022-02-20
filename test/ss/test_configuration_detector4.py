import torch
from test.ss.helper import TestCase
from turing.ss.networks import ConfigurationDetector4

class TestConfigurationDetector4(TestCase):

    def test_get_row_of_v(self):
        s, p = 3, 2
        cd = ConfigurationDetector4(s, p)
        self.assertEqual((0,0,1,0), cd._get_row_of_v(2))
        self.assertEqual(None, cd._get_row_of_v(4))

    def test_generate_v(self):
        s, p = 3, 4
        cd = ConfigurationDetector4(s, p)
        l = [_ for _ in cd._generate_v()] 
        self.assertEqual(3**p, len(l))
 
    def test_generate_v_z(self):
        s, p = 3, 2
        cd = ConfigurationDetector4(s, p)
        l = [_ for _ in cd.generate_v_z()]
        self.assertEqual(s*3**p, len(l))