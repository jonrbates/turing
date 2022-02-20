import unittest
import torch
from turing.wcm.simulator import Simulator

class TestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.tx = Simulator(T=17)   

    def assertTensorsEqual(self, expected, actual, msg=""):        
        if not torch.equal(expected, actual):
            self.fail(f'Not equal: {msg}')