import unittest
import torch

class TestCase(unittest.TestCase):

    def setUp(self) -> None:
        pass
    
    def assertTensorsEqual(self, expected, actual, msg=""):        
        if not torch.equal(expected, actual):
            self.fail(f'Not equal: {msg}')

    def assertTensorsClose(self, expected, actual, atol=1e-3 , msg=""):        
        if not torch.allclose(expected, actual, atol=atol):
            res = (expected - actual).abs().sum()
            self.fail(f'Not equal by {res}: {msg}')