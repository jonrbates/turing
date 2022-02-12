import unittest
import torch
from turing.translators import Translator

class BaseTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.tx = Translator(T=17)   

    def assertTensorsEqual(self, expected, actual, msg=""):        
        if not torch.equal(expected, actual):
            self.fail(f'Not equal: {msg}')