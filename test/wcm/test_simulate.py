import unittest
from turing.wcm.simulator import Simulator


class TestWCMSimulate(unittest.TestCase):

    def setUp(self):
        self.sim = Simulator(T=17)

    def test_balanced_simple(self):
        result = self.sim.simulate('B()E')
        self.assertEqual(result, 'T')

    def test_balanced_nested(self):
        sim = Simulator(T=30)
        result = sim.simulate('B(())E')
        self.assertEqual(result, 'T')

    def test_unbalanced_open(self):
        result = self.sim.simulate('B(E')
        self.assertEqual(result, 'F')

    def test_unbalanced_close_open(self):
        result = self.sim.simulate('B)(E')
        self.assertEqual(result, 'F')

    def test_empty_tape(self):
        result = self.sim.simulate('BE')
        self.assertEqual(result, 'T')


if __name__ == '__main__':
    unittest.main()
