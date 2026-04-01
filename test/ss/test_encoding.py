import unittest
from turing.ss.simulator import encoding_function


class TestEncodingFunction(unittest.TestCase):

    def test_empty_string(self):
        self.assertEqual(encoding_function('', base=4, p=1), 0)

    def test_empty_string_ss1_params(self):
        self.assertEqual(encoding_function('', base=40, p=2), 0)

    def test_single_zero_base4(self):
        # a=[0], base=4, p=1/2
        # num = 4-1+4*0.5*(0-1) = 3-2 = 1, denom = 4
        # result = 1/4
        result = encoding_function([0], base=4, p=0.5)
        self.assertAlmostEqual(result, 0.25)

    def test_single_one_base4(self):
        # a=[1], base=4, p=1/2
        # num = 4-1+4*0.5*(1-1) = 3, denom = 4
        # result = 3/4
        result = encoding_function([1], base=4, p=0.5)
        self.assertAlmostEqual(result, 0.75)

    def test_two_symbols_base4(self):
        # a=[0,1], base=4, p=1/2
        # k=1: num=1, denom=4  → 1/4
        # k=2: num=3, denom=16 → 3/16
        # result = 1/4 + 3/16 = 7/16
        result = encoding_function([0, 1], base=4, p=0.5)
        self.assertAlmostEqual(result, 7 / 16)

    def test_single_zero_ss1_params(self):
        # a=[0], base=40, p=2
        # num = 40-1+4*2*(0-1) = 39-8 = 31, denom = 40
        result = encoding_function([0], base=40, p=2)
        self.assertAlmostEqual(result, 31 / 40)

    def test_single_one_ss1_params(self):
        # a=[1], base=40, p=2
        # num = 40-1+4*2*(1-1) = 39, denom = 40
        result = encoding_function([1], base=40, p=2)
        self.assertAlmostEqual(result, 39 / 40)

    def test_value_in_unit_interval(self):
        """Encoded value should be in [0, 1) for any binary string."""
        for s in [[0], [1], [0, 0], [0, 1], [1, 0], [1, 1], [0, 1, 0, 1]]:
            with self.subTest(s=s):
                val = encoding_function(s, base=4, p=0.5)
                self.assertGreaterEqual(val, 0)
                self.assertLess(val, 1)


if __name__ == '__main__':
    unittest.main()
