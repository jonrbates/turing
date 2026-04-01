import unittest
import torch
from turing.functional import saturated_relu


class TestSaturatedRelu(unittest.TestCase):

    def test_negative_input(self):
        x = torch.tensor([-1.0, -0.5, -100.0])
        result = saturated_relu(x)
        self.assertTrue(torch.equal(result, torch.zeros(3)))

    def test_zero(self):
        x = torch.tensor([0.0])
        result = saturated_relu(x)
        self.assertEqual(result.item(), 0.0)

    def test_one(self):
        x = torch.tensor([1.0])
        result = saturated_relu(x)
        self.assertEqual(result.item(), 1.0)

    def test_interior(self):
        x = torch.tensor([0.25, 0.5, 0.75])
        result = saturated_relu(x)
        self.assertTrue(torch.equal(result, x))

    def test_above_one(self):
        x = torch.tensor([1.5, 2.0, 100.0])
        result = saturated_relu(x)
        self.assertTrue(torch.equal(result, torch.ones(3)))

    def test_mixed(self):
        x = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0])
        expected = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0])
        result = saturated_relu(x)
        self.assertTrue(torch.equal(result, expected))


if __name__ == '__main__':
    unittest.main()
