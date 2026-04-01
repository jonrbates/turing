import unittest
import torch
from torch import nn
from turing.functional import attention_forward


class TestAttentionForward(unittest.TestCase):

    def setUp(self):
        """Set up a small 2-key, 1-query attention problem."""
        self.E = 2  # embedding dim
        # Identity projections (no transformation)
        self.w_q = torch.eye(self.E)
        self.b_q = torch.zeros(self.E)
        self.w_k = torch.eye(self.E)
        self.b_k = torch.zeros(self.E)
        self.w_v = torch.eye(self.E)
        self.b_v = torch.zeros(self.E)
        # Null key/value
        self.k_0 = torch.zeros(self.E)
        self.v_0 = torch.zeros(self.E)

    def test_hard_max_selects_best_match(self):
        query = torch.tensor([[1.0, 0.0]])   # L=1
        key = torch.tensor([[1.0, 0.0],      # S=2; key0 matches query
                            [0.0, 1.0]])
        value = torch.tensor([[5.0, 0.0],
                              [0.0, 5.0]])
        out = attention_forward(
            query, key, value,
            self.w_q, self.b_q, self.w_k, self.b_k, self.w_v, self.b_v,
            self.k_0, self.v_0, use_hard_max=True,
        )
        # key0 has dot product 1, key1 has 0, null key has 0 → selects key0
        expected = torch.tensor([[5.0, 0.0]])
        self.assertTrue(torch.allclose(out, expected))

    def test_hard_max_tie_averages(self):
        query = torch.tensor([[1.0, 1.0]])
        # Both keys have the same dot product with query
        key = torch.tensor([[1.0, 0.0],
                            [0.0, 1.0]])
        value = torch.tensor([[2.0, 0.0],
                              [0.0, 4.0]])
        out = attention_forward(
            query, key, value,
            self.w_q, self.b_q, self.w_k, self.b_k, self.w_v, self.b_v,
            self.k_0, self.v_0, use_hard_max=True,
        )
        # Both score 1, null scores 0 → average of the two values
        expected = torch.tensor([[1.0, 2.0]])
        self.assertTrue(torch.allclose(out, expected))

    def test_soft_max_approximates_hard_max(self):
        query = torch.tensor([[1.0, 0.0]])
        key = torch.tensor([[1.0, 0.0],
                            [0.0, 1.0]])
        value = torch.tensor([[5.0, 0.0],
                              [0.0, 5.0]])
        out_soft = attention_forward(
            query, key, value,
            self.w_q, self.b_q, self.w_k, self.b_k, self.w_v, self.b_v,
            self.k_0, self.v_0, use_hard_max=False,
        )
        out_hard = attention_forward(
            query, key, value,
            self.w_q, self.b_q, self.w_k, self.b_k, self.w_v, self.b_v,
            self.k_0, self.v_0, use_hard_max=True,
        )
        # Soft-max with *9999 should closely approximate hard-max
        self.assertTrue(torch.allclose(out_soft, out_hard, atol=1e-3))

    def test_null_key_wins_when_all_keys_negative(self):
        """When all real keys score negatively, null key (score 0) wins."""
        query = torch.tensor([[1.0, 0.0]])
        key = torch.tensor([[-1.0, 0.0],
                            [-1.0, 0.0]])
        value = torch.tensor([[5.0, 0.0],
                              [0.0, 5.0]])
        out = attention_forward(
            query, key, value,
            self.w_q, self.b_q, self.w_k, self.b_k, self.w_v, self.b_v,
            self.k_0, self.v_0, use_hard_max=True,
        )
        # Null key scores 0, real keys score -1 → selects null value (zeros)
        expected = torch.tensor([[0.0, 0.0]])
        self.assertTrue(torch.allclose(out, expected))

    def test_multiple_queries(self):
        """Test with L=2 queries."""
        query = torch.tensor([[1.0, 0.0],
                              [0.0, 1.0]])
        key = torch.tensor([[1.0, 0.0],
                            [0.0, 1.0]])
        value = torch.tensor([[3.0, 0.0],
                              [0.0, 7.0]])
        out = attention_forward(
            query, key, value,
            self.w_q, self.b_q, self.w_k, self.b_k, self.w_v, self.b_v,
            self.k_0, self.v_0, use_hard_max=True,
        )
        expected = torch.tensor([[3.0, 0.0],
                                 [0.0, 7.0]])
        self.assertTrue(torch.allclose(out, expected))


if __name__ == '__main__':
    unittest.main()
