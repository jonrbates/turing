import random
from test.ss.helper import TestCase
from turing.balanced_parentheses import generate_balanced_parentheses
from turing.ss.simulator import (
    Description,
    Simulator,
    balanced_parentheses_delta_stack,
    balanced_parentheses_terminal_states,
)


def is_balanced(s: str) -> bool:
    depth = 0
    for c in s:
        if c == '(':
            depth += 1
        else:
            depth -= 1
        if depth < 0:
            return False
    return depth == 0


def generate_unbalanced(length: int, rng: random.Random) -> str:
    """Generate a random parentheses string that is NOT balanced."""
    while True:
        s = ''.join(rng.choice('()') for _ in range(length))
        if not is_balanced(s):
            return s


class TestSimulateSS1(TestCase):

    def setUp(self):
        description = Description(
            balanced_parentheses_delta_stack,
            balanced_parentheses_terminal_states,
        )
        self.sim = Simulator(description, version=1)

    def test_balanced_increasing_lengths(self):
        """Test balanced parentheses of increasing length.

        The pop formula amplifies error by base (40) per step.  With float64,
        strings up to length 8 (4 pairs) work reliably.
        """
        for n in range(1, 5):
            strings = generate_balanced_parentheses(n)
            for s in strings:
                with self.subTest(string=s):
                    result = self.sim.simulate(s, T=2 * len(s) + 4)
                    self.assertEqual(result, 'T', f'{s!r} should be balanced')

    def test_unbalanced_increasing_lengths(self):
        """Test unbalanced parentheses of increasing length.

        Unbalanced strings often terminate early (before error accumulates),
        so they work for longer inputs than balanced strings.
        """
        rng = random.Random(42)
        for length in range(1, 10):
            for _ in range(min(5, 2**length)):
                s = generate_unbalanced(length, rng)
                with self.subTest(string=s):
                    result = self.sim.simulate(s, T=2 * len(s) + 4)
                    self.assertEqual(result, 'F', f'{s!r} should be unbalanced')

    def test_empty_string(self):
        result = self.sim.simulate('', T=4)
        self.assertEqual(result, 'T')
