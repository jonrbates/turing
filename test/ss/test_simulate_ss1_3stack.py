import unittest
from test.ss.helper import TestCase
from turing.ss.simulator import Description, Simulator


# Language: a^n b^n a^n (e.g. "", "aba", "aabbaa")
# Stack 1: input (read left to right via pop)
# Stack 2: count leading a's (push during phase A, pop during phase B)
# Stack 3: count b's (push during phase B, pop during phase C)
#
# States: I (initial), A (reading leading a's), B (reading b's),
#         C (reading trailing a's), T (accept), F (reject)

anbnan_delta = {
    # Initial
    ('I', None, None, None): ('T', 'noop', 'noop', 'noop'),
    ('I',  'a', None, None): ('A',  'pop', 'push a', 'noop'),
    ('I',  'b', None, None): ('F', 'noop', 'noop', 'noop'),

    # Phase A: reading leading a's, pushing to stack 2
    ('A', 'a',  '*', None): ('A',  'pop', 'push a', 'noop'),
    ('A', 'b',  'a', None): ('B',  'pop',    'pop', 'push a'),
    ('A', None,  '*', None): ('F', 'noop', 'noop', 'noop'),

    # Phase B: reading b's, popping stack 2, pushing stack 3
    ('B', 'b',  'a',  '*'): ('B',  'pop',    'pop', 'push a'),
    ('B', 'a', None,  'a'): ('C',  'pop', 'noop',      'pop'),
    ('B', 'a',  'a',  '*'): ('F', 'noop', 'noop', 'noop'),
    ('B', None,  '*',  '*'): ('F', 'noop', 'noop', 'noop'),
    ('B', 'b', None,  '*'): ('F', 'noop', 'noop', 'noop'),

    # Phase C: reading trailing a's, popping stack 3
    ('C', 'a', None,  'a'): ('C',  'pop', 'noop', 'pop'),
    ('C', None, None, None): ('T', 'noop', 'noop', 'noop'),
    ('C', None, None,  'a'): ('F', 'noop', 'noop', 'noop'),
    ('C', 'a', None, None): ('F', 'noop', 'noop', 'noop'),
    ('C', 'b',  '*',  '*'): ('F', 'noop', 'noop', 'noop'),

    # Terminal states are absorbing
    ('T', '*', '*', '*'): ('T', 'noop', 'noop', 'noop'),
    ('F', '*', '*', '*'): ('F', 'noop', 'noop', 'noop'),

    # Catch-all for unreachable configurations (stacks 2,3 only hold 'a')
    ('*', '*', '*', 'b'): ('F', 'noop', 'noop', 'noop'),
    ('*', '*', 'b', '*'): ('F', 'noop', 'noop', 'noop'),
}

anbnan_terminal_states = ['T', 'F']


class TestSimulateSS1_3Stack(TestCase):

    def setUp(self):
        description = Description(anbnan_delta, anbnan_terminal_states)
        self.sim = Simulator(description, version=1)

    def test_empty_string(self):
        result = self.sim.simulate('', T=4)
        self.assertEqual(result, 'T')

    def test_aba(self):
        result = self.sim.simulate('aba', T=10)
        self.assertEqual(result, 'T')

    def test_single_a(self):
        result = self.sim.simulate('a', T=6)
        self.assertEqual(result, 'F')

    def test_ab(self):
        result = self.sim.simulate('ab', T=8)
        self.assertEqual(result, 'F')

    def test_single_b(self):
        result = self.sim.simulate('b', T=6)
        self.assertEqual(result, 'F')

    def test_ba(self):
        result = self.sim.simulate('ba', T=8)
        self.assertEqual(result, 'F')

    def test_abba(self):
        """a^1 b^2 a^1 — too many b's."""
        result = self.sim.simulate('abba', T=12)
        self.assertEqual(result, 'F')

    def test_aaba(self):
        """a^2 b^1 a^1 — too many leading a's."""
        result = self.sim.simulate('aaba', T=12)
        self.assertEqual(result, 'F')

    def test_aa(self):
        """aa — no b's, rejected in phase A."""
        result = self.sim.simulate('aa', T=8)
        self.assertEqual(result, 'F')


if __name__ == '__main__':
    unittest.main()
