import unittest
from turing.ss.simulator import Description


class TestExpandWildcards(unittest.TestCase):
    """Test Description.expand_wildcards() behavior."""

    def _make_description(self, delta, terminal_states=None):
        if terminal_states is None:
            terminal_states = ['T', 'F']
        return Description(delta, terminal_states)

    def test_no_wildcards(self):
        delta = {
            ('I', 'a', None): ('T', 'pop', 'noop'),
            ('I', None, None): ('T', 'noop', 'noop'),
            ('T', 'a', None): ('T', 'noop', 'noop'),
            ('T', None, None): ('T', 'noop', 'noop'),
            ('T', 'a', 'a'): ('T', 'noop', 'noop'),
            ('T', None, 'a'): ('T', 'noop', 'noop'),
            ('T', 'b', None): ('T', 'noop', 'noop'),
            ('T', 'b', 'a'): ('T', 'noop', 'noop'),
            ('T', 'b', 'b'): ('T', 'noop', 'noop'),
            ('T', 'a', 'b'): ('T', 'noop', 'noop'),
            ('T', None, 'b'): ('T', 'noop', 'noop'),
            ('T', 'b', None): ('T', 'noop', 'noop'),
            ('F', 'a', None): ('F', 'noop', 'noop'),
            ('F', None, None): ('F', 'noop', 'noop'),
            ('F', 'a', 'a'): ('F', 'noop', 'noop'),
            ('F', None, 'a'): ('F', 'noop', 'noop'),
            ('F', 'b', None): ('F', 'noop', 'noop'),
            ('F', 'b', 'a'): ('F', 'noop', 'noop'),
            ('F', 'b', 'b'): ('F', 'noop', 'noop'),
            ('F', 'a', 'b'): ('F', 'noop', 'noop'),
            ('F', None, 'b'): ('F', 'noop', 'noop'),
            ('I', 'b', None): ('F', 'noop', 'noop'),
            ('I', None, 'a'): ('F', 'noop', 'noop'),
            ('I', 'a', 'a'): ('F', 'noop', 'noop'),
            ('I', 'b', 'a'): ('F', 'noop', 'noop'),
            ('I', None, 'b'): ('F', 'noop', 'noop'),
            ('I', 'a', 'b'): ('F', 'noop', 'noop'),
            ('I', 'b', 'b'): ('F', 'noop', 'noop'),
        }
        desc = self._make_description(delta)
        # With no wildcards, expanded delta should have the same entries
        for k, v in desc.delta.items():
            self.assertNotEqual(k[0], '*')
            self.assertNotEqual(k[1], '*')
            self.assertNotEqual(k[2], '*')

    def test_state_wildcard_expands(self):
        """Wildcard '*' in state position expands to all non-terminal states."""
        delta = {
            ('I', None, None): ('T', 'noop', 'noop'),
            ('*', 'a', None): ('T', 'pop', 'noop'),
            ('*', 'b', None): ('F', 'noop', 'noop'),
            ('T', '*', '*'): ('T', 'noop', 'noop'),
            ('F', '*', '*'): ('F', 'noop', 'noop'),
        }
        desc = self._make_description(delta)
        # ('*', 'a', None) should expand to ('I', 0, None)
        # (a→0 since 'a' is first in sorted alphabet)
        self.assertIn(('I', 0, None), desc.delta)

    def test_specific_overrides_wildcard(self):
        """Specific entries must take priority over wildcard expansions."""
        delta = {
            ('I', None, None): ('T', 'noop', 'noop'),
            ('*', 'a', None): ('F', 'pop', 'noop'),  # wildcard: go to F
            ('I', 'a', None): ('T', 'pop', 'noop'),  # specific: go to T
            ('*', 'b', None): ('F', 'noop', 'noop'),
            ('T', '*', '*'): ('T', 'noop', 'noop'),
            ('F', '*', '*'): ('F', 'noop', 'noop'),
        }
        desc = self._make_description(delta)
        # The specific ('I', 'a', None) → T should win over ('*', 'a', None) → F
        key = ('I', 0, None)
        self.assertEqual(desc.delta[key][0], 'T')

    def test_top_wildcard_expands(self):
        """Wildcard in top-of-stack position expands to {None, 0, 1}."""
        delta = {
            ('I', '*', None): ('T', 'noop', 'noop'),
            ('I', '*', 'a'): ('F', 'noop', 'noop'),
            ('I', '*', 'b'): ('F', 'noop', 'noop'),
            ('T', '*', '*'): ('T', 'noop', 'noop'),
            ('F', '*', '*'): ('F', 'noop', 'noop'),
        }
        desc = self._make_description(delta)
        for top in (None, 0, 1):
            self.assertIn(('I', top, None), desc.delta)


if __name__ == '__main__':
    unittest.main()
