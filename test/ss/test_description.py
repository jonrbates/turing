from test.ss.helper import TestCase
from turing.ss.simulator import Description, balanced_parentheses_delta_stack, balanced_parentheses_terminal_states

class TestDescription(TestCase):

    def test_description(self):
        description = Description(balanced_parentheses_delta_stack, balanced_parentheses_terminal_states)