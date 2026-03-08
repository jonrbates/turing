import torch
from torch import Tensor
from torch.nn import (
    Module,
)
from turing.ss.networks import SiegelmannSontag1, SiegelmannSontag4
from turing.types import (
    State
)
from typing import List


# Default turing machine description
balanced_parentheses_delta_stack = {
    # Note: noop = no operation, i.e. no action
    # is balanced
    ('*',  '(',  '(') : ('A',  'pop', 'push ('),
    ('*',  '(', None) : ('A',  'pop', 'push ('),
    ('*',  ')',  '(') : ('A',  'pop',    'pop'),
    ('I', None, None) : ('T', 'noop',   'noop'),
    ('A', None, None) : ('T', 'noop',   'noop'),
    # not balanced
    ('*',  ')', None) : ('F', 'noop',   'noop'),
    ('*', None,  '(') : ('F', 'noop',   'noop'),
    # terminal state is final
    ('F',  '*',  '*') : ('F', 'noop',   'noop'),
    ('T',  '*',  '*') : ('T', 'noop',   'noop'),
    # doesn't occur
    ('*',  '*',  ')') : ('F', 'noop',   'noop'),
}

balanced_parentheses_terminal_states = ['T', 'F']


class Description():
    """Turing machine description for stack machine

    delta: transition function
    terminal_states: possible end states

    """
    def __init__(self, delta, terminal_states: List[State]):
        """
        (state, a_1, a_2) -> (state+, action stack 1, action stack 2)
        a_i = top stack i
        """
        self.terminal_states = terminal_states
        self.initial_state = 'I'

        self.set_p(next(iter(delta)))

        # get states and original alphabet
        states = set(self.terminal_states)
        original_alphabet = set()
        for k, v in delta.items():
            (z, a), z_next = self.split_key(k), v[0]
            states.add(z)
            states.add(z_next)
            original_alphabet.update(a)
        states.discard('*')
        original_alphabet.discard('*')
        original_alphabet.discard(None)

        if len(original_alphabet) > 2:
            raise NotImplementedError("Network is implemented for 2 symbols")

        # The stack machine is assumed to have 0, 1 symbols
        self.alphabet = (0, 1)

        # set states, with initial state I first
        states.discard(self.initial_state)
        self.states = (self.initial_state, *sorted(states))
        self.original_alphabet = tuple(sorted(original_alphabet))

        a2i = {
            self.original_alphabet[0]: 0,
            self.original_alphabet[1]: 1,
            None: None,
            '*': '*'
        }

        self.z2i = {z: i for i, z in enumerate(self.states)}

        # normalize delta
        normalized_delta = dict()
        for k, v in delta.items():
            state, a = self.split_key(k)
            next_state, actions = v[0], v[1:]
            normalized_a = tuple(map(lambda x: a2i[x], a))
            normalized_actions = tuple(map(self.action_normalizer, actions))
            normalized_k = (state, *normalized_a)
            normalized_v = (next_state, *normalized_actions)
            normalized_delta[normalized_k] = normalized_v

        # expand wild cards
        self.delta = self.expand_wildcards(normalized_delta)

    def set_p(self, key):
        """Infer p from delta key, then set it
        """
        self.p = len(key) - 1

    def split_key(self, key):
        """Split delta key into (state, a, b)
        """
        return key[0], key[1:]

    def action_normalizer(self, action):
        """E.g map 'push (' to 'push 0'
        """
        symbol0 = self.original_alphabet[0]
        symbol1 = self.original_alphabet[1]
        if action == f'push {symbol0}': return 'push 0'
        if action == f'push {symbol1}': return 'push 1'
        return action

    def get_domains(self):
        """Possible values of wildcard for each key
        """
        state_domain = set(self.states) - set(self.terminal_states)
        return [
            state_domain,
            (None, 0, 1),
            (None, 0, 1)
        ]

    def expand_wildcards(self, delta):
        """Fill in wild cards in dict keys with all elements in domain
        """
        # TODO: does not filter out invalid keys like top=1 with nonempty=0
        x = delta
        domains = self.get_domains()
        for i, domain in enumerate(domains):
            out = {}
            for k, v in x.items():
                if k[i] == '*':
                    k = list(k)
                    for elem in domain:
                        k[i] = elem
                        out[tuple(k)] = v
                else:
                    out[k] = v
            x = out
        return x


class Simulator(Module):
    """Neural network simulator for a 2-stack machine.

    Supports version=4 (4-layer RNN) and version=1
    (1-layer RNN).
    """

    def __init__(self, description: Description, version = 1):
        super().__init__()

        if description.alphabet != (0, 1):
            raise NotImplementedError("Implemented for 0, 1 alphabet")

        self.version = version
        delta = description.delta

        # turing machine settings
        self._terminal_states = description.terminal_states
        self._initial_state = description.initial_state
        self._states = description.states
        self._z2i = {z: i for i, z in enumerate(self.states)}
        self._s = len(self._states)

        self._alphabet = description.alphabet
        self._original_alphabet = description.original_alphabet
        self._a2i = {a: i for i, a in enumerate(self.original_alphabet)}

        if version == 4:
            # neural network settings
            self._p = 2
            self._cantor_base = 4
            self._cantor_p = 1/2   # encoding coefficient; base=4 maps {0,1} to {1/4, 3/4}
            # network module definitions
            self.model = SiegelmannSontag4(self.s, self.p)
            self.model.fit(delta, self.z2i, self.states, self.alphabet)
        elif version == 1:
            # neural network settings
            self._p = 2
            # Base = 10p^2 (paper's formula). The pop formula amplifies error
            # by base times per step, limiting float64 to ~9 pops from one stack.
            self._cantor_base = 10 * self._p ** 2
            self._cantor_p = self._p
            self.model = SiegelmannSontag1(self.s, self.p)
            # float64 required: pop formula amplifies error by base per step
            self.model.double()
            self.model.fit(delta, self.z2i, self.states, self.alphabet)

    @property
    def terminal_states(self):
        return self._terminal_states

    @property
    def states(self):
        return self._states

    @property
    def z2i(self):
        return self._z2i

    @property
    def s(self):
        return self._s

    @property
    def p(self):
        return self._p

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def original_alphabet(self):
        return self._original_alphabet

    @property
    def a2i(self):
        return self._a2i

    def forward(self, x):
        return self.model(x)

    def encoding_function(self, a: str):
        return encoding_function(a, self._cantor_base, self._cantor_p)

    def _decode_state(self, x_state: Tensor) -> State:
        """Decode a state vector to a state name."""
        if self.version == 4:
            # x_state has s-1 dims; entry j represents states[j+1]; all-zero = states[0]
            if x_state.max().item() < 0.5:
                return self.states[0]
            return self.states[x_state.argmax().item() + 1]
        else:
            return self.states[x_state.argmax().item()]

    def _decode_stack(self, val: float) -> str:
        """Decode a Cantor-encoded rational number to a string of original symbols.
        Valid for version=4 (base=4, p=1/2).
        """
        if val < 1e-8:
            return ""
        elif 4 * val - 2 > 0:
            sym = self.original_alphabet[1]
            return str(sym) + self._decode_stack(4 * val - 3)
        else:
            sym = self.original_alphabet[0]
            return str(sym) + self._decode_stack(4 * val - 1)

    def simulate(self, string: str, T=10):
        """Simulate the stack machine for the given input string.

        Prints the machine state at each step and returns the terminal state.
        """
        binary_string = [self.a2i[c] for c in string]
        initial_stack = self.encoding_function(binary_string)

        if self.version == 4:
            x = torch.zeros(self.s - 1 + self.p)
            x[self.s - 1] = initial_stack
            o = x
            for step in range(T):
                x_state = o[:self.s - 1].detach()
                x_stack = o[self.s - 1:].detach()
                z = self._decode_state(x_state)
                stacks = [self._decode_stack(v.item()) for v in x_stack]
                print(f"{z:4}  {stacks}")
                if z in self.terminal_states:
                    return z
                o = self(o)

        elif self.version == 1:
            top = int(binary_string[0]) if len(binary_string) > 0 else None
            s, p = self.s, self.p
            B = self._cantor_base
            x = torch.zeros(self.s + 3 * 4 * self.p, dtype=torch.float64)
            noisy_top, noisy_nonempty = self.model._sample(top)
            # state
            x[0] = 1
            # non-active tracks: match linear_sub_top(0) and linear_sub_nonempty(0)
            low_top = -(2*p+1)*(B-2)
            low_ne = -(B-4*p-2)
            x[s + 4*p:s + 8*p] = low_top
            x[s + 8*p:] = low_ne
            # noisy_sub_stack: noop track of stack 0
            x[s] = initial_stack
            # noisy_sub_top: noop track of stack 0
            x[s + 4*p] = noisy_top
            # noisy_sub_nonempty: noop track of stack 0
            x[s + 8*p] = noisy_nonempty
            o = x
            for step in range(T):
                x_state = o[:self.s].detach()
                x_stack = o[self.s:].detach()
                z = self._decode_state(x_state)
                print(f"{z:4}")
                if z in self.terminal_states:
                    return z
                o = self(o)


def encoding_function(a: str, base: int, p: int):
    r"""Encoding function $\delta$ mapping a binary string to a rational number.

    e.g.
        \delta["01"] = 1/4 + 3/16
    """
    if a == '': return 0
    k = len(a)
    a = list(map(int, a))
    a = torch.tensor(a, dtype=torch.float64)
    denom = base**torch.arange(1, k+1, dtype=torch.float64)
    num = base-1+4*p*(a-1)
    o = (num / denom).sum().item()
    return o