import torch
from itertools import product
from torch import Tensor
from torch.nn import (
    Module,
    Linear,
    Parameter
)
from turing.functional import saturated_relu
from typing import Tuple


class SiegelmannSontag1(Module):
    """1-layer RNN simulating a p-stack machine (Siegelmann & Sontag).

    Stacks are encoded as rational numbers via Cantor-set encoding with
    base B = 10p^2.  The pop formula amplifies error by B per step,
    limiting float64 to ~9 pops from one stack.
    """
    def __init__(self, s: int, p: int):
        super().__init__()
        self.s = s
        self.p = p
        B = 10 * p**2
        self.base = B

        # configuration detector
        self.configuration_detector = ConfigurationDetector1(s, p)

        # aggregate sub to full
        w = torch.zeros(p, 4*p)
        for k in range(p):
            w[k, 4*k:4*k+4] = 1
        linear = Linear(4*p, p, bias=False)
        linear.weight = Parameter(w)
        self.linear_stack = linear
        self.linear_top = linear

        # linear combine configuration detectors
        self.beta = Linear(s * 9**p, s, bias=False)
        self.gamma = Linear(s * 9**p, 4*p, bias=False)

        # linear next substack
        w = torch.zeros(4*p, p)
        b = torch.zeros(4*p,)
        for i in range(p):
            # noop
            j = 0
            w[4*i+j, i] = 1
            b[4*i+j] = 0
            # push0
            j = 1
            w[4*i+j, i] = 1/B
            b[4*i+j] = (B-4*p-1)/B
            # push1
            j = 2
            w[4*i+j, i] = 1/B
            b[4*i+j] = (B-1)/B
            # pop
            j = 3
            w[4*i+j, i] = B
            b[4*i+j] = -B+4*p+1
        linear = Linear(4*p, p)
        linear.weight = Parameter(w)
        linear.bias = Parameter(b)
        self.next_sub_stack = linear

        # linear next subtop
        w = torch.zeros(4*p, p)
        for i in range(p):
            j = 3
            w[4*i+j, i] = -4*p
        linear = Linear(4*p, p, bias=False)
        linear.weight = Parameter(w)
        self.next_sub_top = linear

        # linear_sub_top; eq (25) - top detection
        w = (2*p+1)*B * torch.eye(4*p)
        b = -(2*p+1)*(B-2) * torch.ones(4*p,)
        linear = Linear(4*p, 4*p)
        linear.weight = Parameter(w)
        linear.bias = Parameter(b)
        self.linear_sub_top = linear

        # linear_sub_nonempty; eq (26) - nonempty detection
        w = B * torch.eye(4*p)
        b = -(B-4*p-2) * torch.ones(4*p,)
        linear = Linear(4*p, 4*p)
        linear.weight = Parameter(w)
        linear.bias = Parameter(b)
        self.linear_sub_nonempty = linear

    def forward(self, x: Tensor):
        s, p = self.s, self.p

        # current main layer
        state = x[:s]
        noisy_sub_stack = x[s:s+4*p]
        noisy_sub_top = x[s+4*p:s+8*p]
        noisy_sub_nonempty = x[s+8*p:]

        # hidden layer
        w = torch.cat([
            noisy_sub_top,
            noisy_sub_nonempty,
            state
        ])
        cd_out = self.configuration_detector(w)
        sub_stack = saturated_relu(noisy_sub_stack)
        sub_top = saturated_relu(noisy_sub_top)
        stack = self.linear_stack(sub_stack)
        top = self.linear_top(sub_top)

        # next main layer
        next_state = self.beta(cd_out)
        # eq (24)
        next_noisy_sub_stack = self.next_sub_stack(stack)\
            + self.next_sub_top(top)\
            + self.gamma(cd_out) - 1
        next_noisy_sub_top = self.linear_sub_top(next_noisy_sub_stack)
        next_noisy_sub_nonempty = self.linear_sub_nonempty(next_noisy_sub_stack)

        o = torch.cat([
            next_state,
            next_noisy_sub_stack,
            next_noisy_sub_top,
            next_noisy_sub_nonempty
        ])
        return o

    def fit(self, delta, z2i, states, alphabet, fallback_next_state_and_action=('F', 'noop', 'noop')):
        """Set beta and gamma weights by solving CD(x) * C = y on training data.

        Generates (x, y) pairs from the transition function delta, where x
        encodes (state, noisy_top, noisy_nonempty) and y encodes (next_state,
        substack_action).  Solves for C via pseudoinverse, then splits C into
        beta (state rows) and gamma (action rows).
        """
        p, s = self.p, self.s
        B = self.base
        dtype = self.configuration_detector.linear.weight.dtype

        grid_size = len(delta)*4**p
        # Non-active tracks during simulation produce these values from
        # linear_sub_top(0) and linear_sub_nonempty(0).  Training data must match.
        low_top = -(2*p+1)*(B-2)
        low_ne = -(B-4*p-2)
        y = torch.zeros(grid_size, 4*p+s, dtype=dtype)
        x = torch.zeros(grid_size, 8*p+s, dtype=dtype)
        x[:, :4*p] = low_top    # noisy_top positions
        x[:, 4*p:8*p] = low_ne  # noisy_nonempty positions
        row = 0
        for _, ((z, top_0, top_1), (z_next, action_1, action_2)) in enumerate(delta.items()):

            substack_action_indicator = torch.zeros(4*p, dtype=dtype)
            for k, action in enumerate([action_1, action_2]):
                offset = self.map_action(action)
                substack_action_indicator[4*k+offset] = 1
            d = self.noisy_sampler((top_0, top_1))
            # generate various inputs corresponding to the current key
            for i in self.configuration_detector.generate_i():
                # we skip None cases, so this is just combinations of substack 4^p
                if None in i: continue
                # set x
                x[row, 8*p+z2i[z]] = 1
                slice = self.configuration_detector.convert_to_tensor_indices(i)
                x[row, slice] = d
                # set y
                y[row, 4*p+z2i[z_next]] = 1
                y[row, :4*p] = substack_action_indicator
                row += 1

        h = self.configuration_detector(x)
        beta, gamma = self._solve_for_beta_and_gamma(h, y)
        self.beta.weight = Parameter(beta)
        self.gamma.weight = Parameter(gamma)

    def _sample(self, top):
        """Boundary noisy values for (top, nonempty) detection.

        Returns (noisy_top, noisy_nonempty) at the boundary of the valid
        range for the given top symbol.  These are the training targets for
        the configuration detector.

        Ranges (independent of base B):
          top=0  → noisy_top ∈ [-35, -30), noisy_nonempty ∈ [1, ~10)
          top=1  → noisy_top ∈ [5, 10),    noisy_nonempty ∈ [1, ~10)
          None   → noisy_top = -(2p+1)(B-2), noisy_nonempty = -(B-4p-2)
        """
        p = self.p
        B = self.base
        if top == 1:
            return 2*p+1, 4*p+1
        elif top == 0:
            return -(2*p+1)*(4*p-2), 1
        elif top == None:
            return -(2*p+1)*(B-2), -(B-4*p-2)

    def noisy_sampler(self, tops):
        """Return noisy (sub_top, sub_nonempty) values for a pair of top symbols.

        Maps each top symbol to boundary values via _sample(), then
        concatenates into a single tensor for the configuration detector.
        See Lemma 6.2, range table p. 145.
        """
        sampled_top, sampled_nonempty = tuple(zip(*map(self._sample, tops)))
        dtype = self.configuration_detector.linear.weight.dtype
        return torch.tensor(sampled_top + sampled_nonempty, dtype=dtype)

    @staticmethod
    def map_action(action: str):
        if action == 'push 0':
            offset = 1
        elif action == 'push 1':
            offset = 2
        elif action == 'pop':
            offset = 3
        else:
            # noop
            offset = 0
        return offset

    def _solve_for_beta_and_gamma(self, h, y):
        """Solve h * C^T = y via SVD pseudoinverse, return (beta, gamma)."""
        p = self.p
        U, S, Vh = torch.linalg.svd(h.detach(), full_matrices=False)
        C = (Vh.T @ torch.inverse(torch.diag(S)) @ U.T @ y).T
        return C[4*p:,:], C[:4*p,:]


class ConfigurationDetector1(Module):
    """Configuration detector for SS1: maps (state, noisy_top, noisy_nonempty) to
    indicator features via sigma(L * x + bias).

    Each neuron fires when the input matches a particular (state, substack)
    multi-index from {None,0,1,2,3}^{2p} x {1..s}.  The output is a
    s x 9^p vector consumed by beta/gamma to produce the next state and action.
    """
    def __init__(self, s: int, p: int):
        super().__init__()
        # number of states
        self.s = s
        # number of stacks
        self.p = p

        w = torch.zeros(s * 9**p, 8*p+s)
        b = torch.zeros(s * 9**p, )
        counter = 0
        for i, z in self.generate_i_z():
            slice = self.convert_to_tensor_indices(i)
            k = len(slice)
            if k == 0:
                # state weight
                w[counter, 8*p+z] = 1
            else:
                # top, nonempty weights
                w[counter, slice] = 1
                # state weight
                w[counter, 8*p+z] = k*(4*p+2)
                # bias
                b[counter] = -k*(4*p+2)
            counter += 1

        linear = Linear(8*p+s, s * 9**p)
        linear.weight = Parameter(w)
        linear.bias = Parameter(b)
        self.linear = linear

    def forward(self, x):
        return saturated_relu(self.linear(x))

    def generate_i_z(self):
        for i in self.generate_i():
            for z in range(self.s):
                yield i, z

    def generate_i(self):
        """Generate multi-index from {None, 0, 1, 2, 3}^{2p},
        skip invalid multi-indices
        """
        # substack indices
        subs = (None, 0, 1, 2, 3)
        # choose i_nonempty first
        for i_nonempty in product(*self.p*[subs]):
            # then get i_top
            for i_top in self._generate_i_top(i_nonempty):
                yield (*i_top, *i_nonempty)

    def _generate_i_top(self, i_nonempty):
        cands = [{None, j} for j in i_nonempty]
        for j in product(*cands):
            yield j

    def convert_to_tensor_indices(self, i: Tuple[int]):
        """Map (i1, i2, i3, ...) to (i1, i2+4, i3+8, ...)
        """
        offsets = range(0, 8*self.p, 4)
        return tuple(a+b for a, b in zip(i, offsets) if a is not None)


class SiegelmannSontag4(Module):
    """4-layer RNN simulating a 2-stack machine (Siegelmann & Sontag, §4).

    Input x has dimension s-1+p, encoding (state, stack_1, stack_2).
    Stacks are Cantor-encoded with base=4, p=1/2.
    """
    def __init__(self, s: int, p: int = 2):
        super().__init__()

        self.s = s
        self.p = p

        # F_4, linear state0
        linear = Linear(s-1, s)
        w = torch.zeros(s, s-1)
        b = torch.zeros(s, )
        # 0-state
        w[0, :] = -1
        b[0] = 1
        w[1:, :] = torch.eye(s-1)
        linear.weight = Parameter(w)
        linear.bias = Parameter(b)
        self.linear_state0 = linear

        # F_4, linear top, zeta, a
        linear = Linear(p, p)
        w = 4 * torch.eye(p)
        b = -2 * torch.ones(p)
        linear.weight = Parameter(w)
        linear.bias = Parameter(b)
        self.linear_top = linear

        # F_4, linear nonempty, tau, b
        linear = Linear(p, p)
        w = 4 * torch.eye(p)
        b = torch.zeros(p)
        linear.weight = Parameter(w)
        linear.bias = Parameter(b)
        self.linear_nonempty = linear

        # F_3, F_2
        self.configuration_detector = ConfigurationDetector4(s, p)

        d = s*3**p
        self.beta = Linear(d, s, bias=False)
        self.gamma = Linear(d, 4*p, bias=False)

        eta = s+4*p
        linear = Linear(s+3*p, eta)
        w = torch.zeros(eta, s+3*p)
        b = torch.zeros(eta,)
        # substacks
        stack_, top_ = s+2*p, s
        k = 0
        for i in range(p):
            w[s+k+0, stack_+i] = 1
            b[s+k+0] = 0
            w[s+k+1, stack_+i] = 1/4
            b[s+k+1] = 1/4
            w[s+k+2, stack_+i] = 1/4
            b[s+k+2] = 3/4
            w[s+k+3, stack_+i] = 4
            w[s+k+3, top_+i] = -2
            b[s+k+3] = -1
            k += 4
        linear.weight = Parameter(w)
        linear.bias = Parameter(b)
        self.linear_update = linear

        # F_1
        linear = Linear(eta, s-1+p)
        w = torch.zeros(s-1+p, eta)
        b = torch.zeros(s-1+p,)
        w[:s-1, 1:s] = torch.eye(s-1)
        for i in range(p):
            w[s-1+i, s+4*i:s+4*i+4] = 1

        linear.weight = Parameter(w)
        linear.bias = Parameter(b)
        self.linear_F_1 = linear


    def forward(self, x: Tensor):
        s, p = self.s, self.p
        x_state, x_stack = x[:s-1], x[s-1:]
        # F4
        o = torch.cat([
            self.linear_state0(x_state),
            self.linear_top(x_stack),
            self.linear_nonempty(x_stack),
            x_stack])
        o = saturated_relu(o)

        # F3, F2
        u = self.configuration_detector(o[:s+2*p])
        proj = torch.cat([self.beta(u), self.gamma(u)-1])
        o = proj + self.linear_update(o)
        o = saturated_relu(o)
        # F1
        o = self.linear_F_1(o)
        o = saturated_relu(o)
        return o

    def fit(self, delta, z2i, states, alphabet, fallback_next_state_and_action=('F', 'noop', 'noop')):
        """Set beta and gamma weights by solving CD(x) * C = y on training data."""
        y = self._generate_y(delta, z2i, states, alphabet, fallback_next_state_and_action)
        x = self.configuration_detector.linear.weight.detach()
        h = self.configuration_detector(x)
        beta, gamma  = self._solve_for_beta_and_gamma(h, y)
        self.beta.weight = Parameter(beta)
        self.gamma.weight = Parameter(gamma)

    def _generate_y(self, delta, z2i, states, alphabet, fallback_next_state_and_action):
        p, s = self.p, self.s
        d = s * 3**p
        y = torch.zeros(d, s+4*p)
        for row, (v, z) in enumerate(self.configuration_detector.generate_v_z()):
            key = self._get_key(v, z, states)
            if key not in delta:
                print(f'The transition from {key} is not specified. Using fallback.')

            z_next, action_1, action_2 = delta.get(key, fallback_next_state_and_action)
            y[row, z2i[z_next]] = 1
            for k, action in enumerate([action_1, action_2]):
                offset = self.map_action(action)
                y[row, s+4*k+offset] = 1

        return y

    @staticmethod
    def map_action(action: str):
        if action == 'push 0':
            offset = 1
        elif action == 'push 1':
            offset = 2
        elif action == 'pop':
            offset = 3
        else:
            # noop
            offset = 0
        return offset

    def _get_key(self, v, z, states):
        a_1, a_2, b_1, b_2 = v
        return states[z], a_1 if b_1==1 else None, a_2 if b_2==1 else None

    def _solve_for_beta_and_gamma(self, h, y):
        """Solve h * C^T = y directly, return (beta, gamma).

        beta selects the next state (first s rows of C), gamma selects the
        stack actions (remaining rows).
        """
        s = self.s
        C = torch.linalg.solve(h, y).detach().T
        return C[:s,:], C[s:,:]


class ConfigurationDetector4(Module):
    """Configuration detector for SS4: maps (state, a, b) to indicator features.

    Computes C * sigma(L * input + bias) - 1, where:
        input = (x, a, b) is (s+2p) x 1
        L is s*3^p x (s+2p), with rows [e_j | v_i] pairing each state
            unit vector e_j with each symbol coefficient vector v_i
        bias = -sum(v_i) per row (so neuron fires only when all inputs are 1)
        C is 1 x s*3^p, set by _solve_for_beta_and_gamma

    L and bias are universal (independent of the transition function);
    C encodes the specific machine.  Unlike the paper, we use s states
    instead of s+1.
    """

    def __init__(self, s: int, p: int):
        super().__init__()
        # number of states
        self.s = s
        # number of stacks
        self.p = p

        # network
        d = s*3**p
        w = torch.zeros(d, s+2*p)
        b = torch.zeros(d,)
        counter = 0
        for v, z in self.generate_v_z():
            w[counter, s:] = torch.Tensor(v)
            # copy state
            w[counter, z] = 1
            # bias
            b[counter] = -sum(v)
            counter += 1

        linear = Linear(s+2*p, d)
        linear.weight = Parameter(w)
        linear.bias = Parameter(b)
        self.linear = linear

    def forward(self, x):
        return saturated_relu(self.linear(x))

    def generate_v_z(self):
        for v in self._generate_v():
            for z in range(self.s):
                yield v, z

    def _generate_v(self):
        for i in range(4**self.p):
            coefs = self._get_row_of_v(i)
            if coefs: yield coefs

    def _get_row_of_v(self, i: int):
        t = 2*self.p
        coefs = tuple(map(int, f"{i:0{t}b}"))
        # skip invalid case where top/peek is 1 and nonempty is 0
        for k in range(self.p):
            if coefs[k]==1 and coefs[k+self.p]==0:
                return None
        return coefs