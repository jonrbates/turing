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
    """TODO:
    """
    def __init__(self, s: int, p: int):
        super().__init__()
        self.s = s
        self.p = p

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
            w[4*i+j, i] = 1/(10*p**2)
            b[4*i+j] = (10*p**2-4*p-1)/(10*p**2)
            # push1
            j = 2
            w[4*i+j, i] = 1/(10*p**2)
            b[4*i+j] = (10*p**2-1)/(10*p**2)
            # pop
            j = 3
            w[4*i+j, i] = 10*p**2
            b[4*i+j] = -10*p**2+4*p+1
        linear = Linear(4*p, p)
        linear.weight = Parameter(w)
        linear.bias = Parameter(b)
        self.next_sub_stack = linear

        # linear next subtop
        w = torch.zeros(4*p, p)
        for i in range(p):
            j = 3
            w[4*i+j, i] = -4
        linear = Linear(4*p, p, bias=False)
        linear.weight = Parameter(w)
        self.next_sub_top = linear

        # linear_st; eq (25)
        w = (2*p+1)*10*p**2 * torch.eye(4*p)
        b = -(2*p+1)*(10*p**2-2) * torch.ones(4*p,)
        linear = Linear(4*p, 4*p)
        linear.weight = Parameter(w)
        linear.bias = Parameter(b)
        self.linear_st = linear

        # linear_sn; eq (26)
        w = 10*p**2 * torch.eye(4*p)
        b = -(10*p**2-4*p-2) * torch.ones(4*p,)
        linear = Linear(4*p, 4*p)
        linear.weight = Parameter(w)
        linear.bias = Parameter(b)
        self.linear_sn = linear

    def forward(self, x: Tensor):
        s, p = self.s, self.p

        # current main layer
        st_, subs_, subt_, subn_ = 0, s, s+4*p, s+8*p
        state = x[st_:subs_]
        noisy_sub_stack = x[subs_:subt_]
        noisy_sub_top = x[subt_:subn_]
        noisy_sub_nonempty = x[subn_:]

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
        next_noisy_sub_top = self.linear_st(next_noisy_sub_stack)
        next_noisy_sub_nonempty = self.linear_sn(next_noisy_sub_stack)

        o = torch.cat([
            next_state,
            next_noisy_sub_stack,
            next_noisy_sub_top,
            next_noisy_sub_nonempty
        ])
        return o

    def fit(self, delta, z2i, states, alphabet, fallback_next_state_and_action=('F', 'noop', 'noop')):
        """
        TODO

        Create datapoints to set beta, gamma
        """
        # y = self._generate_y(delta, z2i, states, alphabet, i2b, fallback_next_state_and_action)
        # x = self.configuration_detector.linear.weight.detach()
        # h = self.configuration_detector(x)
        p, s = self.p, self.s

        grid_size = len(delta)*4**p
        low_value = (-8*p**2+2)
        y = torch.zeros(grid_size, 4*p+s)
        x = low_value * torch.ones(grid_size, 8*p+s)
        row = 0
        for _, ((z, top_0, top_1), (z_next, action_1, action_2)) in enumerate(delta.items()):

            substack_action_indicator = torch.zeros(4*p)
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
        """See range table; p. 145
        """
        p = self.p
        if top == 1:
            return 2*p+1, 4*p+1
        elif top == 0:
            return -8*p**2+2, 1
        elif top == None:
            return -20*p**3-10*p**2+4*p+2, -10*p**2+4*p+2

    def noisy_sampler(self, tops):
        """Sample d (the meaningful values?) of Lemma 6.2

        See range table; p. 145
        """
        # a_1, a_2, i2b[b_1], i2b[b_2]
        sampled_top, sampled_nonempty = tuple(zip(*map(self._sample, tops)))
        return torch.Tensor(sampled_top + sampled_nonempty)

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
        """TODO:
        """
        p = self.p
        U, S, Vh = torch.linalg.svd(h.detach(), full_matrices=False)
        C = (Vh.T @ torch.inverse(torch.diag(S)) @ U.T @ y).T
        return C[4*p:,:], C[:4*p,:]


class ConfigurationDetector1(Module):
    """TODO:

        \beta(d_1^(1),...,d_t^(r)) x = \sum_{i \in I} c_i \sigma(v_i^T \mu_i)
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
    """Feedforward layers to

    Notes
        A layer used in

        input x: s + p = len(states) + len(alphabet); we ignore the counter

    Reference
        SS95
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
        # b[:s] = 1
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

        def decode(xx: float) -> str:
            # top
            if xx < 1e-8:
                return ''
            elif 4*xx-2 > 0:
                # top is 1
                return '1' + decode(4*xx-2-1)
            else:
                # top is 0
                return '0' + decode(4*xx-1)

        for w in o[s+2*p:s+3*p].detach():
            print(w, decode(w.item()))
        # F3, F2
        u = self.configuration_detector(o[:s+2*p])
        # print('beta, gamma', [self.beta(u), self.gamma(u)-1])
        proj = torch.cat([self.beta(u), self.gamma(u)-1])
        # self.gamma(self.configuration_detector(torch.Tensor((0,1,0,0,0,1,1,1))))
        # print('update', self.linear_update(o))
        o = proj + self.linear_update(o)
        o = saturated_relu(o)
        # print(o[:s], o[s:])
        # F1
        o = self.linear_F_1(o)
        o = saturated_relu(o)
        print()
        return o

    def fit(self, delta, z2i, states, alphabet, fallback_next_state_and_action=('F', 'noop', 'noop')):
        """
        """
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
        """TODO:
         y = y[batch, :]
           = cd(x[batch, :]) * C
           = h * C
        """
        s = self.s
        C = torch.linalg.solve(h, y).detach().T
        return C[:s,:], C[s:,:]


class ConfigurationDetector4(Module):
    """TODO:

        = -1 + \sum_{j=1}^s \beta_j(a_1,...,a_p,b_1,...,b_p) x_j
        = -1 + \sum_{j=1}^s \sum_{i=1}^{3^p} c_i \sigma(v_{i}{1} a_1 + ... + v_{i}{2p} b_p + x_j - const)
        = C * \sigma (L * input + bias) - 1

    where
        input = (x,a,b) is (s+2p) x 1
        C is 1 x s*3^p
        L is s*3^p x (s+2p) =
            | e_1'  v |
            | e_2'  v |
            |   ...   |
            | e_s'  v |

    Note
        L, bias are universal, hence constant
        C is independent

    Unlike the paper, we use s instead of s+1 states.

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
        for i in range(self.p):
            if coefs[i]==1 and coefs[i+self.p]==0:
                return None
        return coefs