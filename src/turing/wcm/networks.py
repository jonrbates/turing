import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    Module,
    Linear,
    Parameter
)
from turing.functional import attention_forward
from turing.types import (
    State,
    Symbol,
    Delta
)
from typing import Dict, List, Tuple


class Transition(Module):
    """Feedforward layers to transition from current state z and symbol a to (z_next, u, q),
    where z_next is the next state, u is the symbol to be written, q {left, right} is the movement
    direction.

    Notes
        A layer used in theorem 4.1, step 1 of WCM.
    
    Reference
        WCM
    """
    def __init__(self, slices, delta: Delta, states: List[State], alphabet: List[Symbol], z2i: Dict[State, int], a2i: Dict[Symbol, int]):
        super().__init__()    
        d_in, w_pos, st_, sym1_, sym2_, pos1_, pos2_, pos3_, scr1_, scr2_, scr3_, scr4_, scr5_ = slices    
        ds = len(states) * len(alphabet)  # domain size
        # initialize layer 1 weights        
        upper = torch.zeros((ds, d_in)) # upper Z*A block of weights1
        lower = torch.eye(d_in)
        # initialize layer 2 weights
        weight2 = torch.zeros((d_in, d_in+ds))
        k = 0
        for i in range(len(states)):
            for j in range(len(alphabet)):
                # weights layer 1
                upper[k, st_ + i] = 1
                upper[k, sym1_ + j] = 1
                # weights layer 2
                z, a = states[i], alphabet[j]
                if (z, a) in delta:
                    # state transition
                    z_next, u, q = delta[(z, a)]
                    weight2[st_ + z2i[z_next], k] = 1
                    # write symbol
                    weight2[sym2_ + a2i[u], k] = 1
                    # move            
                    weight2[scr5_+(q+1)//2, k] = 1
                else:
                    # unknown
                    pass    
                k += 1
        weight = torch.cat([upper, lower])
        linear1 = Linear(d_in, d_in+ds)
        linear1.weight = Parameter(weight)
        bias = torch.zeros(d_in+ds)
        bias[:ds] = -1
        linear1.bias = Parameter(bias)
        weight2[pos1_:pos2_, (ds+pos1_):(ds+pos2_)] = torch.eye(w_pos)
        weight2[pos2_:pos3_, (ds+pos2_):(ds+pos3_)] = torch.eye(w_pos)
        linear2 = Linear(d_in+ds, d_in)
        linear2.weight = Parameter(weight2)
        linear2.bias = Parameter(torch.zeros_like(linear2.bias))
        self.linear1 = linear1
        self.linear2 = linear2

    def forward(self, x):
        o = self.linear1(x)
        o = F.relu(o)
        o = self.linear2(o)
        return o


class HalfAdder(Module):
    """A half adder circuit.

    Notes
        A layer used in theorem 4.1, step 2 of WCM.
    """
    def __init__(self, d_in: int, i: int, j: int):
        super().__init__()        
        """half adder using xor

        the sum(a[i]+a[j]) will be written at index i
        the carry(a[i]+a[j]) will be written at index j

        sum = xor(a[i], a[j])
          = and(or(a[i], a[j]), not(and(a[i], a[j])))

        carry = and(a[i], a[j])
          = not(not(and(a[i], a[j])))

        """        
        # first layer half adder
        weight = torch.eye(d_in)
        bias = torch.zeros(d_in)
        # and written to i
        weight[i, j] = 1
        bias[i] = -1
        # or written to j
        weight[j, i] = -1
        weight[j, j] = -1
        bias[j] = 1
        linear_or_and = Linear(d_in, d_in)
        linear_or_and.weight = Parameter(weight)
        linear_or_and.bias = Parameter(bias)
        # second layer half adder
        weight = torch.eye(d_in)
        bias = torch.zeros(d_in)
        # not of the and
        weight[i, i] = -1
        bias[i] = 1
        # complete the or
        weight[j, j] = -1
        bias[j] = 1
        linear_or_not = Linear(d_in, d_in)
        linear_or_not.weight = Parameter(weight)
        linear_or_not.bias = Parameter(bias)
        # third layer half adder
        weight = torch.eye(d_in)
        bias = torch.zeros(d_in)
        # and to get the sum
        weight[i, i] = 1
        weight[i, j] = 1
        bias[i] = -1
        # not of not of and to get the carry
        weight[j, i] = -1
        weight[j, j] = 0
        bias[j] = 1
        half_adder_final = Linear(d_in, d_in)
        half_adder_final.weight = Parameter(weight)
        half_adder_final.bias = Parameter(bias)
        self.linear_or_and = linear_or_and
        self.linear_or_not = linear_or_not
        self.half_adder_final = half_adder_final
        
    def forward(self, x):   
        o = self.linear_or_and(x)    
        o = F.relu(o)  
        o = self.linear_or_not(o)  
        o = F.relu(o)
        o = self.half_adder_final(o)
        return o


class FullAdder(Module):
    """A full adder circuit.

    Notes
        A layer used in theorem 4.1, step 2 of WCM.

    Reference
        https://www.101computing.net/binary-subtraction-using-logic-gates/
    """
    def __init__(self, d_in: int, i: int, j: int, k: int):
        super().__init__()        
        self.halfadder1 = HalfAdder(d_in, i, j)
        self.halfadder2 = HalfAdder(d_in, i, k)
        # new carry
        # or
        weight = torch.eye(d_in)
        bias = torch.zeros(d_in)
        weight[k, j] = -1
        weight[k, k] = -1
        bias[k] = 1
        linear_or1 = Linear(d_in, d_in)
        linear_or1.weight = Parameter(weight)
        linear_or1.bias = Parameter(bias)
        weight = torch.eye(d_in)
        bias = torch.zeros(d_in)
        weight[k, k] = -1
        bias[k] = 1
        linear_or2 = Linear(d_in, d_in)
        linear_or2.weight = Parameter(weight)
        linear_or2.bias = Parameter(bias)
        self.linear_or1 = linear_or1
        self.linear_or2 = linear_or2

    def forward(self, x):
        o = self.halfadder1(x)
        o = self.halfadder2(o)
        o = self.linear_or1(o)
        o = F.relu(o)
        o = self.linear_or2(o)
        return o


class PreprocessForAdder(Module):
    """A preprocessing layer that initializes the minuend and subtrahend.

    Notes
        A layer used in theorem 4.1, step 2 of WCM.
    """
    def __init__(self, slices: Tuple[int]):    
        super().__init__()      

        d_in, w_pos, st_, sym1_, sym2_, pos1_, pos2_, pos3_, scr1_, scr2_, scr3_, scr4_, scr5_ = slices

        weight = torch.cat([torch.eye(d_in), torch.zeros(w_pos, d_in)])
        bias = torch.zeros(d_in+w_pos)
        # copy Bin(l) to pos3, the minuend
        for k in range(w_pos):
            weight[pos3_+k, pos2_+k] = 1
        # q's value is encoded in o[scr5_:scr5_+1]
        # use it to initialize o[w:w+w_pos], the subtrahend
        # case: add
        weight[d_in:d_in+w_pos, scr5_+1] = torch.zeros([w_pos])
        weight[d_in+w_pos-1, scr5_+1] = 1        
        # case: subtract
        weight[d_in:d_in+w_pos-1, scr5_] = torch.ones([w_pos-1])
        linear = Linear(d_in, d_in+w_pos)       
        linear.weight = Parameter(weight)
        linear.bias = Parameter(bias)
        self.linear = linear

    def forward(self, tgt: Tensor):
        x = tgt
        return self.linear(x)


class ProjectDown(Module):
    """Drop the extra dimensions used for addition
    
    TODO: figure out how to remove this layer
    """
    def __init__(self, slices: Tuple[int]) -> None:
        super().__init__()

        d_in, w_pos, st_, sym1_, sym2_, pos1_, pos2_, pos3_, scr1_, scr2_, scr3_, scr4_, scr5_ = slices
        weight = torch.cat([torch.eye(d_in), torch.zeros(d_in, w_pos)], dim=1)
        bias = torch.zeros(d_in)   
        linear = Linear(d_in+w_pos, d_in)       
        linear.weight = Parameter(weight)
        linear.bias = Parameter(bias)
        self.project_down = linear

    def forward(self, tgt: Tensor):
        x = tgt
        return self.project_down(x)


class IndicateVisitedPosition(Module):
    """Indicate if position has been written to by the machine.

    Notes        
        The f^0 layer indicating whether i' > 0 in Claim C.4 of WCM.

    Reference
        WCM
    """
    def __init__(self, slices: Tuple[int]):

        d_in, w_pos, st_, sym1_, sym2_, pos1_, pos2_, pos3_, scr1_, scr2_, scr3_, scr4_, scr5_ = slices

        super().__init__() 
        # query
        w_q = torch.zeros(w_pos+1, d_in)
        b_q = -torch.ones(w_pos+1)
        for k in range(w_pos):
            w_q[k, pos3_+k] = 2
        b_q[w_pos] = 1
        # key
        w_k = torch.zeros(w_pos+1, d_in)
        b_k = -torch.ones(w_pos+1)
        for k in range(w_pos):
            w_k[k, pos2_+k] = 2
        b_k[w_pos] = 0
        # value
        w_v = torch.zeros(d_in, d_in)
        b_v = torch.zeros(d_in)
        b_v[scr4_] = 1
        # null
        k_0 = torch.zeros(w_pos+1)
        k_0[w_pos] = w_pos-1
        v_0 = torch.zeros(d_in)
        self.w_q = Parameter(w_q)
        self.b_q = Parameter(b_q)
        self.w_k = Parameter(w_k)
        self.b_k = Parameter(b_k)
        self.w_v = Parameter(w_v)
        self.b_v = Parameter(b_v)
        self.k_0 = Parameter(k_0)
        self.v_0 = Parameter(v_0)

    def forward(self, x):
        x += attention_forward(x, x, x, 
            self.w_q, self.b_q, self.w_k, self.b_k, self.w_v, self.b_v, self.k_0, self.v_0)
        return x


class BinarySearchStep(Module):
    """Extract the next bit of the last step that wrote to this position if it exists.

    Notes
        The f^{j+1} layer performing a step of binary search in Claim C.5 of WCM.

    Reference 
        WCM
    """
    def __init__(self, slices: Tuple[int], j: int):
        super().__init__() 

        d_in, w_pos, st_, sym1_, sym2_, pos1_, pos2_, pos3_, scr1_, scr2_, scr3_, scr4_, scr5_ = slices

        # query
        w_q = torch.zeros(w_pos+j+2, d_in)
        b_q = -torch.ones(w_pos+j+2)
        for k in range(w_pos):
            w_q[k, pos3_+k] = 2
        for k in range(j):
            w_q[w_pos+k, scr3_+k] = 2

        b_q[w_pos+j:] = 1

        # key
        w_k = torch.zeros(w_pos+j+2, d_in)
        b_k = -torch.ones(w_pos+j+2)

        for k in range(w_pos):
            w_k[k, pos2_+k] = 2
        for k in range(j+1):
            w_k[w_pos+k, pos1_+k] = 2
        b_k[w_pos+j+1] = 0

        # value
        w_v = torch.zeros(d_in, d_in)
        b_v = torch.zeros(d_in)
        b_v[scr4_+2] = 1

        # null
        k_0 = torch.zeros(w_pos+j+2)
        k_0[w_pos+j+1] = w_pos+j
        v_0 = torch.zeros(d_in)

        # linear layer
        linear = Linear(d_in, d_in, bias=False)
        weight = torch.eye(d_in)
        weight[scr3_+j, scr4_+2] = 1
        weight[scr4_+2, scr4_+2] = 0
        linear.weight = Parameter(weight)

        self.w_q = Parameter(w_q)
        self.b_q = Parameter(b_q)
        self.w_k = Parameter(w_k)
        self.b_k = Parameter(b_k)
        self.w_v = Parameter(w_v)
        self.b_v = Parameter(b_v)
        self.k_0 = Parameter(k_0)
        self.v_0 = Parameter(v_0)
        self.linear = linear

    def forward(self, x):
        x += attention_forward(x, x, x, 
            self.w_q, self.b_q, self.w_k, self.b_k, self.w_v, self.b_v, self.k_0, self.v_0)
        x = self.linear(x)
        return x


class GetLastWrittenSymbol(Module):
    """Extract the last written symbol at this position if it exists.

    Notes    
        The f^{w_pos+1} layer in Claim C.6 of WCM.

    Reference
        WCM
    """
    def __init__(self, slices: Tuple[int]):
        super().__init__() 

        d_in, w_pos, st_, sym1_, sym2_, pos1_, pos2_, pos3_, scr1_, scr2_, scr3_, scr4_, scr5_ = slices

        # query
        w_q = torch.zeros(2*w_pos+1, d_in)
        b_q = -torch.ones(2*w_pos+1)
        for k in range(w_pos):
            w_q[k, pos3_+k] = 2
        for k in range(w_pos):
            w_q[w_pos+k, scr3_+k] = 2
        b_q[2*w_pos] = 1

        # key
        w_k = torch.zeros(2*w_pos+1, d_in)
        b_k = -torch.ones(2*w_pos+1)

        for k in range(w_pos):
            w_k[k, pos2_+k] = 2
        for k in range(w_pos):
            w_k[w_pos+k, pos1_+k] = 2
        b_k[2*w_pos] = 0

        # value
        w_v = torch.zeros(d_in, d_in)
        b_v = torch.zeros(d_in)

        for k in range(scr2_-scr1_):
            w_v[scr1_+k, sym2_+k] = 1

        # null
        k_0 = torch.zeros(2*w_pos+1)
        k_0[2*w_pos] = 2*w_pos-1
        v_0 = torch.zeros(d_in)

        # linear layer
        linear = Linear(d_in, d_in, bias=False)
        weight = torch.eye(d_in)
        for k in range(w_pos):
            weight[scr3_+k, scr3_+k] = 0
        linear.weight = Parameter(weight)

        self.w_q = Parameter(w_q)
        self.b_q = Parameter(b_q)
        self.w_k = Parameter(w_k)
        self.b_k = Parameter(b_k)
        self.w_v = Parameter(w_v)
        self.b_v = Parameter(b_v)
        self.k_0 = Parameter(k_0)
        self.v_0 = Parameter(v_0)
        self.linear = linear

    def forward(self, x):
        x += attention_forward(x, x, x, 
            self.w_q, self.b_q, self.w_k, self.b_k, self.w_v, self.b_v, self.k_0, self.v_0)
        x = self.linear(x)
        return x


class GetInitialSymbol(Module):
    """Get the initial symbol at this position if it exists.

    Notes
        The f layer in Lemmma C.7 of WCM.

    Reference
        WCM
    """
    def __init__(self, slices: Tuple[int]):
        super().__init__()

        d_in, w_pos, st_, sym1_, sym2_, pos1_, pos2_, pos3_, scr1_, scr2_, scr3_, scr4_, scr5_ = slices
        
        # query
        w_q = torch.zeros(w_pos+1, d_in)
        b_q = -torch.ones(w_pos+1)

        for k in range(w_pos):
            w_q[k, pos3_+k] = 2
                
        b_q[w_pos] = 1

        # key
        w_k = torch.zeros(w_pos+1, d_in)
        b_k = -torch.ones(w_pos+1)

        for k in range(w_pos):
            w_k[k, pos1_+k] = 2
       
        b_k[w_pos] = 0

        # value
        w_v = torch.zeros(d_in, d_in)
        b_v = torch.zeros(d_in)

        for k in range(scr3_-scr2_):
            w_v[scr2_+k, sym1_+k] = 1

        b_v[scr4_+1] = 1

        # null
        k_0 = torch.zeros(w_pos+1)
        k_0[w_pos] = w_pos-1
        v_0 = torch.zeros(d_in)
        
        self.w_q = Parameter(w_q)
        self.b_q = Parameter(b_q)
        self.w_k = Parameter(w_k)
        self.b_k = Parameter(b_k)
        self.w_v = Parameter(w_v)
        self.b_v = Parameter(b_v)
        self.k_0 = Parameter(k_0)
        self.v_0 = Parameter(v_0)

    def forward(self, tgt, memory):
        x = tgt
        x += attention_forward(x, memory, memory, 
            self.w_q, self.b_q, self.w_k, self.b_k, self.w_v, self.b_v, self.k_0, self.v_0)
        return x


class GetV(Module):
    """TODO

    Notes
        The v layer in Lemmma C.8, Step 1 of WCM.

        with 1-indexing this creates
        v_1 = h_1^scr4
        v_2 = ~h_1^scr4 ^ h_2^scr4
        v_3 = ~h_1^scr4 ^ ~h_2^scr4

    Reference
        WCM
    """
    def __init__(self, slices: Tuple[int]):
        super().__init__()

        d_in, w_pos, st_, sym1_, sym2_, pos1_, pos2_, pos3_, scr1_, scr2_, scr3_, scr4_, scr5_ = slices

        linear = Linear(d_in, d_in)
        w = torch.eye(d_in)
        b = torch.zeros(d_in)
        # v_1 = h_1^scr4
        # we already have w[tx.scr4_, tx.scr4_] = 1
        # v_2 = ~h_1^scr4 ^ h_2^scr4 => \phi(1-h_1+h_2-1)
        w[scr4_+1, scr4_] = -1
        w[scr4_+1, scr4_+1] = 1
        # v_3 = ~h_1^scr4 ^ ~h_2^scr4 => \phi(1-h_1+1-h_2-1)
        w[scr4_+2, scr4_] = -1
        w[scr4_+2, scr4_+1] = -1        
        b[scr4_+2] = 1
        linear.weight = Parameter(w)
        linear.bias = Parameter(b)
        self.linear = linear        

    def forward(self, tgt: Tensor):
        x = tgt    
        return F.relu(self.linear(x))   


class ArrangeSymbols(Module):
    """TODO

    Notes
        The TODO layer in Lemmma C.8, Steps 2,3,4 of WCM.

        AND(v_1, h_scr1) AND(v_2, h_scr2) AND(v_3, empty)

    Reference
        WCM
    """
    def __init__(self, slices: Tuple[int], empty_symbol: Tensor):
        super().__init__()

        d_in, w_pos, st_, sym1_, sym2_, pos1_, pos2_, pos3_, scr1_, scr2_, scr3_, scr4_, scr5_ = slices

        assert empty_symbol.ndim == 1
        assert empty_symbol.size(0) == sym2_ - sym1_
        
        linear = Linear(d_in, d_in)
        w = torch.eye(d_in)
        b = torch.zeros(d_in)    

        # iterate over length of alphabet
        for k in range(sym2_-sym1_):  
            # Step 2
            w[scr1_+k, scr4_] = 1
            b[scr1_+k] = -1
            # Step 3
            w[scr2_+k, scr4_+1] = 1
            b[scr2_+k] = -1
            # Step 4
            w[sym1_+k, sym1_+k] = 0
            w[sym1_+k, scr4_+2] = 1
            b[sym1_+k] = -1 + empty_symbol[k]
       
        linear.weight = Parameter(w)
        linear.bias = Parameter(b)
        self.linear = linear

    def forward(self, tgt):
        x = tgt    
        return F.relu(self.linear(x))


class CombineSymbols(Module):
    """TODO

    Notes
        The TODO layer in Lemmma C.8, Step 5 of WCM.

    Reference
        WCM
    """
    def __init__(self, slices: Tuple[int]):
        super().__init__()

        d_in, w_pos, st_, sym1_, sym2_, pos1_, pos2_, pos3_, scr1_, scr2_, scr3_, scr4_, scr5_ = slices

        linear = Linear(d_in, d_in, bias=False)
        w = torch.eye(d_in)

        # iterate over length of alphabet
        for k in range(sym2_-sym1_):  
            w[sym1_+k, scr1_+k] = 1
            w[sym1_+k, scr2_+k] = 1

        # Copy pos3 to pos2
        for k in range(w_pos):
            w[pos2_+k, pos3_+k] = 1
            w[pos2_+k, pos2_+k] = 0

        # zero out
        w[sym2_:pos1_] = 0
        w[pos1_:pos2_] = 0
        w[pos3_:scr1_] = 0
        w[scr1_:] = 0
        linear.weight = Parameter(w)

        self.linear_combine = linear

    def forward(self, tgt: Tensor):
        x = tgt        
        return self.linear_combine(x)