import torch
from math import ceil, log2
from torch import Tensor
from torch.nn import (
    Module, 
    ModuleList
)
from turing.wcm.networks import (
    ArrangeSymbols, 
    BinarySearchStep, 
    CombineSymbols, 
    FullAdder, 
    GetInitialSymbol,
    GetLastWrittenSymbol, 
    GetV, 
    IndicateVisitedPosition, 
    PreprocessForAdder, 
    ProjectDown, 
    Transition, 
)
from turing.types import (    
    Delta,
    State
)
from typing import List


# Default turing machine description
balanced_parentheses_transition_function = {
    ("I", "B") : ("R", "B",  1),
    ("R", "(") : ("R", "(",  1),
    ("R", ")") : ("M", "*", -1),
    ("R", "*") : ("R", "*",  1),
    ("R", "E") : ("V", "E", -1),
    ("M", "B") : ("F", "*", -1),
    ("M", "(") : ("R", "*",  1),
    ("M", "*") : ("M", "*", -1),
    ("V", "(") : ("F", "*", -1),
    ("V", "*") : ("V", "*", -1),
    ("V", "B") : ("T", "B",  1),
}

balanced_parentheses_terminal_states = ["T", "F"]


class Description():
    """Turing machine description

    delta: transition function
    terminal_states: possible end states

    """
    def __init__(self, delta: Delta = balanced_parentheses_transition_function, terminal_states: List[State] = balanced_parentheses_terminal_states):
        self.delta = delta
        self.terminal_states = terminal_states


class Simulator():
    """Class to simulate a turing machine for a given delta
    
    Available Translation Models:
        - Model "wcm" implements https://arxiv.org/abs/2107.13163

    """
  
    def __new__(self, description: Description=Description(), T: int=100, model: str="wcm"):
        
        if model == "wcm":
            return WCMSimulator(description, T)
        else:
            raise NotImplementedError("We haven't implemented this translation model.")


class WCMSimulator(Module):
    """Implements https://arxiv.org/abs/2107.13163
    """
  
    def __init__(self, description: Description, T: int):
        super().__init__()
       
        # turing machine settings
        self._delta = description.delta 
        self._terminal_states = description.terminal_states
        states = set()    
        alphabet = set()
        for (z, a), (z_next, u, _) in self.delta.items():
            states.add(z)
            states.add(z_next)
            alphabet.add(a)
            alphabet.add(u)

        self._alphabet = sorted(alphabet)
        self._states = sorted(states)
        self._a2i = {a: i for i, a in enumerate(self.alphabet)}
        self._z2i = {z: i for i, z in enumerate(self.states)}

        # neural network settings
        self._T = T
        self._w_pos = ceil(log2(self.T))
        self.w_scr = 2*len(alphabet) + self.w_pos + 3 + 2
        self.w = len(states) + 2*len(alphabet) + 3*self.w_pos + self.w_scr
        
        # starting indexes
        self._st = 0
        self._sym1 = len(states)
        self._sym2 = self._sym1 + len(alphabet)
        self._pos1 = self._sym2 + len(alphabet)
        self._pos2 = self._pos1 + self.w_pos
        self._pos3 = self._pos2 + self.w_pos
        self._scr1 = self._pos3 + self.w_pos
        self._scr2 = self._scr1 + len(alphabet)
        self._scr3 = self._scr2 + len(alphabet)
        self._scr4 = self._scr3 + self.w_pos
        # q encoding
        self._scr5 = self._scr4 + 3

        # network module definitions
        self._slices = self.w, self.w_pos, self.st_, self.sym1_, self.sym2_, self.pos1_, self.pos2_, self.pos3_, self.scr1_, self.scr2_, self.scr3_, self.scr4_, self.scr5_
        self.transition = Transition(self.slices, self.delta, self.states, self.alphabet, self.z2i, self.a2i)
        self.preprocess_for_adder = PreprocessForAdder(self.slices)
        self.adder_layers = ModuleList((FullAdder(d_in=self.w+self.w_pos, i=self._pos3+k, j=self.w+k, k=self._scr5) for k in range(self.w_pos-1, -1, -1)))
        # TODO: do we need the project_down layer?
        self.project_down = ProjectDown(self.slices)
        self.indicate_visited_position = IndicateVisitedPosition(self.slices)
        self.binary_search_layers = ModuleList((BinarySearchStep(self.slices, j=j) for j in range(self.w_pos)))
        self.get_last_written_symbol = GetLastWrittenSymbol(self.slices)
        self.get_initial_symbol = GetInitialSymbol(self.slices)      
        self.get_v = GetV(self.slices)
        empty_symbol = self.one_alphabet("E")
        self.arrange_symbols = ArrangeSymbols(self.slices, empty_symbol)
        self.combine_symbols = CombineSymbols(self.slices)

    @property
    def delta(self):     
        return self._delta

    @property
    def terminal_states(self):     
        return self._terminal_states

    @property
    def T(self):     
        return self._T

    @property
    def alphabet(self):     
        return self._alphabet

    @property
    def states(self):     
        return self._states

    @property
    def a2i(self):     
        return self._a2i

    @property
    def z2i(self):     
        return self._z2i

    @property
    def w_pos(self):     
        return self._w_pos

    @property
    def st_(self):     
        return self._st

    @property
    def sym1_(self):     
        return self._sym1

    @property
    def sym2_(self):     
        return self._sym2

    @property
    def pos1_(self):     
        return self._pos1

    @property
    def pos2_(self):     
        return self._pos2

    @property
    def pos3_(self):     
        return self._pos3

    @property
    def scr1_(self):     
        return self._scr1

    @property
    def scr2_(self):     
        return self._scr2

    @property
    def scr3_(self):     
        return self._scr3

    @property
    def scr4_(self):     
        return self._scr4

    @property
    def scr5_(self):     
        return self._scr5

    @property
    def slices(self):
        return self._slices

    def h(self, z: str, a: str, i: int, l: int):
        """Decoder input for timestep i
        """
        return torch.cat(self.h_partitions(z, a, i, l))

    def h_partitions(self, z: str, a: str, i: int, l: int):
        o_st = self.one_states(z)
        o_sym1 = self.one_alphabet(a)
        o_sym2 = torch.zeros(len(self.alphabet))
        o_pos1 = Tensor(self.Bin(i))
        o_pos2 = Tensor(self.Bin(l))
        o_pos3 = torch.zeros(self.w_pos)
        o_scr = torch.zeros(self.w_scr)
        return (o_st, o_sym1, o_sym2, o_pos1, o_pos2, o_pos3, o_scr)

    def beta(self, i: int):
        """Position embedding
        """
        beta = torch.zeros(self.w)
        beta[self._pos1:self._pos2] = Tensor(self.Bin(i))
        return beta        

    def one_alphabet(self, a):
        """Indicator/one-hot vector for symbol
        """
        o = torch.zeros([len(self.alphabet)])
        o[self.a2i[a]] = 1
        return o

    def one_alphabet_inverse(self, x: Tensor) -> str:       
        assert x.ndim == 1
        assert len(self.alphabet) == x.numel()
        n = torch.count_nonzero(x)
        if n == 1:
            i = x.nonzero()
            i = i.item()
            return self.alphabet[i]
        else:
            return "UNK"

    def one_states(self, z):
        """Indicator/one-hot vector for state
        """
        o = torch.zeros([len(self.states)])
        o[self.z2i[z]] = 1
        return o

    def one_states_inverse(self, x: Tensor) -> str:        
        assert x.ndim == 1
        assert len(self.states) == x.numel()
        n = torch.count_nonzero(x)
        if n == 1:
            i = x.nonzero()
            i = i.item()
            return self.states[i]
        else:
            return "UNK"

    def Bin(self, i: int) -> List[int]:
        """Binary representation of i as a List[int]
        """
        stringbin = f"{i:0{self.w_pos}b}"
        return list(map(int, stringbin))

    def Bin_inverse(self, x: Tensor) -> int:        
        assert x.ndim == 1
        x = x.dot(2.**torch.arange(x.numel()-1,-1,-1))
        i = x.item()
        return int(i)

    def project_and_normalize(self, o: Tensor, subspace: str):
        """Project tensor to named subspace. Decode meaning of subspace.
        """
        o = o.detach()

        if o.ndim == 2:  
            o = o[-1, :]   

        projections = {
            "st": o[self._st:self._sym1],
            "sym1": o[self._sym1:self._sym2],
            "sym2": o[self._sym2:self._pos1],
            "pos1": o[self._pos1:self._pos2],
            "pos2": o[self._pos2:self._pos3],
            "pos3": o[self._pos3:self._scr1],            
            "scr1": o[self._scr1:self._scr2],
            "scr2": o[self._scr2:self._scr3],
            "scr3": o[self._scr3:self._scr4],
            "scr4": o[self._scr4:self._scr5],
            "scr5": o[self._scr5:self.w],
        }
        x = projections.get(subspace)
        if subspace == "st":
            return x, self.one_states_inverse(x)
        elif subspace in ["sym1", "sym2"]:               
            return x, self.one_alphabet_inverse(x)
        elif subspace in ["pos1", "pos2", "pos3"]:
            return x, self.Bin_inverse(x)
        elif x.numel() == len(self.states):
            # TODO: this is a bad hack
            return x, self.one_states_inverse(x)
        elif x.numel() == len(self.alphabet):
            # TODO: this is a bad hack
            return x, self.one_alphabet_inverse(x)
        elif x.numel() == self.w_pos:
            # TODO: this is a bad hack
            return x, self.Bin_inverse(x)
        return x, None

    def projecto(self, o, subspaces=["st", "sym1", "sym2", "pos1", "pos2", "pos3", "scr1", "scr2", "scr3", "scr4", "scr5"], info_only=False):
        """Project decoder layer to meaningful subspaces
        """        
        for subspace in subspaces:
            projection, info = self.project_and_normalize(o, subspace)
            if info_only:
                print(f"{subspace: <6}", info)
            else:
                print(f"{subspace: <6}", projection, info)

    def forward(self, tgt: Tensor, memory: Tensor):
        """
        tgt is the decoder input
        memory is the encoder output
        """
        x = tgt        
        x = self.transition(x)
        # get the written symbol
        _, u = self.project_and_normalize(x, "sym2")
        x = self.preprocess_for_adder(x)
        for layer in self.adder_layers:
            x = layer(x)
        x = self.project_down(x)             
        x = self.indicate_visited_position(x)        
        for layer in self.binary_search_layers:
            x = layer(x)
        x = self.get_last_written_symbol(x)
        x = self.get_initial_symbol(x, memory)
        x = self.get_v(x)
        x = self.arrange_symbols(x)        
        x = self.combine_symbols(x)
        return x, u

    def encode_tape(self, tape: str):
        x = torch.zeros(len(tape), self.w)
        for i, s in enumerate(tape):
            if i == self.T: break
            x[i, self._sym1:self._sym2] = self.one_alphabet(s)
            x[i, self._pos1:self._pos2] = torch.Tensor(self.Bin(i))            
        return x

    def simulate(self, tape: str):
        """Simulate the turing machine for the given tape via the WCM network
        """
        E = self.encode_tape(tape)
        cursor = "^"
        pad = " "
        n = len(tape)
        h = self.h("I", "B", 0, 0)
        h = h.unsqueeze(0)
        for step in range(1, self.T):
            _, z = self.project_and_normalize(h, "st")
            _, a = self.project_and_normalize(h, "sym1")
            _, i = self.project_and_normalize(h, "pos1")
            _, head = self.project_and_normalize(h, "pos2")   

            # print tape
            print("  ", tape, sep="")
            # print machine info    
            print(z, " ", pad*head, cursor, pad*(n-head), sep="")

            if z in self.terminal_states:
                return z

            o, u = self.forward(h, E)

            tape = tape[:head] + u + tape[head+1:]
            o = o[[-1], :]
            h = torch.cat([h, o + self.beta(step)])