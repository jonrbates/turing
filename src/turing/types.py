from typing import Dict, Literal, NamedTuple, Tuple

State = str
Symbol = str
Direction = int
Delta = Dict[Tuple[State, Symbol], Tuple[State,  Symbol, Direction]]

StackOp = Literal['noop', 'pop', 'push0', 'push1']
StackElement = Literal[0, 1, '*']
StackNonEmpty = Literal[False, True, '*']

class CurrentStateAndStack(NamedTuple):
    state: State
    top0: StackElement
    top1: StackElement
    nonempty0: StackNonEmpty
    nonempty1: StackNonEmpty

class NextStateAndStack(NamedTuple):
    state: State
    op0: StackOp
    op1: StackOp

StackDelta = Dict[CurrentStateAndStack, NextStateAndStack]