# Siegelmann & Sontag: Neural Nets as Turing Machines

Siegelmann and Sontag proved in 1992 that a fixed recurrent neural network can simulate any Turing machine. Their construction works by encoding the Turing machine tape as numbers stored in the network's hidden state.

This document explains their approach and how to use the implementation in this repo.

---

### The Core Idea: Stacks Instead of a Tape

Rather than operating on a tape directly, Siegelmann & Sontag reformulate the problem using *$p$-stack machines*. A $p$-stack machine is equivalent to a Turing machine but stores memory in $p$ stacks instead of a tape. Each stack supports three operations: **push**, **pop**, and **peek** (read the top).

For the balanced parentheses problem, two stacks are enough:

<p><img src="img/bpstack_terminal.png" alt="2-stack machine solving balanced parentheses" width="40%" /></p>

The transition function for a $p$-stack machine takes the current state and the top of each stack as inputs, and outputs the next state and a stack operation for each stack:

```python
# (state, top_of_stack_1, top_of_stack_2) → (next_state, op_stack_1, op_stack_2)
# ops: 'noop', 'push (', 'push )', 'pop'
# None = stack is empty; '*' = wildcard (matches any value)

balanced_parentheses_delta_stack = {
    ('*',  '(',  '(') : ('A',  'pop',    'push ('),
    ('*',  '(', None) : ('A',  'pop',    'push ('),
    ('*',  ')',  '(') : ('A',  'pop',       'pop'),
    ('I', None, None) : ('T', 'noop',      'noop'),
    ('A', None, None) : ('T', 'noop',      'noop'),
    # not balanced
    ('*',  ')', None) : ('F', 'noop',      'noop'),
    ('*', None,  '(') : ('F', 'noop',      'noop'),
    # terminal states loop
    ('F',  '*',  '*') : ('F', 'noop',      'noop'),
    ('T',  '*',  '*') : ('T', 'noop',      'noop'),
    ('*',  '*',  ')') : ('F', 'noop',      'noop'),
}
terminal_states = ['T', 'F']
```

---

### Encoding a Stack as a Number

The key mathematical trick is encoding an entire stack as a single rational number using a *Cantor-set encoding*. For a binary stack (symbols 0 and 1), each stack value is encoded as:

$$\delta(a_1 a_2 \cdots a_k) = \sum_{i=1}^{k} \frac{b - 1 + 4p(a_i - 1)}{b^i}$$

where $b$ is the base and $p$ is a scaling factor. For example with $b=4$, $p=1/2$:

| Stack contents | Encoded value |
|---|---|
| (empty) | 0 |
| `[0]` | 1/4 |
| `[1]` | 3/4 |
| `[0, 0, 0]` | 21/64 |


<p align="center"><img src="img/cantor_set.png" alt="cycling values in the 4-Cantor set" width="60%" /></p>

The encoding has a crucial property: **push, pop, and peek are all linear functions of the encoded value**. This means the network can manipulate the stack using only linear layers and a saturated ReLU activation $\sigma(x) = \mathrm{clamp}(x, 0, 1)$.

---

### Two Network Architectures

The paper describes two constructions:

**4-layer (`version=4`)** - uses 4 layers per step. Each layer is a stage of the computation. Fully implemented and working.

**1-layer (`version=1`)** - uses a single recurrent layer, operating in "real time" (one network step per Turing machine step). Uses base $b = 10p^2$ (= 40 for $p=2$) and float64 arithmetic. Reliable for strings up to ~8 characters due to error amplification in the pop operation ($b\times$ per pop).

---

### Usage

```python
from turing.ss.simulator import (
    Description, Simulator,
    balanced_parentheses_delta_stack,
    balanced_parentheses_terminal_states,
)

description = Description(
    balanced_parentheses_delta_stack,
    balanced_parentheses_terminal_states,
)

# 4-layer
sim4 = Simulator(description, version=4)
sim4.simulate("(()())")

# 1-layer - requires float64, reliable up to ~8 chars
sim1 = Simulator(description, version=1)
sim1.simulate("(())", T=12)
```

**Output:** each step prints the current machine state.

```
# version=4 output (state vector, one-hot):
   I  ['', '(()())']
   A  ['(', '()()']
   ...
   T  ['', '']

# version=1 output (state name):
   I
   A
   ...
   T
```

**Defining your own machine:**

The `Description` class accepts any p-stack transition function. Constraints:
- **Alphabet must have at most 2 symbols** (mapped internally to 0/1)
- `'*'` in a key is a wildcard matching any value
- `None` means the stack is empty

```python
my_delta = {
    # (state, top_stack1, top_stack2) : (next_state, op1, op2)
    ("start", None, None): ("done", "noop", "noop"),
}
description = Description(my_delta, terminal_states=["done"])
tx = Simulator(description, version=4)
```

---

### Architecture

<p align="center"><img src="img/ss1_architecture.png" alt="Siegelmann-Sontag architecture" width="60%" /></p>

The network state at each step is a vector containing:
- **State part** - one-hot encoding of the current machine state
- **Stack part** - each stack encoded as a single scalar using the Cantor-set encoding

At each step, a *configuration detector* reads the current state and the top of each stack, identifies the matching transition rule, and produces the next state and stack operations. The weights are set analytically via least-squares from the transition function.

The `saturated_relu` $\sigma(x) = \mathrm{clamp}(x,0,1)$ serves as the nonlinearity throughout, since its behavior on rational inputs is exact.

---

### Further Reading

- [Siegelmann & Sontag (1995)](https://www.sciencedirect.com/science/article/pii/S0022000085710136)
- `src/turing/ss/networks.py` - `SiegelmannSontag4`, `ConfigurationDetector4`, `SiegelmannSontag1`, `ConfigurationDetector1`