from graphviz import Digraph
from typing import Dict, List, Tuple

StateType = str
SymbolType = str
DirectionType = int

def generate(
        delta: Dict[Tuple[StateType, SymbolType], Tuple[StateType,  SymbolType, DirectionType]], 
        initial_state: StateType = "I",
        terminal_states: List[StateType]=[]):
    """Generate graphviz graph from dictionary representing the Turing machine's transition function delta.

    """

    # get unique states and symbols    
    states = set()    
    alphabet = set()
    for (z, a), (z_next, u, _) in delta.items():
        states.add(z)
        states.add(z_next)
        alphabet.add(a)
        alphabet.add(u)

    dot = Digraph(
        "tm", 
        format="png", 
        node_attr={
            'color': 'lightblue2', 
            'fontcolor': 'black',
            'style': 'filled',
        }, 
        edge_attr={
            'color': 'gray',
            'fontcolor': 'white',
            'style': 'filled',
        }, 
        graph_attr={
            'bgcolor': 'grey12',
            'fontname': 'Cascadia Mono'
        }
    )

    # nodes
    def get_shape(state):
        if state == initial_state: return "diamond"
        if state in terminal_states: return "square"
        return "circle"

    for state in states:        
        dot.node(state, state, shape=get_shape(state))       

    # edges
    for (z, a), (z_next, u, _) in delta.items():
        dot.edge(z, z_next, a)

    return dot