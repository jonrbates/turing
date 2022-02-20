from turing.graph import generate
from turing.wcm.simulator import Simulator
tx = Simulator()
dot = generate(tx.delta, terminal_states=["T", "F"])
dot.render("tm")