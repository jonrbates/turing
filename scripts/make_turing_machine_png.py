from turing.graph import generate
from turing.translators import Translator
tx = Translator()
dot = generate(tx.delta, terminal_states=["T", "F"])
dot.render("tm")