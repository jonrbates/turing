"""Render Turing-machine tape animation as an APNG.

Uses Ubuntu Mono TTF rendered via PIL at 2× resolution — lossless, full colour,
Retina-sharp when displayed at half width in the README.
"""
import sys
sys.path.insert(0, "src")

import json
from matplotlib.font_manager import findfont, FontProperties
from PIL import Image, ImageDraw, ImageFont
from turing.wcm.simulator import Simulator

# ── config (loaded from terminal_colors.json) ─────────────────────────────────
def _resolve_font(spec: str) -> str:
    if spec.startswith("/"):
        return spec
    return findfont(FontProperties(family=spec))

def _hex(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

with open("scripts/terminal_colors.json") as f:
    _C = json.load(f)

BG     = _hex(_C["bg"])
TAPE   = _hex(_C["tape"])
LABELS = _hex(_C["labels"])
HEAD   = _hex(_C["head"])
STATE  = _hex(_C["state"])

PAD  = 32

font      = ImageFont.truetype(_resolve_font(_C["font"]), size=48)
font_bold = ImageFont.truetype(findfont(FontProperties(family=_C["font"], weight="bold")), size=48)
CW   = font.getbbox("X")[2]
LH   = font.getbbox("X")[3] + 8  # line height + gap

# Use the widest label to set column width so all labels align
LW = max(font.getbbox(s)[2] for s in ("tape:  ", "head:  ", "state: "))


def make_frame(tape: str, state: str, head: int,
               canvas_w: int, canvas_h: int) -> Image.Image:
    img = Image.new("RGB", (canvas_w, canvas_h), BG)
    d   = ImageDraw.Draw(img)

    r0, r1, r2 = PAD, PAD + LH, PAD + 2 * LH

    d.text((PAD, r0), "tape:  ", font=font, fill=LABELS)
    d.text((PAD + LW, r0), tape, font=font, fill=TAPE)

    d.text((PAD, r1), "head:  ", font=font, fill=LABELS)
    d.text((PAD + LW + CW * head, r1), "^", font=font_bold, fill=HEAD)

    d.text((PAD, r2), "state: ", font=font, fill=LABELS)
    d.text((PAD + LW, r2), state, font=font, fill=STATE)

    return img


# ── simulation ────────────────────────────────────────────────────────────────
tx    = Simulator()
tape  = "B()((()(()))())E"
head  = 0
state = "I"
final_states = {"T", "F"}

# Pre-compute fixed canvas size (tape length is constant; head stays within tape)
tape_w    = font.getbbox(tape)[2]
canvas_w  = PAD + LW + tape_w + PAD
canvas_h  = PAD + LH * 3 + PAD

steps: list[tuple] = []   # (tape, state, head, duration_ms)
steps.append((tape, state, head, 600))

while state not in final_states:
    state, write, move = tx.delta[(state, tape[head])]
    tape  = tape[:head] + write + tape[head + 1:]
    head += move
    steps.append((tape, state, head, 500))

steps.append((tape, state, head, 1200))  # hold on final

frames    = [make_frame(t, s, h, canvas_w, canvas_h) for t, s, h, _ in steps]
durations = [ms for *_, ms in steps]

# ── save as APNG (lossless, full colour) ─────────────────────────────────────
out = "docs/img/bptape_terminal.png"
fw, fh = frames[0].size
frames[0].save(
    out, save_all=True, append_images=frames[1:],
    loop=0, duration=durations,
)
print(f"Saved {out}  ({fw}x{fh}, {len(frames)} frames)  — display at width={fw//2}")
