"""Render 2-stack machine animation as an APNG.

Uses Ubuntu Mono TTF rendered via PIL at 2× resolution — lossless, full colour,
Retina-sharp when displayed at half width in the README.
"""
import sys
sys.path.insert(0, "src")

import json
from matplotlib.font_manager import findfont, FontProperties
from PIL import Image, ImageDraw, ImageFont
from turing.ss.simulator import (
    Description,
    balanced_parentheses_delta_stack,
    balanced_parentheses_terminal_states,
)

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

BG           = _hex(_C["bg"])
TAPE         = _hex(_C["tape"])
LABELS       = _hex(_C["labels"])
STATE        = _hex(_C["state"])
TOP_OF_STACK = _hex(_C["top_of_stack"])
MATCH        = _hex(_C["match"])

PAD  = 32

font      = ImageFont.truetype(_resolve_font(_C["font"]), size=48)
font_bold = ImageFont.truetype(findfont(FontProperties(family=_C["font"], weight="bold")), size=48)
CW   = font.getbbox("X")[2]
LH   = font.getbbox("X")[3] + 8  # line height + gap

LW = max(font.getbbox(s)[2] for s in ("stack 0: ", "stack 1: ", "state:   "))


def make_frame(s1: str, s2: str, state: str,
               canvas_w: int, canvas_h: int,
               top_color: tuple = TOP_OF_STACK) -> Image.Image:
    img = Image.new("RGB", (canvas_w, canvas_h), BG)
    d   = ImageDraw.Draw(img)

    for i, (lbl, stack) in enumerate([("stack 0: ", s1), ("stack 1: ", s2)]):
        y = PAD + i * LH
        d.text((PAD, y), lbl, font=font, fill=LABELS)
        if stack:
            d.text((PAD + LW, y), stack[0], font=font_bold, fill=top_color)
            if len(stack) > 1:
                d.text((PAD + LW + CW, y), stack[1:], font=font, fill=TAPE)

    y2 = PAD + 2 * LH
    d.text((PAD, y2), "state:   ", font=font, fill=LABELS)
    d.text((PAD + LW, y2), state, font=font, fill=STATE)

    return img


# ── simulation ────────────────────────────────────────────────────────────────
desc  = Description(balanced_parentheses_delta_stack, balanced_parentheses_terminal_states)
delta = desc.delta
final_states = set(desc.terminal_states)


def top(stack):
    return None if not stack else (0 if stack[0] == "(" else 1)


def apply_action(action, stack):
    if action == "pop":    return stack[1:]
    if action == "push 0": return "(" + stack
    if action == "push 1": return ")" + stack
    return stack


s1, s2, z = "()(())", "", "I"

steps: list[tuple] = []  # (s1, s2, state, top_color, duration_ms)

for _ in range(60):
    steps.append((s1, s2, z, TOP_OF_STACK, 600))
    if z in final_states:
        break
    key = (z, top(s1), top(s2))
    if key not in delta:
        break
    z_new, a1, a2 = delta[key]
    if a1 == a2 == "pop":
        steps.append((s1, s2, z, MATCH, 500))  # pause + green flash on match
    z  = z_new
    s1 = apply_action(a1, s1)
    s2 = apply_action(a2, s2)

steps.append((s1, s2, z, TOP_OF_STACK, 1200))  # hold on final

# Fixed canvas = widest stack across all steps
max_stack_w = max(
    max((font.getbbox(s1)[2] if s1 else 0,
         font.getbbox(s2)[2] if s2 else 0,
         CW))
    for s1, s2, *_ in steps
)
canvas_w = PAD + LW + max_stack_w + PAD
canvas_h = PAD + LH * 3 + PAD

frames    = [make_frame(s1, s2, st, canvas_w, canvas_h, tc) for s1, s2, st, tc, _ in steps]
durations = [ms for *_, ms in steps]

# ── save as APNG (lossless, full colour) ─────────────────────────────────────
out = "docs/img/bpstack_terminal.png"
fw, fh = frames[0].size
frames[0].save(
    out, save_all=True, append_images=frames[1:],
    loop=0, duration=durations,
)
print(f"Saved {out}  ({fw}x{fh}, {len(frames)} frames)  — display at width={fw//2}")
