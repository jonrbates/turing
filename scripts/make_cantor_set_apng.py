import json
import tempfile
import os
import matplotlib.pyplot as plt
from PIL import Image
from turing.ss.simulator import encoding_function

with open("scripts/terminal_colors.json") as _f:
    _C = json.load(_f)

import matplotlib
matplotlib.rcParams["font.family"] = _C["font"]


def show_encoding_in_cantor_set_grid(s: str):
    face_color           = _C["bg"]
    axis_color           = _C["tape"]
    grid_color           = _C["dim"]
    cantor_set_color     = _C["dim"]
    active_encoding_color = _C["orange"]
    active_text_color    = _C["orange"]

    fig, ax = plt.subplots(1, 1)

    strings=['']
    u, l = 0, -1
    for level in range(5):
        strings = [s+'0' for s in strings] + [s+'1' for s in strings]
        for a in strings:
            x = encoding_function(a, base=4, p=1/2)
            plt.plot([x, x], [u, l], color=cantor_set_color, linewidth=2)
        u, l = l, l - (u - l)

    u, l = -len(s)+1, -len(s)
    x = encoding_function(s, base=4, p=1/2)
    plt.plot([x, x], [u, l], color=active_encoding_color, linewidth=2)
    plt.text(x, .5*(u+l), f' \"{s}\"', color=active_text_color, weight='bold')

    # x axis
    ax.set_xlabel('encoding')
    ax.spines['top'].set_position('zero')
    ax.spines['top'].set_color(axis_color)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.tick_top()
    _strs = ['0', '1', '00', '01', '10', '11']
    rug = sorted({0.0, 1.0} | {float(encoding_function(t, base=4, p=1/2)) for t in _strs})
    plt.xticks(rug, [f'{x:.4g}' for x in rug], rotation=45, color=axis_color)
    ax.xaxis.label.set_color(axis_color)
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='x', colors=axis_color)
    plt.xlim([0, 1])

    # y axis
    ax.set_ylabel('string length')
    ax.spines['left'].set_color(axis_color)
    ax.spines['right'].set_visible(False)
    plt.yticks([-0.5, -1.5, -2.5, -3.5, -4.5], color=axis_color)
    ax.tick_params(axis='y', colors=axis_color)
    ax.set_yticklabels([1, 2, 3, 4, 5])
    ax.yaxis.label.set_color(axis_color)
    plt.ylim([-5, 0])

    # down arrow
    ax.plot(0, -5, ls="", marker="v", ms=10, color=axis_color,
        transform=ax.get_yaxis_transform(), clip_on=False)

    # background
    fig.set_facecolor(face_color)
    ax.set_facecolor(face_color)
    ax.grid(True, linewidth=0.2, color=grid_color)
    return fig, ax


out = "docs/img/cantor_set.png"

with tempfile.TemporaryDirectory() as d:
    fig_names = []
    for s in ["0", "00", "000", "0000", "00000", "1", "10", "101", "1011", "10110"]:
        fig, ax = show_encoding_in_cantor_set_grid(s)
        name = os.path.join(d, f'cantor_set_{s}.png')
        fig.savefig(name, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        fig_names.append(name)

    frames = [Image.open(f).convert("RGBA") for f in fig_names]

frames[0].save(
    out, save_all=True, append_images=frames[1:],
    loop=0, duration=1000,
)
fw, fh = frames[0].size
print(f"Saved {out}  ({fw}x{fh}, {len(frames)} frames)")
