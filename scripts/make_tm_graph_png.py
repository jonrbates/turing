"""Generate an xkcd-style Turing machine transition graph using matplotlib."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
import numpy as np
from turing.wcm.simulator import Simulator

# ── Colors (shared config) ────────────────────────────────────────────────────
with open("scripts/terminal_colors.json") as _f:
    _J = json.load(_f)

C = dict(
    bg        = _J["bg"],
    ink       = _J["tape"],
    node_fill = _J["bg"],
    label     = _J["labels"],
    outline   = _J["bg"],
)

# ── TM data ───────────────────────────────────────────────────────────────────
tx = Simulator()
delta = tx.delta
terminal_states = ["T", "F"]
initial_state = "I"

# Hand-tuned layout (scaled to ~80% of original edge lengths)
pos = {
    'I': (0.0, 1.0),
    'R': (2.0, 1.0),
    'V': (4.0, 1.72),
    'T': (6.0, 1.72),
    'M': (4.0, 0.28),
    'F': (6.0, 0.28),
}

# Self-loop "up" direction per node (angle in radians, pointing away from graph)
self_loop_angle = {
    'R': np.pi / 2,    # above
    'M': -np.pi / 2,   # below
    'V': np.pi / 2,    # above
}

# Group edges by (from, to), collect read-symbols
edges: dict[tuple, list] = {}
for (z, a), (z_next, u, d) in delta.items():
    key = (z, z_next)
    edges.setdefault(key, []).append(a)

NODE_R = 0.30
LOOP_SPREAD = 0.28   # half-angle spread of self-loop endpoints
LOOP_RAD = 3.2       # FancyArrowPatch arc radius for self-loops


def circle_point(center, r, angle):
    return (center[0] + r * np.cos(angle), center[1] + r * np.sin(angle))


def boundary(state):
    """Radial gap from node centre to use for arrow start/end points."""
    if state in terminal_states:
        return NODE_R * 1.25   # just outside the square (half-side = NODE_R*1.15)
    if state == initial_state:
        return NODE_R * 1.4    # diamond half-diagonal
    return NODE_R


def midpoint_along_arc(p1, p2, rad):
    """Approximate label position for a curved arrow (arc3 style)."""
    mx = (p1[0] + p2[0]) / 2
    my = (p1[1] + p2[1]) / 2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = np.hypot(dx, dy)
    if length < 1e-9:
        return mx, my
    perp = np.array([-dy, dx]) / length
    offset = rad * length * 0.5
    return mx + perp[0] * offset, my + perp[1] * offset


with plt.xkcd(scale=1.2, length=120, randomness=3):
    import matplotlib.patheffects as _pe
    plt.rcParams.update({
        'font.family': 'xkcd Script',
        'path.effects': [_pe.withStroke(linewidth=4, foreground=C['outline'])],
    })
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(C['bg'])
    ax.set_facecolor(C['bg'])
    ax.set_xlim(-1.0, 7.2)
    ax.set_ylim(-0.9, 2.8)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Draw edges ────────────────────────────────────────────────────────────
    for (z, z_next), labels in edges.items():
        label_str = ', '.join(sorted(labels))
        x1, y1 = pos[z]
        x2, y2 = pos[z_next]

        if z == z_next:
            ang = self_loop_angle[z]
            p_start = circle_point(pos[z], NODE_R, ang - LOOP_SPREAD)
            p_end   = circle_point(pos[z], NODE_R, ang + LOOP_SPREAD)
            arrow = FancyArrowPatch(
                p_start, p_end,
                connectionstyle=f'arc3,rad={LOOP_RAD}',
                arrowstyle='-|>',
                mutation_scale=18,
                color=C['ink'], linewidth=1.6, zorder=1,
            )
            ax.add_patch(arrow)
            lx = x1 + (NODE_R + 0.42) * np.cos(ang)
            ly = y1 + (NODE_R + 0.42) * np.sin(ang)
            ax.text(lx, ly, label_str, ha='center', va='center',
                    fontsize=16, color=C['label'])

        else:
            curved = (z_next, z) in edges
            rad = 0.25 if curved else 0.0

            dx, dy = x2 - x1, y2 - y1
            dist = np.hypot(dx, dy)
            nx, ny = dx / dist, dy / dist
            p_start = (x1 + nx * boundary(z),      y1 + ny * boundary(z))
            p_end   = (x2 - nx * boundary(z_next), y2 - ny * boundary(z_next))

            arrow = FancyArrowPatch(
                p_start, p_end,
                connectionstyle=f'arc3,rad={rad}',
                arrowstyle='-|>',
                mutation_scale=18,
                color=C['ink'], linewidth=1.6, zorder=1,
            )
            ax.add_patch(arrow)

            lx, ly = midpoint_along_arc(p_start, p_end, rad)
            if abs(rad) > 0.01:
                perp = np.array([-(p_end[1]-p_start[1]), p_end[0]-p_start[0]])
                perp /= np.linalg.norm(perp) + 1e-9
                lx += perp[0] * 0.12
                ly += perp[1] * 0.12
            else:
                ly += 0.15
            ax.text(lx, ly, label_str, ha='center', va='center',
                    fontsize=16, color=C['label'])

    # ── Draw nodes ────────────────────────────────────────────────────────────
    for state, (x, y) in pos.items():
        if state == initial_state:
            r = NODE_R * 1.4
            pts = np.array([[x - r, y], [x, y + r], [x + r, y], [x, y - r], [x - r, y]])
            ax.fill(pts[:, 0], pts[:, 1], color=C['node_fill'], zorder=2)
            ax.plot(pts[:, 0], pts[:, 1], color=C['ink'], linewidth=2.0, zorder=3)
        elif state in terminal_states:
            s = NODE_R * 1.15
            sq = FancyBboxPatch((x - s, y - s), 2 * s, 2 * s,
                                boxstyle='square,pad=0',
                                facecolor=C['node_fill'], edgecolor=C['ink'],
                                linewidth=2.0, zorder=2)
            ax.add_patch(sq)
        else:
            c = Circle((x, y), NODE_R, facecolor=C['node_fill'], edgecolor=C['ink'],
                       linewidth=2.0, zorder=2)
            ax.add_patch(c)

        ax.text(x, y, state, ha='center', va='center',
                fontsize=17, fontweight='bold', color=C['ink'], zorder=5)

    plt.tight_layout()
    out = 'docs/img/tm.png'
    plt.savefig(out, dpi=600, bbox_inches='tight', facecolor=C['bg'])
    print(f"Saved {out}")
