"""
pitch.py — mplsoccer pitch drawing utilities for the PPN Football Dashboard.
Coordinates: origin at centre, x: -52.5 → +52.5, y: -34 → +34.
(SkillCorner coordinate system: pitch_length=105, pitch_width=68)
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from mplsoccer import Pitch

# ── Kept for backward-compat references in dashboard.py ──────────────────────
PITCH_BG   = "#1a472a"
PAPER_BG   = "#0d0d0d"
LINE_COLOR = "white"

TEAM_COLORS = {
    "Auckland FC": "#3498db",
    "Newcastle":   "#e74c3c",
}

ERROR_COLORS = {
    "Passer Error":                    "#e74c3c",
    "Structure Error — Overambitious": "#e67e22",
    "Structure Error — Trapped":       "#9b59b6",
    "Success":                         "#2ecc71",
}

_PITCH_KW = dict(
    pitch_type="skillcorner",
    pitch_length=105,
    pitch_width=68,
    pitch_color="grass",
    line_color="white",
    stripe=True,
    stripe_color="#1a5c2a",
)


# ── Core pitch builder ────────────────────────────────────────────────────────

def make_pitch_fig(title="", height=520, show_legend=True, bg=PITCH_BG):
    """
    Returns (fig, ax) using mplsoccer Pitch.
    height is treated as approximate pixels at 96 dpi for sizing.
    """
    height_in = max(height / 96, 5.0)
    width_in  = height_in * (105 / 68) * 1.08   # slight extra for margins

    pitch = Pitch(**_PITCH_KW)
    fig, ax = pitch.draw(figsize=(width_in, height_in))

    fig.patch.set_facecolor("#0d0d0d")

    if title:
        ax.set_title(title, color="white", fontsize=9.5, pad=6,
                     fontfamily="sans-serif")

    return fig, ax


# ── Player scatter ────────────────────────────────────────────────────────────

def add_players(ax, snap_df, sender_num, receiver_num, attacking_team,
                show_names=True):
    """
    Plot all players from a snapshot DataFrame onto a matplotlib Axes.
    snap_df columns: team, player_number, player_name, x, y
    """
    for team, grp in snap_df.groupby("team"):
        color = TEAM_COLORS.get(team, "#888888")

        for _, p in grp.iterrows():
            pnum  = int(p["player_number"])
            pname = str(p.get("player_name", ""))
            x, y  = float(p["x"]), float(p["y"])

            is_sender   = (pnum == sender_num   and team == attacking_team)
            is_receiver = (pnum == receiver_num and team == attacking_team)

            marker = "D"   if is_sender   else "o"
            size   = 200   if is_sender   else 170 if is_receiver else 100
            lw     = 2.5   if (is_sender or is_receiver) else 1.0
            edge   = "gold" if is_sender else "white"

            ax.scatter(x, y, s=size, c=color, marker=marker,
                       linewidths=lw, edgecolors=edge, zorder=5)

            # Player number drawn inside the dot
            ax.text(x, y, str(pnum),
                    ha="center", va="center",
                    fontsize=6.5, fontweight="bold", color="white",
                    zorder=6)

            # Short surname label below
            if show_names and pname:
                short_name = pname.split()[-1]
                ax.text(x, y - 2.8, short_name,
                        ha="center", va="top",
                        fontsize=7, color="white",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor="black", alpha=0.55,
                                  edgecolor="none"),
                        zorder=6)


# ── Ball ─────────────────────────────────────────────────────────────────────

def add_ball(ax, bx, by):
    """Add the ball marker to the pitch."""
    if bx is None or by is None:
        return
    try:
        bx, by = float(bx), float(by)
        if np.isnan(bx) or np.isnan(by):
            return
    except (TypeError, ValueError):
        return

    ax.scatter(bx, by, s=160, c="white", marker="o",
               linewidths=1.5, edgecolors="#333333", zorder=7,
               label="Ball")
    ax.text(bx, by + 2.5, "⚽",
            ha="center", va="bottom", fontsize=11, zorder=8)


# ── Ghost players ─────────────────────────────────────────────────────────────

def add_ghost_players(ax, opos_df, show_nudge_arrow=True):
    """
    Overlay ghost (optimised) positions for off-ball players.
    opos_df columns: orig_x/y, opt_x/y, player_number, player_name,
                     nudge_m, g_before, g_after, is_receiver
    """
    for _, row in opos_df.iterrows():
        ox, oy = float(row["orig_x"]), float(row["orig_y"])
        nx, ny = float(row["opt_x"]),  float(row["opt_y"])
        pnum   = int(row["player_number"])
        pname  = str(row.get("player_name", ""))
        nudge  = float(row["nudge_m"])
        g_bef  = float(row["g_before"])
        g_aft  = float(row["g_after"])
        is_rec = bool(row["is_receiver"])

        if nudge < 0.3:
            continue

        g_gain      = g_aft - g_bef
        ghost_alpha = 0.55
        ghost_size  = 170 if is_rec else 100
        ghost_lw    = 2.0 if is_rec else 1.5
        ghost_rgba  = (1.0, 1.0, 1.0, ghost_alpha)
        edge_rgba   = (1.0, 1.0, 1.0, 0.9)

        # Dashed circle outline at ghost position
        radius = 1.8 if is_rec else 1.3
        circ = mpatches.Circle((nx, ny), radius,
                                fill=False, linestyle="--",
                                edgecolor=edge_rgba, linewidth=ghost_lw,
                                zorder=4)
        ax.add_patch(circ)

        # Solid ghost dot (low alpha)
        ax.scatter(nx, ny, s=ghost_size, c=[ghost_rgba], marker="o",
                   linewidths=ghost_lw, edgecolors=[edge_rgba],
                   zorder=4,
                   label="Ghost (optimal pos.)" if is_rec else None)

        # Nudge arrow: original → ghost
        if show_nudge_arrow:
            arrow_col = "#f1c40f" if is_rec else (1.0, 1.0, 1.0, 0.5)
            ax.annotate(
                "",
                xy=(nx, ny), xytext=(ox, oy),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=arrow_col,
                    lw=2.0 if is_rec else 1.0,
                    mutation_scale=12,
                ),
                zorder=4,
            )

        # ΔG label above ghost position
        if abs(g_gain) > 0.005:
            sign = "+" if g_gain >= 0 else ""
            ax.text(nx, ny + 2.8,
                    f"{sign}{g_gain:.3f}",
                    ha="center", va="bottom",
                    fontsize=7,
                    color="#f1c40f" if is_rec else "white",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="black", alpha=0.6,
                              edgecolor="none"),
                    zorder=5)


# ── Pass arrow ────────────────────────────────────────────────────────────────

def add_pass_arrow(ax, sx, sy, rx, ry, color="#ffffff", width=2,
                   label=None, hover=None):
    """
    Sender → receiver pass arrow.
    `hover` is ignored (no interactivity in matplotlib).
    `width` is Plotly pixel width; scaled to matplotlib linewidth.
    """
    lw = max(width * 0.75, 0.6)
    ax.annotate(
        "",
        xy=(rx, ry), xytext=(sx, sy),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=12,
        ),
        zorder=5,
    )
    if label:
        mx, my = (sx + rx) / 2, (sy + ry) / 2
        ax.text(mx, my, label,
                ha="center", va="center",
                fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="black", alpha=0.6,
                          edgecolor="none"),
                zorder=6)


# ── Legend helper ─────────────────────────────────────────────────────────────

def add_legend(ax, teams=None, extra_handles=None):
    """
    Add a dark legend to the axes for team colours and optional extras.
    teams: list of team name strings (must be in TEAM_COLORS)
    extra_handles: list of matplotlib legend handles to append
    """
    handles = []
    for team in (teams or list(TEAM_COLORS.keys())):
        color = TEAM_COLORS.get(team, "#888888")
        handles.append(mpatches.Patch(facecolor=color, edgecolor="white",
                                      linewidth=0.8, label=team))
    if extra_handles:
        handles.extend([h for h in extra_handles if h is not None])
    if handles:
        ax.legend(
            handles=handles,
            loc="upper right",
            framealpha=0.85,
            facecolor="#141414",
            edgecolor="white",
            labelcolor="white",
            fontsize=9,
        )
