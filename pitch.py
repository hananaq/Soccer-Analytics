"""
pitch.py — Plotly pitch drawing utilities for the PPN Football Dashboard.
Coordinates: origin at centre, x: -52.5 → +52.5, y: -34 → +34.
"""
import numpy as np
import plotly.graph_objects as go

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


# ── Shape helpers ─────────────────────────────────────────────────────────────

def _line(x0, y0, x1, y1, color=LINE_COLOR, width=1.8):
    return dict(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color=color, width=width))


def _circle_trace(cx, cy, r, color=LINE_COLOR, width=1.8, n=120, dash="solid"):
    t = np.linspace(0, 2 * np.pi, n)
    return go.Scatter(
        x=cx + r * np.cos(t), y=cy + r * np.sin(t),
        mode="lines", line=dict(color=color, width=width, dash=dash),
        showlegend=False, hoverinfo="skip"
    )


# ── Core pitch builder ────────────────────────────────────────────────────────

def make_pitch_fig(title="", height=520, show_legend=True, bg=PITCH_BG):
    shapes = [
        # Outline
        _line(-52.5, -34,  52.5, -34),
        _line(-52.5,  34,  52.5,  34),
        _line(-52.5, -34, -52.5,  34),
        _line( 52.5, -34,  52.5,  34),
        # Halfway
        _line(0, -34, 0, 34),
        # Left penalty box
        _line(-52.5, -20.16, -36, -20.16),
        _line(-52.5,  20.16, -36,  20.16),
        _line(-36,   -20.16, -36,  20.16),
        # Right penalty box
        _line( 52.5, -20.16,  36, -20.16),
        _line( 52.5,  20.16,  36,  20.16),
        _line(  36,  -20.16,  36,  20.16),
        # Left small box
        _line(-52.5, -4.58, -47, -4.58),
        _line(-52.5,  4.58, -47,  4.58),
        _line(-47,   -4.58, -47,  4.58),
        # Right small box
        _line( 52.5, -4.58,  47, -4.58),
        _line( 52.5,  4.58,  47,  4.58),
        _line(  47,  -4.58,  47,  4.58),
        # Left goal
        _line(-52.5, -3.66, -54.5, -3.66, color="#bbbbbb", width=1.2),
        _line(-52.5,  3.66, -54.5,  3.66, color="#bbbbbb", width=1.2),
        _line(-54.5,  -3.66, -54.5,  3.66, color="#bbbbbb", width=1.2),
        # Right goal
        _line( 52.5, -3.66,  54.5, -3.66, color="#bbbbbb", width=1.2),
        _line( 52.5,  3.66,  54.5,  3.66, color="#bbbbbb", width=1.2),
        _line(  54.5, -3.66,  54.5,  3.66, color="#bbbbbb", width=1.2),
        # Centre spot
        dict(type="circle", x0=-0.35, y0=-0.35, x1=0.35, y1=0.35,
             fillcolor=LINE_COLOR, line=dict(color=LINE_COLOR, width=0)),
        # Penalty spots
        dict(type="circle", x0=-41.2, y0=-0.3, x1=-40.6, y1=0.3,
             fillcolor=LINE_COLOR, line=dict(color=LINE_COLOR, width=0)),
        dict(type="circle", x0= 40.6, y0=-0.3, x1= 41.2, y1=0.3,
             fillcolor=LINE_COLOR, line=dict(color=LINE_COLOR, width=0)),
    ]

    fig = go.Figure()
    fig.add_trace(_circle_trace(0, 0, 9.15))

    # Penalty arcs
    t_l = np.linspace(np.radians(37), np.radians(143), 60)
    fig.add_trace(go.Scatter(x=-40.9 + 9.15*np.cos(t_l), y=9.15*np.sin(t_l),
                              mode="lines", line=dict(color=LINE_COLOR, width=1.8),
                              showlegend=False, hoverinfo="skip"))
    t_r = np.linspace(np.radians(180+37), np.radians(180+143), 60)
    fig.add_trace(go.Scatter(x=40.9 + 9.15*np.cos(t_r), y=9.15*np.sin(t_r),
                              mode="lines", line=dict(color=LINE_COLOR, width=1.8),
                              showlegend=False, hoverinfo="skip"))

    fig.update_layout(
        shapes=shapes,
        plot_bgcolor=bg,
        paper_bgcolor=PAPER_BG,
        font=dict(color="white"),
        title=dict(text=title, font=dict(color="white", size=12), x=0.5),
        xaxis=dict(range=[-56, 56], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-37, 37], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="x", scaleratio=1,
                   fixedrange=True),
        margin=dict(l=0, r=0, t=45, b=0),
        showlegend=show_legend,
        legend=dict(bgcolor="rgba(20,20,20,0.85)", bordercolor="white",
                    borderwidth=1, font=dict(color="white", size=10)),
        height=height,
    )
    return fig


# ── Player scatter ────────────────────────────────────────────────────────────

def add_players(fig, snap_df, sender_num, receiver_num, attacking_team,
                show_names=True):
    """
    Plot all players from a snapshot DataFrame.
    snap_df columns: team, player_number, player_name, x, y
    """
    for team, grp in snap_df.groupby("team"):
        color     = TEAM_COLORS.get(team, "#888888")
        opp_color = "#e74c3c" if team == "Auckland FC" else "#3498db"

        for _, p in grp.iterrows():
            pnum  = int(p["player_number"])
            pname = str(p.get("player_name", ""))
            x, y  = float(p["x"]), float(p["y"])

            is_sender   = (pnum == sender_num   and team == attacking_team)
            is_receiver = (pnum == receiver_num and team == attacking_team)

            symbol = "diamond"   if is_sender   else \
                     "circle"    if is_receiver  else "circle"
            size   = 22          if is_sender    else \
                     20          if is_receiver   else 14
            border = 3           if (is_sender or is_receiver) else 1.2
            border_col = "gold"  if is_sender else \
                         "white" if is_receiver else "white"

            short_name = pname.split()[-1] if pname else ""
            label      = f"{pnum}"
            hover_txt  = (f"<b>{pname}</b> (#{pnum})<br>{team}"
                          f"{'<br>⭐ Passer' if is_sender else ''}"
                          f"{'<br>🎯 Receiver' if is_receiver else ''}")

            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text" if show_names else "markers",
                marker=dict(size=size, color=color, symbol=symbol,
                            line=dict(color=border_col, width=border)),
                text=[label],
                textposition="middle center",
                textfont=dict(size=7, color="white", family="Arial Black"),
                name=team,
                legendgroup=team,
                showlegend=False,
                hovertemplate=hover_txt + "<extra></extra>",
                customdata=[[pname, pnum, team]],
            ))

            # Name label below player
            if show_names and short_name:
                fig.add_annotation(
                    x=x, y=y - 2.8,
                    text=f'<span style="font-size:8px">{short_name}</span>',
                    showarrow=False,
                    font=dict(color="white", size=8),
                    bgcolor="rgba(0,0,0,0.55)",
                    borderpad=1,
                )


def add_ball(fig, bx, by):
    """Add the ball marker to the pitch."""
    if bx is None or by is None:
        return
    fig.add_trace(go.Scatter(
        x=[bx], y=[by],
        mode="markers",
        marker=dict(size=13, color="white", symbol="circle",
                    line=dict(color="#333", width=1.5)),
        name="Ball",
        showlegend=True,
        hovertemplate="Ball<extra></extra>",
    ))
    fig.add_annotation(x=bx, y=by + 2.5, text="⚽",
                       showarrow=False, font=dict(size=12))


def add_ghost_players(fig, opos_df, show_nudge_arrow=True):
    """
    Overlay ghost (optimised) positions for off-ball players.
    opos_df: DataFrame with orig_x/y, opt_x/y, player_number,
             player_name, nudge_m, g_before, g_after, is_receiver
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

        if nudge < 0.3:          # negligible nudge — skip ghost
            continue

        g_gain = g_aft - g_bef
        ghost_color = "rgba(255,255,255,0.55)"
        ghost_line  = "rgba(255,255,255,0.9)"
        ghost_size  = 22 if is_rec else 14
        ghost_border= 2.5 if is_rec else 1.5

        # Ghost circle (dashed outline)
        fig.add_trace(_circle_trace(
            nx, ny, r=ghost_size * 0.18,
            color=ghost_line, width=ghost_border, dash="dash"
        ))

        # Ghost dot
        fig.add_trace(go.Scatter(
            x=[nx], y=[ny],
            mode="markers",
            marker=dict(
                size=ghost_size,
                color=ghost_color,
                symbol="circle",
                line=dict(color=ghost_line, width=ghost_border),
            ),
            name="Ghost (optimal)" if is_rec else "Ghost",
            legendgroup="ghost",
            showlegend=is_rec,
            hovertemplate=(
                f"<b>Ghost #{pnum}</b> ({pname})<br>"
                f"Nudge: {nudge:.1f} m<br>"
                f"G before: {g_bef:.4f}<br>"
                f"G after:  {g_aft:.4f}<br>"
                f"ΔG: {g_gain:+.4f}"
                + (" &nbsp;<b>← Receiver</b>" if is_rec else "")
                + "<extra></extra>"
            ),
        ))

        # Nudge arrow: original → ghost
        if show_nudge_arrow:
            arrow_col = "#f1c40f" if is_rec else "rgba(255,255,255,0.5)"
            fig.add_annotation(
                ax=ox, ay=oy, x=nx, y=ny,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=3, arrowsize=1.0,
                arrowwidth=2.5 if is_rec else 1.2,
                arrowcolor=arrow_col,
                text="",
            )

        # Label gain change on ghost
        if abs(g_gain) > 0.005:
            sign = "+" if g_gain >= 0 else ""
            fig.add_annotation(
                x=nx, y=ny + 2.8,
                text=f'<span style="font-size:8px">{sign}{g_gain:.3f}</span>',
                showarrow=False,
                font=dict(color="#f1c40f" if is_rec else "white", size=8),
                bgcolor="rgba(0,0,0,0.6)", borderpad=1,
            )


def add_pass_arrow(fig, sx, sy, rx, ry, color="#ffffff", width=2,
                   label=None, hover=None):
    """Sender → receiver pass arrow."""
    fig.add_annotation(
        ax=sx, ay=sy, x=rx, y=ry,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True,
        arrowhead=2, arrowsize=1.2, arrowwidth=width,
        arrowcolor=color, text=label or "",
        font=dict(color="white", size=9),
    )
    if hover:
        fig.add_trace(go.Scatter(
            x=[(sx + rx) / 2], y=[(sy + ry) / 2],
            mode="markers",
            marker=dict(size=10, color=color, opacity=0.01),
            hovertemplate=hover + "<extra></extra>",
            showlegend=False,
        ))
