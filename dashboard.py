"""
Pass Decision Analytics Dashboard
Auckland FC vs Newcastle United Jets — A-League 2024/25
Run: streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from pitch import (make_pitch_fig, add_players, add_ball, add_ghost_players,
                   add_pass_arrow, add_legend, TEAM_COLORS, ERROR_COLORS,
                   PITCH_BG, PAPER_BG)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pass Decision Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container { padding-top: 3.5rem; }

  /* ── Metric cards (light style) ── */
  .metric-card {
      background:#ffffff; border-radius:10px; padding:14px 18px;
      border-left:4px solid #2563eb; margin-bottom:8px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  }
  .metric-label { color:#64748b; font-size:12px; margin-bottom:3px; }
  .metric-value { color:#1a1a2e; font-size:26px; font-weight:700; }
  .metric-sub   { color:#94a3b8; font-size:11px; margin-top:2px; }
  .badge-best  { background:#dcfce7; color:#166534; border-radius:6px;
                 padding:2px 8px; font-size:12px; font-weight:600; }
  .badge-worst { background:#fee2e2; color:#991b1b; border-radius:6px;
                 padding:2px 8px; font-size:12px; font-weight:600; }

  /* ── Sidebar: always dark with white text ── */
  section[data-testid="stSidebar"] { background:#0d0d1a !important; }
  section[data-testid="stSidebar"] * { color:#f0f0f0 !important; }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stMultiSelect label,
  section[data-testid="stSidebar"] .stCaption,
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] div { color:#f0f0f0 !important; }
  /* Selectbox + multiselect control background */
  section[data-testid="stSidebar"] div[data-baseweb="select"] {
      background:#1e2a4a !important;
  }
  section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
      background:#1e2a4a !important; border-color:#334155 !important;
  }
  /* Multiselect tags (selected pills) */
  section[data-testid="stSidebar"] div[data-baseweb="tag"] {
      background:#2563eb !important; color:#ffffff !important;
  }
  /* X button and arrow icon inside multiselect/select */
  section[data-testid="stSidebar"] div[data-baseweb="tag"] span,
  section[data-testid="stSidebar"] div[data-baseweb="tag"] svg {
      color:#ffffff !important; fill:#ffffff !important;
  }
  section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
      fill:#f0f0f0 !important;
  }
  /* Input text inside widgets */
  section[data-testid="stSidebar"] input {
      background:#1e2a4a !important; color:#f0f0f0 !important;
  }
  section[data-testid="stSidebar"] hr { border-color:#334155 !important; }
  section[data-testid="stSidebar"] .streamlit-expanderHeader {
      font-size:12px; padding:4px 8px; color:#f0f0f0 !important;
  }
  section[data-testid="stSidebar"] .streamlit-expanderContent p {
      font-size:11px; color:#cbd5e1 !important; margin:0;
  }
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path("dashboard_data")

# ── Chart theme (light background, dark text) ─────────────────────────────────
CHART_BG   = "#ffffff"
CHART_PLOT = "#f8fafc"
CHART_FONT = "#1a1a2e"
CHART_GRID = "#e2e8f0"

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    gain    = pd.read_csv(DATA_DIR / "gain.csv")
    ppn     = pd.read_csv(DATA_DIR / "ppn.csv")
    pos     = pd.read_csv(DATA_DIR / "pass_positions.csv")
    snaps   = pd.read_csv(DATA_DIR / "player_snapshots.csv")
    options = pd.read_csv(DATA_DIR / "ppn_options.csv")
    names   = pd.read_csv(DATA_DIR / "player_names.csv")

    opt_agg  = pd.DataFrame()
    opt_pos  = pd.DataFrame()
    if (DATA_DIR / "opt.csv").exists():
        opt_agg = pd.read_csv(DATA_DIR / "opt.csv")
    if (DATA_DIR / "opt_positions.csv").exists():
        opt_pos = pd.read_csv(DATA_DIR / "opt_positions.csv")

    # ppn already contains P_t, V_t, G_t, error_type, team, success from the notebook merge.
    # Only bring spatial columns from pos to avoid duplicate-column conflicts.
    pos_cols = ["pass_id","sender_number","receiver_number",
                "sx","sy","rx","ry","ball_x","ball_y","start_frame"]
    merged = ppn.merge(pos[pos_cols], on="pass_id", how="left")

    f_min = merged["start_frame"].min()
    f_max = merged["start_frame"].max()
    def phase(f):
        p = (f - f_min) / (f_max - f_min + 1e-9)
        if p < 1/3: return "1st Third (0–30 min)"
        if p < 2/3: return "2nd Third (30–60 min)"
        return "3rd Third (60–90 min)"
    merged["match_phase"] = merged["start_frame"].apply(phase)

    # player label: "Name (#N)"
    names["label"] = names["player_name"].str.split().str[-1] + \
                     " (#" + names["player_number"].astype(str) + ")"
    names["full_label"] = names["player_name"] + \
                          " (#" + names["player_number"].astype(str) + ")"
    return gain, ppn, pos, snaps, options, names, opt_agg, opt_pos, merged

try:
    gain, ppn, pos, snaps, options, names, opt_agg, opt_pos, merged = load_data()
    data_ok = True
except FileNotFoundError as e:
    data_ok = False
    _missing = str(e)

# ── Helpers ───────────────────────────────────────────────────────────────────
def metric_card(col, label, value, sub=""):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def player_label(team, number):
    """Return 'Surname (#N)' for a player."""
    row = names[(names.team == team) & (names.player_number == number)]
    if row.empty:
        return f"#{number}"
    return row.iloc[0]["full_label"]

PHASES = ["1st Third (0–30 min)", "2nd Third (30–60 min)", "3rd Third (60–90 min)"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ Pass Decision Analytics")
    st.markdown("**Auckland FC vs Newcastle Jets**  \nA-League 2024/25")
    st.markdown("""
<div style='background:#0f2027;border:1px solid #334155;border-radius:8px;
            padding:10px 14px;margin:6px 0;text-align:center;'>
  <span style='color:#f0f0f0;font-size:13px;font-weight:600;'>Auckland FC</span>
  &nbsp;
  <span style='background:#166534;color:#4ade80;font-size:20px;font-weight:900;
               padding:4px 14px;border-radius:6px;letter-spacing:2px;'>2 – 0</span>
  &nbsp;
  <span style='color:#94a3b8;font-size:13px;font-weight:600;'>Newcastle Jets</span>
  <div style='color:#64748b;font-size:11px;margin-top:6px;'>🏆 Auckland FC — Winner</div>
</div>
""", unsafe_allow_html=True)
    st.caption("29 November 2024")
    st.divider()

    if data_ok:
        team_filter    = st.selectbox("Team", ["Both", "Auckland FC", "Newcastle"])
        outcome_filter = st.multiselect("Outcome",
                                        ["Successful", "Unsuccessful"],
                                        default=["Successful", "Unsuccessful"])
        phase_filter   = st.multiselect("Match Phase", PHASES, default=PHASES)
        st.divider()
        st.caption("Re-run the export cell in the notebook to refresh data.")

    # ── Glossary ──────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📖 Glossary")

    GLOSSARY = [
        ("P(t)",               "Pass success probability — output of the Logistic Regression model trained on distance, pressure, obstruction, and receiver centrality."),
        ("V(t)",               "Space value at the receiver's position — a blended score of how much territory the attacker controls and how dangerous that location is."),
        ("G(t)",               "Gain function — geometric mean √(P(t) × V(t)). Rewards passes that are both probable and positionally valuable."),
        ("DOS",                "Decision Optimality Score — G(chosen) ÷ G(best option). 1.0 = optimal choice, 0.0 = worst possible."),
        ("Opportunity Cost",   "G(best option) − G(chosen). The gain left unrealised by not picking the best available pass."),
        ("PPN",                "Potential Pass Network — G(t) evaluated for every available teammate simultaneously, revealing the full decision landscape."),
        ("Passer Error",       "Pass failed despite G(t) ≥ threshold — the decision was correct but execution failed."),
        ("Structure Error — Overambitious", "Pass failed with V(t) > P(t) — the space was valuable but the pass was too risky."),
        ("Structure Error — Trapped",       "Pass failed with P(t) ≥ V(t) — the pass was safe but into worthless space."),
        ("Ghost player",       "Optimised position for the receiver (or best option) after gradient ascent — where the player should have been to maximise G(t)."),
        ("Nudge (m)",          "Distance in metres between a player's actual position and their gradient-ascent optimised position (max 3 m)."),
        ("Formation Gain",     "Improvement in total team G(t) after repositioning all off-ball teammates via gradient ascent."),
    ]

    for term, definition in GLOSSARY:
        with st.expander(f"**{term}**"):
            st.markdown(f"<small>{definition}</small>", unsafe_allow_html=True)

if not data_ok:
    st.error(f"⚠️  Data not found ({_missing}).  "
             "Run the **Export** cell in the notebook first, then copy "
             "`dashboard_data/` next to `dashboard.py`.")
    st.stop()

def apply_filters(df):
    out = df.copy()
    if team_filter != "Both":
        out = out[out["team"] == team_filter]
    omap = {"Successful": 1, "Unsuccessful": 0}
    keep = [omap[o] for o in outcome_filter if o in omap]
    out  = out[out["success"].isin(keep)]
    out  = out[out["match_phase"].isin(phase_filter)]
    return out


# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Match Overview",
    "🎯 Pass Explorer",
    "🧠 Decision Quality",
    "❌ Error Analysis",
])

# ╔═══════════════════════════════════════════════════════════════════════════
# TAB 1 — MATCH OVERVIEW
# ╚═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Match Overview")
    filt = apply_filters(merged)

    c1, c2, c3, c4, c5 = st.columns(5)
    metric_card(c1, "Total Passes",    len(filt))
    metric_card(c2, "Success Rate",    f"{filt['success'].mean()*100:.1f}%")
    metric_card(c3, "Avg G(t)",        f"{filt['G_t'].mean():.3f}",    "Gain function")
    metric_card(c4, "Avg DOS",         f"{filt['DOS'].mean():.3f}",    "Decision quality")
    metric_card(c5, "Avg Opp. Cost",   f"{filt['opportunity_cost'].mean():.3f}", "Unrealised gain")

    st.divider()

    # ── Best / Worst match third ──────────────────────────────────────────────
    st.markdown("#### Best & Worst Match Third")
    phase_stats = (filt.groupby("match_phase")
                   .agg(n=("pass_id","count"),
                        avg_DOS=("DOS","mean"),
                        avg_G=("G_t","mean"),
                        avg_opp=("opportunity_cost","mean"),
                        fail_rate=("success", lambda x: (x==0).mean()))
                   .reindex(PHASES).reset_index())

    best_third  = phase_stats.loc[phase_stats["avg_DOS"].idxmax(), "match_phase"]
    worst_third = phase_stats.loc[phase_stats["avg_DOS"].idxmin(), "match_phase"]

    bw1, bw2 = st.columns(2)
    best_dos  = phase_stats.loc[phase_stats.match_phase == best_third,  "avg_DOS"].values[0]
    worst_dos = phase_stats.loc[phase_stats.match_phase == worst_third, "avg_DOS"].values[0]
    bw1.markdown(f"""
<div style='background:#dcfce7;border-left:5px solid #16a34a;border-radius:8px;
            padding:14px 18px;min-height:80px;'>
  <div style='color:#166534;font-size:12px;font-weight:700;margin-bottom:6px;'>✅ BEST THIRD</div>
  <div style='color:#14532d;font-size:17px;font-weight:800;'>{best_third}</div>
  <div style='color:#166534;font-size:13px;margin-top:4px;'>Avg DOS &nbsp;<b>{best_dos:.3f}</b></div>
</div>""", unsafe_allow_html=True)
    bw2.markdown(f"""
<div style='background:#fee2e2;border-left:5px solid #dc2626;border-radius:8px;
            padding:14px 18px;min-height:80px;'>
  <div style='color:#991b1b;font-size:12px;font-weight:700;margin-bottom:6px;'>❌ WORST THIRD</div>
  <div style='color:#7f1d1d;font-size:17px;font-weight:800;'>{worst_third}</div>
  <div style='color:#991b1b;font-size:13px;margin-top:4px;'>Avg DOS &nbsp;<b>{worst_dos:.3f}</b></div>
</div>""", unsafe_allow_html=True)

    # Phase bar charts
    col_ph1, col_ph2 = st.columns(2)
    with col_ph1:
        fig_ph = go.Figure()
        for col_name, label, color in [
            ("avg_DOS",  "Avg DOS",    "#3498db"),
            ("avg_G",    "Avg G(t)",   "#2ecc71"),
            ("fail_rate","Fail Rate",  "#e74c3c"),
        ]:
            fig_ph.add_trace(go.Bar(
                x=phase_stats["match_phase"].str.replace(" min)",")",regex=False),
                y=phase_stats[col_name],
                name=label, marker_color=color,
                text=phase_stats[col_name].round(3), textposition="outside",
            ))
        fig_ph.update_layout(
            title="Match Phase — Key Metrics", barmode="group",
            paper_bgcolor=CHART_BG, plot_bgcolor=CHART_PLOT,
            font=dict(color=CHART_FONT), height=420,
            yaxis_title="Value",
            yaxis=dict(range=[0, phase_stats[["avg_DOS","avg_G","fail_rate"]].max().max() * 1.25], gridcolor=CHART_GRID),
        )
        st.plotly_chart(fig_ph, use_container_width=True)

    with col_ph2:
        fig_opp_ph = px.bar(
            phase_stats,
            x="match_phase", y="avg_opp",
            color="match_phase", text=phase_stats["avg_opp"].round(4),
            title="Avg Opportunity Cost by Phase (gain left unrealised)",
            labels={"avg_opp":"Avg Opp. Cost","match_phase":"Phase"},
            color_discrete_sequence=["#9b59b6","#e67e22","#e74c3c"],
        )
        fig_opp_ph.update_traces(textposition="outside")
        fig_opp_ph.update_layout(
            paper_bgcolor=CHART_BG, plot_bgcolor=CHART_PLOT,
            font=dict(color=CHART_FONT), showlegend=False, height=420,
            yaxis=dict(range=[0, phase_stats["avg_opp"].max() * 1.25], gridcolor=CHART_GRID),
        )
        st.plotly_chart(fig_opp_ph, use_container_width=True)

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        fig_gt = px.histogram(
            filt, x="G_t", color="team",
            color_discrete_map=TEAM_COLORS,
            barmode="overlay", nbins=30, opacity=0.75,
            title="G(t) Distribution by Team",
            labels={"G_t":"Gain G(t)","team":"Team"},
        )
        fig_gt.add_vline(x=filt["G_t"].median(), line_dash="dash",
                         line_color="yellow",
                         annotation_text="median", annotation_font_color="yellow")
        fig_gt.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_PLOT,
                              font=dict(color=CHART_FONT))
        st.plotly_chart(fig_gt, use_container_width=True)

    with col_r:
        fig_pv = px.scatter(
            filt, x="P_t", y="V_t", color="success",
            color_discrete_map={1:"#2ecc71", 0:"#e74c3c"},
            opacity=0.6, title="P(t) vs V(t)",
            labels={"P_t":"P(t) — success prob.","V_t":"V(t) — space value"},
            hover_data=["pass_id","G_t","team"],
        )
        fig_pv.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_PLOT,
                              font=dict(color=CHART_FONT))
        st.plotly_chart(fig_pv, use_container_width=True)

    st.divider()
    st.markdown("#### Team Summary")
    ts = (filt.groupby("team")
          .agg(passes=("pass_id","count"),
               success_rt=("success","mean"),
               avg_G=("G_t","mean"),
               avg_DOS=("DOS","mean"),
               avg_opp=("opportunity_cost","mean"))
          .reset_index()
          .rename(columns={"team":"Team","passes":"Passes",
                            "success_rt":"Success Rate","avg_G":"Avg G(t)",
                            "avg_DOS":"Avg DOS","avg_opp":"Avg Opp. Cost"}))
    ts["Success Rate"] = ts["Success Rate"].map("{:.1%}".format)
    for c in ["Avg G(t)","Avg DOS","Avg Opp. Cost"]:
        ts[c] = ts[c].map("{:.4f}".format)
    st.dataframe(ts, use_container_width=True, hide_index=True)


# ╔═══════════════════════════════════════════════════════════════════════════
# TAB 2 — PASS EXPLORER
# ╚═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Pass Explorer")
    st.caption("Select a pass to see its Potential Pass Network on the pitch.")

    filt2 = apply_filters(merged)
    col_sel, col_detail = st.columns([1, 2])

    with col_sel:
        err_opts = ["All"] + sorted(filt2["error_type"].dropna().unique().tolist())
        err_sel  = st.selectbox("Error type", err_opts, key="pe_err")
        if err_sel != "All":
            filt2 = filt2[filt2["error_type"] == err_sel]

        show_df = (filt2[["pass_id","team","success","G_t","DOS",
                           "opportunity_cost","error_type"]]
                   .copy()
                   .assign(success=lambda d: d.success.map({1:"✓",0:"✗"}))
                   .rename(columns={"pass_id":"Pass","team":"Team","success":"OK",
                                    "G_t":"G(t)","opportunity_cost":"Opp Cost",
                                    "error_type":"Error Type"})
                   .round(4))

        ev_sel = st.dataframe(show_df, use_container_width=True,
                              hide_index=True, on_select="rerun",
                              selection_mode="single-row", height=440)

    with col_detail:
        rows = ev_sel.selection.rows if hasattr(ev_sel,"selection") else []
        if rows:
            idx     = rows[0]
            pass_id = int(filt2.iloc[idx]["pass_id"])
            row     = merged[merged.pass_id == pass_id].iloc[0]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("P(t)", f"{row['P_t']:.3f}")
            m2.metric("V(t)", f"{row['V_t']:.3f}")
            m3.metric("G(t)", f"{row['G_t']:.3f}")
            m4.metric("DOS",  f"{row['DOS']:.3f}")

            et     = row["error_type"]
            cmap   = {"Success":"green","Passer Error":"red",
                      "Structure Error — Overambitious":"orange",
                      "Structure Error — Trapped":"violet"}
            snum   = int(row["sender_number"]) if pd.notna(row["sender_number"]) else None
            sender_lbl = player_label(row["team"], snum) if snum else "?"
            st.markdown(
                f"**Team:** {row['team']}  |  "
                f"**Passer:** {sender_lbl}  |  "
                f"**Outcome:** {'✅' if row['success']==1 else '❌'}  |  "
                f"**Type:** :{cmap.get(et,'white')}[{et}]"
            )

            pass_opts = options[options.pass_id == pass_id]
            snap      = snaps[snaps.pass_id == pass_id]

            if not pass_opts.empty and not snap.empty:
                g_max = pass_opts["G_t"].max()
                g_min = pass_opts["G_t"].min()

                fig_p, ax_p = make_pitch_fig(
                    title=f"PPN — Pass {pass_id}  |  "
                          f"Passer: {sender_lbl}  |  DOS: {row['DOS']:.3f}",
                    height=480)

                rnum = int(row["receiver_number"]) if pd.notna(row["receiver_number"]) else None
                add_players(ax_p, snap, snum, rnum, row["team"])
                add_ball(ax_p, row.get("ball_x"), row.get("ball_y"))

                optimal_handle = None
                for _, opt_row in pass_opts.iterrows():
                    g_norm = (opt_row["G_t"] - g_min) / (g_max - g_min + 1e-9)
                    # Matplotlib-compatible colour: red→green gradient
                    clr = (
                        (255*(1-g_norm))/255,
                        (200*g_norm + 55)/255,
                        60/255,
                    )
                    sx, sy = row["sx"], row["sy"]
                    rx, ry = opt_row["rx"], opt_row["ry"]
                    if any(pd.isna([sx, sy, rx, ry])): continue
                    is_chosen  = bool(opt_row["is_chosen"])
                    is_optimal = (opt_row["G_t"] == g_max)
                    add_pass_arrow(ax_p, sx, sy, rx, ry,
                                   color=clr, width=1.5 + 3.5*g_norm)
                    if is_chosen:
                        # White hollow ring around chosen receiver
                        ax_p.scatter(rx, ry, s=420, facecolors="none",
                                     edgecolors="white", linewidths=2.5, zorder=8)
                    if is_optimal:
                        ax_p.scatter(rx, ry, s=180, c="gold", marker="*",
                                     edgecolors="black", linewidths=0.8, zorder=9)
                        optimal_handle = Line2D(
                            [0], [0], marker="*", color="w",
                            markerfacecolor="gold", markersize=10,
                            label="Optimal ★", linestyle="None")

                add_legend(ax_p, extra_handles=[optimal_handle])
                st.pyplot(fig_p, use_container_width=True)
                plt.close(fig_p)
                st.caption("★ Gold star = optimal receiver · White ring = chosen · "
                           "Arrow width/colour = G(t)")

                with st.expander("All options"):
                    disp = pass_opts[["receiver_number","player_name","distance",
                                      "P_t","V_t","G_t","is_chosen"]].copy()
                    disp["is_chosen"] = disp["is_chosen"].map({True:"✓",False:""})
                    disp = disp.sort_values("G_t", ascending=False).reset_index(drop=True)
                    st.dataframe(disp, use_container_width=True, hide_index=True)
        else:
            st.info("← Select a pass from the table to explore it.")


# ╔═══════════════════════════════════════════════════════════════════════════
# TAB 3 — GHOST VIEW  (hidden — tab removed from UI)
# ╚═══════════════════════════════════════════════════════════════════════════
if False:  # noqa — kept for future re-enable
 if False:
    st.subheader("👻 Ghost Player View")
    st.caption(
        "The **ghost** (transparent white) shows where each player **should have been** "
        "to maximise the team's total formation gain — derived from the gradient ascent optimiser.  \n"
        "**Yellow arrows** = receiver's nudge · White arrows = other players · "
        "Gold ΔG label = gain change per player."
    )

    if opt_pos.empty:
        st.warning("No ghost data yet. Run the **Export** cell in the notebook "
                   "(it re-runs gradient ascent and saves `opt_positions.csv`).")
    else:
        filt3g = apply_filters(merged)

        col_gsel, col_gmap = st.columns([1, 2])
        with col_gsel:
            # Filter to passes where optimisation ran
            avail_ids = set(opt_pos["pass_id"].unique())
            filt3g    = filt3g[filt3g["pass_id"].isin(avail_ids)]

            err_opts3 = ["All"] + sorted(filt3g["error_type"].dropna().unique().tolist())
            err_sel3  = st.selectbox("Error type", err_opts3, key="ghost_err")
            if err_sel3 != "All":
                filt3g = filt3g[filt3g["error_type"] == err_sel3]

            sort_by = st.radio("Sort by", ["Highest % Gain", "Lowest DOS", "Pass ID"],
                               horizontal=True)
            if not opt_agg.empty and "pct_improvement" in opt_agg.columns:
                filt3g = filt3g.merge(
                    opt_agg[["pass_id","pct_improvement","gain_before","gain_after"]],
                    on="pass_id", how="left")
                if sort_by == "Highest % Gain":
                    filt3g = filt3g.sort_values("pct_improvement", ascending=False)
                elif sort_by == "Lowest DOS":
                    filt3g = filt3g.sort_values("DOS")

            show_ghost = filt3g[["pass_id","team","success","G_t","DOS","error_type"]
                                 + (["pct_improvement"] if "pct_improvement" in filt3g else [])
                                ].copy()
            show_ghost["success"] = show_ghost["success"].map({1:"✓",0:"✗"})
            if "pct_improvement" in show_ghost:
                show_ghost["pct_improvement"] = show_ghost["pct_improvement"].round(2)
            show_ghost = show_ghost.rename(columns={
                "pass_id":"Pass","team":"Team","success":"OK",
                "G_t":"G(t)","error_type":"Error",
                "pct_improvement":"% Gain"}).round(4)

            g_event = st.dataframe(show_ghost, use_container_width=True,
                                   hide_index=True, on_select="rerun",
                                   selection_mode="single-row", height=480)

        with col_gmap:
            ghost_rows = g_event.selection.rows if hasattr(g_event,"selection") else []
            if ghost_rows:
                idx      = ghost_rows[0]
                pass_id  = int(filt3g.iloc[idx]["pass_id"])
                row      = merged[merged.pass_id == pass_id].iloc[0]
                opos     = opt_pos[opt_pos.pass_id == pass_id]
                snap     = snaps[snaps.pass_id == pass_id]

                snum = int(row["sender_number"]) if pd.notna(row["sender_number"]) else None
                rnum = int(row["receiver_number"]) if pd.notna(row["receiver_number"]) else None

                pct_imp = ""
                if "pct_improvement" in filt3g.columns:
                    v = filt3g.iloc[idx].get("pct_improvement", None)
                    if v is not None and not pd.isna(v):
                        pct_imp = f"  |  Formation gain: +{v:.1f}%"

                fig_g, ax_g = make_pitch_fig(
                    title=(f"Ghost View — Pass {pass_id}  |  "
                           f"{row['team']}  |  "
                           f"Passer: {player_label(row['team'], snum) if snum else '?'}"
                           f"{pct_imp}"),
                    height=520)

                # Real players
                add_players(ax_g, snap, snum, rnum, row["team"])

                # Ball
                add_ball(ax_g, row.get("ball_x"), row.get("ball_y"))

                # Actual pass arrow
                if all(not pd.isna(v) for v in [row["sx"], row["sy"],
                                                  row["rx"], row["ry"]]):
                    add_pass_arrow(ax_g,
                                   float(row["sx"]), float(row["sy"]),
                                   float(row["rx"]), float(row["ry"]),
                                   color="white", width=2.5)

                # Ghost players — only show actual receiver + best PPN option
                pass_opts_g = options[options.pass_id == pass_id] if not options.empty else pd.DataFrame()
                best_opt_num = None
                if not pass_opts_g.empty:
                    best_opt_num = int(pass_opts_g.loc[pass_opts_g["G_t"].idxmax(), "receiver_number"])

                ghost_nums = set(filter(None, [rnum, best_opt_num]))
                opos_filtered = opos[opos["player_number"].isin(ghost_nums)] if not opos.empty else opos

                show_nudge = st.checkbox("Show nudge arrows", value=True,
                                         key="ghost_arrows")
                if not opos_filtered.empty:
                    add_ghost_players(ax_g, opos_filtered,
                                      show_nudge_arrow=show_nudge)

                # Legend with ghost handle
                ghost_handle = mpatches.Patch(
                    facecolor=(1.0, 1.0, 1.0, 0.5),
                    edgecolor="white", linewidth=1.5,
                    label="Ghost (optimal pos.)")
                add_legend(ax_g, extra_handles=[ghost_handle])
                st.pyplot(fig_g, use_container_width=True)
                plt.close(fig_g)

                # Per-player breakdown table — only the two ghost players
                if not opos_filtered.empty:
                    st.markdown("**Ghost player breakdown**")
                    disp_o = opos_filtered[["player_number","player_name","nudge_m",
                                            "g_before","g_after","is_receiver"]].copy()
                    disp_o["ΔG"] = (disp_o["g_after"]-disp_o["g_before"]).round(4)
                    disp_o["Role"] = disp_o.apply(
                        lambda r: "Chosen receiver" if r["is_receiver"]
                                  else "Best option", axis=1)
                    disp_o = (disp_o.sort_values("ΔG", ascending=False)
                              .rename(columns={"player_number":"#",
                                               "player_name":"Name",
                                               "nudge_m":"Nudge (m)",
                                               "g_before":"G before",
                                               "g_after":"G after",
                                               "is_receiver":"Receiver?"})
                              .drop(columns=["Receiver?"])
                              .reset_index(drop=True))
                    st.dataframe(disp_o.round(4), use_container_width=True,
                                 hide_index=True)
            else:
                st.info("← Select a pass from the table to see the ghost player view.")


# ╔═══════════════════════════════════════════════════════════════════════════
# TAB 3 — DECISION QUALITY
# ╚═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Decision Quality")
    filt4 = apply_filters(merged)

    col1, col2 = st.columns(2)

    with col1:
        fig_v = go.Figure()
        for team, color in TEAM_COLORS.items():
            sub = filt4[filt4["team"] == team]
            fig_v.add_trace(go.Violin(
                x=sub["success"].map({1:"Successful",0:"Unsuccessful"}),
                y=sub["DOS"], name=team,
                fillcolor=color, line_color=color,
                opacity=0.7, box_visible=True, meanline_visible=True,
            ))
        fig_v.update_layout(
            title="DOS Distribution by Team & Outcome",
            paper_bgcolor=CHART_BG, plot_bgcolor=CHART_PLOT,
            font=dict(color=CHART_FONT), violinmode="group",
            yaxis_title="Decision Optimality Score",
        )
        st.plotly_chart(fig_v, use_container_width=True)

    with col2:
        ph_s = (filt4.groupby("match_phase")
                .agg(avg_DOS=("DOS","mean"), avg_G=("G_t","mean"),
                     fail_rate=("success", lambda x: (x==0).mean()))
                .reindex(PHASES).reset_index())
        fig_ph2 = go.Figure()
        for col_n, lbl, clr in [("avg_DOS","Avg DOS","#3498db"),
                                  ("avg_G","Avg G(t)","#2ecc71")]:
            fig_ph2.add_trace(go.Bar(
                x=ph_s["match_phase"].str.replace(" min)",")",regex=False),
                y=ph_s[col_n], name=lbl, marker_color=clr,
                text=ph_s[col_n].round(3), textposition="outside"))
        fig_ph2.update_layout(
            title="Match Phase Analysis", barmode="group",
            paper_bgcolor=CHART_BG, plot_bgcolor=CHART_PLOT,
            font=dict(color=CHART_FONT), yaxis_title="Value")
        st.plotly_chart(fig_ph2, use_container_width=True)

    st.divider()

    # ── Top & Worst players ───────────────────────────────────────────────────
    st.markdown("#### Player Decision Quality (min 5 passes)")

    player_stats = (filt4.dropna(subset=["sender_number"])
                    .groupby(["team","sender_number"])
                    .agg(n_passes=("pass_id","count"),
                         avg_DOS=("DOS","mean"),
                         avg_G=("G_t","mean"),
                         avg_opp=("opportunity_cost","mean"),
                         success_rt=("success","mean"))
                    .reset_index()
                    .query("n_passes >= 5")
                    .copy())

    # Join names
    player_stats = player_stats.merge(
        names[["team","player_number","full_label"]],
        left_on=["team","sender_number"],
        right_on=["team","player_number"], how="left")
    player_stats["full_label"] = player_stats["full_label"].fillna(
        "#" + player_stats["sender_number"].astype(int).astype(str))

    top5  = player_stats.nlargest(5,  "avg_DOS")
    bot5  = player_stats.nsmallest(5, "avg_DOS")

    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown('<span class="badge-best">🏆 Top 5 Decision Makers</span>',
                    unsafe_allow_html=True)
        fig_top = go.Figure(go.Bar(
            x=top5["avg_DOS"], y=top5["full_label"],
            orientation="h", marker_color="#2ecc71",
            text=top5["avg_DOS"].round(3), textposition="outside",
            customdata=np.stack([top5["n_passes"],
                                  top5["success_rt"].round(2),
                                  top5["team"]], axis=-1),
            hovertemplate="<b>%{y}</b><br>DOS: %{x:.3f}<br>"
                          "Team: %{customdata[2]}<br>"
                          "Passes: %{customdata[0]}<br>"
                          "Success: %{customdata[1]:.0%}<extra></extra>",
        ))
        fig_top.update_layout(
            paper_bgcolor=CHART_BG, plot_bgcolor=CHART_PLOT,
            font=dict(color=CHART_FONT), xaxis=dict(range=[0,1.1], gridcolor=CHART_GRID),
            xaxis_title="Avg DOS", yaxis_title="", height=420,
            margin=dict(l=0,r=40,t=10,b=0))
        st.plotly_chart(fig_top, use_container_width=True)

    with bc2:
        st.markdown('<span class="badge-worst">⚠️ Bottom 5 Decision Makers</span>',
                    unsafe_allow_html=True)
        fig_bot = go.Figure(go.Bar(
            x=bot5["avg_DOS"], y=bot5["full_label"],
            orientation="h", marker_color="#e74c3c",
            text=bot5["avg_DOS"].round(3), textposition="outside",
            customdata=np.stack([bot5["n_passes"],
                                  bot5["success_rt"].round(2),
                                  bot5["team"]], axis=-1),
            hovertemplate="<b>%{y}</b><br>DOS: %{x:.3f}<br>"
                          "Team: %{customdata[2]}<br>"
                          "Passes: %{customdata[0]}<br>"
                          "Success: %{customdata[1]:.0%}<extra></extra>",
        ))
        fig_bot.update_layout(
            paper_bgcolor=CHART_BG, plot_bgcolor=CHART_PLOT,
            font=dict(color=CHART_FONT), xaxis=dict(range=[0,1.1], gridcolor=CHART_GRID),
            xaxis_title="Avg DOS", yaxis_title="", height=420,
            margin=dict(l=0,r=40,t=10,b=0))
        st.plotly_chart(fig_bot, use_container_width=True)

    st.divider()
    st.markdown("#### Full Player Rankings")
    col_pA, col_pB = st.columns(2)
    for col_p, team in zip([col_pA, col_pB], ["Auckland FC", "Newcastle"]):
        with col_p:
            td = player_stats[player_stats["team"]==team].sort_values("avg_DOS")
            if td.empty:
                st.info(f"No data for {team}"); continue
            colors = ["#2ecc71" if v>=0.75 else "#e67e22" if v>=0.60 else "#e74c3c"
                      for v in td["avg_DOS"]]
            fig_pl = go.Figure(go.Bar(
                x=td["avg_DOS"], y=td["full_label"],
                orientation="h", marker_color=colors,
                text=td["avg_DOS"].round(3), textposition="outside",
                customdata=np.stack([td["n_passes"],
                                      td["success_rt"].round(2)], axis=-1),
                hovertemplate="<b>%{y}</b><br>DOS: %{x:.3f}<br>"
                              "Passes: %{customdata[0]}<br>"
                              "Success: %{customdata[1]:.0%}<extra></extra>",
            ))
            fig_pl.add_vline(x=0.75, line_dash="dash", line_color="#2ecc71",
                              annotation_text="Good", annotation_font_color="#2ecc71")
            fig_pl.add_vline(x=0.60, line_dash="dash", line_color="#e67e22",
                              annotation_text="Mod.", annotation_font_color="#e67e22")
            fig_pl.update_layout(
                title=team, paper_bgcolor=CHART_BG, plot_bgcolor=CHART_PLOT,
                font=dict(color=CHART_FONT), xaxis=dict(range=[0,1.15], gridcolor=CHART_GRID),
                xaxis_title="Avg DOS", height=420)
            st.plotly_chart(fig_pl, use_container_width=True)


# ╔═══════════════════════════════════════════════════════════════════════════
# TAB 4 — ERROR ANALYSIS
# ╚═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Error Analysis")
    filt5 = apply_filters(merged)
    fails = filt5[filt5["success"]==0].dropna(subset=["error_type"])

    short_map = {
        "Passer Error":                    "Passer Error",
        "Structure Error — Overambitious": "Struct. Overambitious",
        "Structure Error — Trapped":       "Struct. Trapped",
    }

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        ec = (fails.groupby(["team","error_type"])
              .size().reset_index(name="count"))
        ec["error_short"] = ec["error_type"].map(short_map)
        fig_ec = px.bar(ec, x="error_short", y="count", color="team",
                        color_discrete_map=TEAM_COLORS,
                        barmode="group", text="count",
                        title="Failed Pass Error Types by Team",
                        labels={"error_short":"Error Type","count":"Count"})
        fig_ec.update_traces(textposition="outside")
        fig_ec.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_PLOT,
                              font=dict(color=CHART_FONT))
        st.plotly_chart(fig_ec, use_container_width=True)

    with col_e2:
        fig_de = go.Figure()
        for etype, color in ERROR_COLORS.items():
            if etype == "Success": continue
            sub = fails[fails["error_type"]==etype]
            if sub.empty: continue
            fig_de.add_trace(go.Box(
                y=sub["DOS"], name=short_map.get(etype,etype),
                marker_color=color, boxmean=True))
        fig_de.update_layout(title="DOS by Error Type",
                              paper_bgcolor=CHART_BG, plot_bgcolor=CHART_PLOT,
                              font=dict(color=CHART_FONT),
                              yaxis_title="Decision Optimality Score")
        st.plotly_chart(fig_de, use_container_width=True)

    col_e3, col_e4 = st.columns(2)
    with col_e3:
        fig_oe = px.box(fails, x="error_type", y="opportunity_cost",
                        color="error_type", color_discrete_map=ERROR_COLORS,
                        title="Opportunity Cost by Error Type",
                        labels={"opportunity_cost":"Opp. Cost",
                                "error_type":"Error Type"})
        fig_oe.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_PLOT,
                              font=dict(color=CHART_FONT), showlegend=False,
                              xaxis_tickangle=-15)
        st.plotly_chart(fig_oe, use_container_width=True)

    with col_e4:
        fig_pve = px.scatter(
            fails, x="P_t", y="V_t", color="error_type",
            color_discrete_map=ERROR_COLORS,
            title="P(t) vs V(t) for Failed Passes",
            labels={"P_t":"P(t)","V_t":"V(t)","error_type":"Error Type"},
            hover_data=["pass_id","G_t","team"], opacity=0.7)
        fig_pve.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(color="white", dash="dash", width=1))
        fig_pve.add_annotation(x=0.82, y=0.88, text="P = V",
                                showarrow=False, font=dict(color=CHART_FONT, size=10))
        fig_pve.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_PLOT,
                               font=dict(color=CHART_FONT))
        st.plotly_chart(fig_pve, use_container_width=True)
        st.caption("Above diagonal (V>P): Overambitious — space open but pass too risky.  "
                   "Below (P≥V): Trapped — pass safe but no good space available.")


# ╔═══════════════════════════════════════════════════════════════════════════
# TAB 6 — FORMATION OPTIMIZER  (hidden — tab removed from UI)
# ╚═══════════════════════════════════════════════════════════════════════════
if False:  # noqa — kept for future re-enable
 if False:
    st.subheader("Formation Optimisation via Gradient Ascent")

    if opt_agg.empty:
        st.warning("No optimisation data. Run the Section 11 batch cell "
                   "in the notebook, then re-run the export cell.")
    else:
        opt_f = opt_agg.copy()
        if team_filter != "Both":
            opt_f = opt_f[opt_f["team"] == team_filter]

        o1, o2, o3, o4 = st.columns(4)
        metric_card(o1, "Events Optimised",  len(opt_f))
        metric_card(o2, "Avg Gain Before",   f"{opt_f['gain_before'].mean():.4f}")
        metric_card(o3, "Avg Gain After",    f"{opt_f['gain_after'].mean():.4f}")
        metric_card(o4, "Avg Improvement",   f"{opt_f['pct_improvement'].mean():.1f}%")

        st.divider()
        col_o1, col_o2 = st.columns(2)

        with col_o1:
            fig_imp = px.histogram(opt_f, x="pct_improvement", color="team",
                                   color_discrete_map=TEAM_COLORS,
                                   nbins=30, barmode="overlay", opacity=0.75,
                                   title="Formation Gain Improvement (%)",
                                   labels={"pct_improvement":"% Improvement"})
            fig_imp.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_PLOT,
                                   font=dict(color=CHART_FONT))
            st.plotly_chart(fig_imp, use_container_width=True)

        with col_o2:
            lim = max(opt_f["gain_after"].max(), opt_f["gain_before"].max())*1.05
            fig_ba = px.scatter(opt_f, x="gain_before", y="gain_after",
                                color="success",
                                color_discrete_map={1:"#2ecc71",0:"#e74c3c"},
                                hover_data=["pass_id","team","pct_improvement"],
                                title="Formation Gain: Before vs After",
                                labels={"gain_before":"Before","gain_after":"After"},
                                opacity=0.6)
            fig_ba.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                             line=dict(color="white", dash="dash", width=1))
            fig_ba.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_PLOT,
                                  font=dict(color=CHART_FONT))
            st.plotly_chart(fig_ba, use_container_width=True)

        st.divider()
        col_o3, col_o4 = st.columns(2)

        with col_o3:
            bt = (opt_f.groupby("team")[["gain_before","gain_after",
                                          "pct_improvement"]].mean()
                  .reset_index().round(4))
            fig_bt = go.Figure()
            for col_n, lbl, clr in [("gain_before","Before","#3498db"),
                                     ("gain_after","After","#2ecc71")]:
                fig_bt.add_trace(go.Bar(
                    x=bt["team"], y=bt[col_n], name=lbl, marker_color=clr,
                    text=bt[col_n].round(4), textposition="outside"))
            fig_bt.update_layout(title="Avg Formation Gain by Team",
                                  barmode="group",
                                  paper_bgcolor=CHART_BG, plot_bgcolor=CHART_PLOT,
                                  font=dict(color=CHART_FONT),
                                  yaxis_title="Avg F(t)")
            st.plotly_chart(fig_bt, use_container_width=True)

        with col_o4:
            bo = (opt_f.groupby("success")[["pct_improvement"]].mean()
                  .reset_index())
            bo["label"] = bo["success"].map({1:"Successful",0:"Unsuccessful"})
            fig_bo = go.Figure(go.Bar(
                x=bo["label"], y=bo["pct_improvement"],
                marker_color=["#2ecc71","#e74c3c"],
                text=bo["pct_improvement"].round(2).astype(str)+"%",
                textposition="outside"))
            fig_bo.update_layout(title="Avg % Improvement by Outcome",
                                  paper_bgcolor=CHART_BG, plot_bgcolor=CHART_PLOT,
                                  font=dict(color=CHART_FONT),
                                  yaxis_title="Avg % Improvement")
            st.plotly_chart(fig_bo, use_container_width=True)

        st.divider()
        st.markdown("#### Top 10 Events — Highest Formation Gain Improvement")
        top10 = (opt_f.nlargest(10,"pct_improvement")
                 [["pass_id","team","success","gain_before","gain_after",
                   "gain_improvement","pct_improvement"]]
                 .reset_index(drop=True))
        top10["success"] = top10["success"].map({1:"✓",0:"✗"})
        st.dataframe(top10.round(4).rename(columns={
            "pass_id":"Pass","team":"Team","success":"OK",
            "gain_before":"Before","gain_after":"After",
            "gain_improvement":"ΔGain","pct_improvement":"% Gain"}),
            use_container_width=True, hide_index=True)
