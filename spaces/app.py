"""ORBIT Dashboard — HuggingFace Spaces version.

Interactive dashboard for robot learning failure analysis.
Uses pre-generated synthetic deployment data.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import (
    deserialize_episodes,
    discover_sessions,
    episodes_to_summary_df,
    get_failure_type_counts,
    load_session_episodes,
    run_detector_pipeline,
    run_prescriber,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLOR_SUCCESS = "#2ecc71"
COLOR_FAILURE = "#e74c3c"
COLOR_UNKNOWN = "#95a5a6"
OUTCOME_COLORS = {"success": COLOR_SUCCESS, "failure": COLOR_FAILURE, "unknown": COLOR_UNKNOWN}

DEFAULT_DATA_DIR = "./data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_data_dir() -> str:
    if "data_dir" not in st.session_state:
        st.session_state["data_dir"] = DEFAULT_DATA_DIR
    return st.session_state["data_dir"]


def _ensure_episodes():
    if "selected_session" not in st.session_state:
        return None
    session = st.session_state["selected_session"]
    if not session:
        return None
    cache_key = f"episodes_{session['session_id']}"
    if cache_key not in st.session_state:
        ep_dicts = load_session_episodes(session["file_path"])
        st.session_state[cache_key] = ep_dicts
    return st.session_state[cache_key]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> None:
    st.sidebar.title("🛰️ ORBIT")
    st.sidebar.caption("Deployment Diagnostics for Robot Policies")
    st.sidebar.divider()

    data_dir = st.sidebar.text_input("Data directory", value=_get_data_dir())
    st.session_state["data_dir"] = data_dir

    sessions = discover_sessions(data_dir)
    if not sessions:
        st.sidebar.warning("No session files found in this directory.")
        st.session_state["selected_session"] = None
        return

    labels = [f"{s['session_id'][:8]}... ({s['episode_count']} episodes)" for s in sessions]
    idx = st.sidebar.selectbox("Session", range(len(labels)), format_func=lambda i: labels[i])
    st.session_state["selected_session"] = sessions[idx]

    if st.sidebar.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("[GitHub](https://github.com/Rahillasne/Orbit)")
    st.sidebar.markdown("Made by Rahil Lasne")


# ---------------------------------------------------------------------------
# PAGE 1: Session Overview
# ---------------------------------------------------------------------------


def page_session_overview() -> None:
    st.header("Session Overview")

    st.info(
        "This demo uses synthetic deployment data (20 episodes, 60% success rate). "
        "In real use, you'd point ORBIT at your own deployment logs."
    )

    ep_dicts = _ensure_episodes()
    if not ep_dicts:
        st.info("Select a session from the sidebar to view episodes.")
        return

    episodes = deserialize_episodes(ep_dicts)
    df = episodes_to_summary_df(episodes)
    if df.empty:
        st.info("No episodes in this session.")
        return

    total = len(df)
    successes = int((df["outcome"] == "success").sum())
    success_rate = successes / total if total > 0 else 0
    avg_duration = df["duration"].dropna().mean() if not df["duration"].isna().all() else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Episodes", total)
    c2.metric("Success Rate", f"{success_rate:.0%}")
    c3.metric("Avg Duration", f"{avg_duration:.1f}s" if avg_duration else "N/A")
    c4.metric("Avg Frames", f"{df['num_frames'].mean():.0f}")

    st.subheader("Episode Timeline")
    fig = go.Figure()
    for outcome, color in OUTCOME_COLORS.items():
        mask = df["outcome"] == outcome
        subset = df[mask]
        if subset.empty:
            continue
        fig.add_trace(go.Bar(
            y=subset["episode_id"], x=subset["num_frames"], orientation="h",
            name=outcome.capitalize(), marker_color=color,
            hovertemplate="<b>%{y}</b><br>Frames: %{x}<br>Reward: %{customdata[0]:.2f}<br><extra></extra>",
            customdata=subset[["total_reward"]].values,
        ))
    fig.update_layout(
        xaxis_title="Number of Frames", yaxis_title="Episode", barmode="stack",
        height=max(300, 30 * total), template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Quick Failure Summary")
    detection_results = run_detector_pipeline(episodes)
    counts = get_failure_type_counts(detection_results)
    if counts:
        top3 = dict(list(counts.items())[:3])
        cols = st.columns(len(top3))
        for col, (detector, count) in zip(cols, top3.items()):
            label = detector.replace("Detector", "")
            col.metric(label, count)
    else:
        st.success("No failures detected by heuristic detectors.")


# ---------------------------------------------------------------------------
# PAGE 2: Distribution View
# ---------------------------------------------------------------------------


def page_distribution_view() -> None:
    st.header("Distribution View")

    ep_dicts = _ensure_episodes()
    if not ep_dicts:
        st.info("Select a session from the sidebar.")
        return

    episodes = deserialize_episodes(ep_dicts)

    st.subheader("Embedding Space (UMAP)")
    st.info(
        "UMAP visualization requires the full ORBIT installation with ML dependencies. "
        "Install the full package and run the EmbeddingAnalyzer to generate interactive plots."
    )

    df = episodes_to_summary_df(episodes)
    if len(df) > 1:
        st.subheader("Episode Metric Scatter")
        fig = px.scatter(
            df, x="total_reward", y="num_frames", color="outcome",
            size="avg_action_mag", hover_data=["episode_id", "task_name"],
            color_discrete_map=OUTCOME_COLORS, title="Reward vs Frames (colored by outcome)",
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Reward per Episode")
    if not df.empty:
        fig = px.bar(df, x="episode_id", y="total_reward", color="outcome", color_discrete_map=OUTCOME_COLORS)
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 3: Failure Analysis
# ---------------------------------------------------------------------------


def page_failure_analysis() -> None:
    st.header("Failure Analysis")

    ep_dicts = _ensure_episodes()
    if not ep_dicts:
        st.info("Select a session from the sidebar.")
        return

    episodes = deserialize_episodes(ep_dicts)

    st.subheader("Heuristic Detection Results")
    detection_results = run_detector_pipeline(episodes)

    n_failures = sum(1 for r in detection_results if r["is_failure"])
    st.write(f"**{len(detection_results)}** episodes analyzed, **{n_failures}** failures detected.")

    counts = get_failure_type_counts(detection_results)
    if counts:
        reason_df = pd.DataFrame([{"Detector": k, "Count": v} for k, v in counts.items()])
        fig = px.pie(
            reason_df, names="Detector", values="Count",
            title="Failure Type Distribution", color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)

    failure_rows = []
    for r in detection_results:
        if r["is_failure"]:
            failure_rows.append({
                "Episode": r["episode_id"][:8],
                "Probability": f"{r['failure_probability']:.0%}",
                "Detections": len(r["detections"]),
                "Details": "; ".join(d["description"] for d in r["detections"][:3]),
            })
    if failure_rows:
        st.dataframe(pd.DataFrame(failure_rows), use_container_width=True)

    st.subheader("Failure Clusters")
    st.info(
        "Cluster analysis requires the full ORBIT installation with ML dependencies. "
        "Run the EmbeddingAnalyzer with `generate_viz=True` to generate cluster reports."
    )


# ---------------------------------------------------------------------------
# PAGE 4: Prescriptions
# ---------------------------------------------------------------------------


def page_prescriptions() -> None:
    st.header("Prescriptions")

    st.info(
        "These prescriptions are generated automatically from the distribution gap analysis. "
        "Each task targets a specific failure cluster."
    )

    ep_dicts = _ensure_episodes()
    if not ep_dicts:
        st.info("Select a session from the sidebar.")
        return

    episodes = deserialize_episodes(ep_dicts)
    detection_results = run_detector_pipeline(episodes)

    n_failures = sum(1 for r in detection_results if r["is_failure"])
    if n_failures == 0:
        st.success("No failures detected — no prescriptions needed.")
        return

    report = run_prescriber(detection_results, episodes)
    st.write(f"**Summary**: {report['summary']}")
    st.write(f"Failures analyzed: **{report['num_failures_analyzed']}**")

    st.subheader("Recommendations")
    prescriptions = report["prescriptions"]
    if prescriptions:
        table_data = []
        for p in prescriptions:
            desc = p["description"]
            table_data.append({
                "Priority": p["priority"],
                "Type": p["type"].replace("_", " ").title(),
                "Title": p["title"],
                "Confidence": f"{p['confidence']:.0%}",
                "Description": desc[:120] + "..." if len(desc) > 120 else desc,
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        for p in prescriptions:
            with st.expander(f"[{p['priority']}] {p['title']}"):
                st.write(p["description"])
                if p["evidence"]:
                    st.write("**Evidence:**")
                    for e in p["evidence"]:
                        st.write(f"- {e}")
                if p["suggested_params"]:
                    st.write("**Suggested parameters:**")
                    st.json(p["suggested_params"])

        st.subheader("Export")
        c1, c2, c3 = st.columns(3)

        json_str = json.dumps(prescriptions, indent=2)
        c1.download_button("Download JSON", data=json_str, file_name="orbit_prescriptions.json", mime="application/json")

        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=["priority", "type", "title", "description", "confidence", "evidence"])
        writer.writeheader()
        for p in prescriptions:
            writer.writerow({
                "priority": p["priority"], "type": p["type"], "title": p["title"],
                "description": p["description"], "confidence": p["confidence"],
                "evidence": "; ".join(p["evidence"]),
            })
        c2.download_button("Download CSV", data=csv_buf.getvalue(), file_name="orbit_prescriptions.csv", mime="text/csv")

        md_lines = ["# Orbit Prescriptions\n"]
        for p in prescriptions:
            md_lines.append(f"## [{p['priority']}] {p['title']}")
            md_lines.append(f"**Type:** {p['type']}  ")
            md_lines.append(f"**Confidence:** {p['confidence']:.0%}\n")
            md_lines.append(p["description"] + "\n")
            if p["evidence"]:
                md_lines.append("**Evidence:**")
                for e in p["evidence"]:
                    md_lines.append(f"- {e}")
                md_lines.append("")
        c3.download_button("Download Markdown", data="\n".join(md_lines), file_name="orbit_prescriptions.md", mime="text/markdown")

    st.subheader("Progress Tracking")
    st.text_area(
        "Notes on new data collection",
        value=st.session_state.get("progress_notes", ""),
        key="progress_notes",
        placeholder="Track your progress collecting new data here...",
    )


# ---------------------------------------------------------------------------
# PAGE 5: Settings
# ---------------------------------------------------------------------------


def page_settings() -> None:
    st.header("Settings")

    st.subheader("Detector Thresholds")
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Stall velocity threshold", value=0.001, format="%.4f", key="stall_velocity_threshold")
        st.number_input("Stall min frames", value=10, min_value=1, key="stall_min_frames")
        st.number_input("Gripper closed threshold", value=0.3, format="%.2f", key="gripper_closed_threshold")
    with c2:
        st.number_input("Timeout max duration (s)", value=60.0, format="%.1f", key="timeout_max_duration")
        st.number_input("Timeout max frames", value=1000, min_value=1, key="timeout_max_frames")
        st.number_input("Min total reward threshold", value=0.0, format="%.2f", key="min_reward_threshold")

    st.subheader("Embedding Model")
    st.selectbox("Model", ["google/siglip-base-patch16-224 (recommended)", "ViT-B-32 (OpenCLIP)", "ViT-L-14 (OpenCLIP)"], key="embedding_model")
    st.selectbox("Device", ["cpu", "cuda", "mps"], key="embedding_device")

    st.subheader("Workspace Bounds (Joint Limits)")
    st.caption("Configure per-joint position limits for the OutOfBoundsDetector.")
    n_joints = st.number_input("Number of joints", value=6, min_value=1, max_value=12)
    cols = st.columns(int(n_joints))
    for j, col in enumerate(cols):
        with col:
            st.number_input(f"J{j} lower", value=-3.14, format="%.3f", key=f"joint_{j}_lower")
            st.number_input(f"J{j} upper", value=3.14, format="%.3f", key=f"joint_{j}_upper")


# ---------------------------------------------------------------------------
# App entry
# ---------------------------------------------------------------------------


def _run_app() -> None:
    st.set_page_config(page_title="ORBIT Demo", page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")
    render_sidebar()
    pages = [
        st.Page(page_session_overview, title="Session Overview"),
        st.Page(page_distribution_view, title="Distribution View"),
        st.Page(page_failure_analysis, title="Failure Analysis"),
        st.Page(page_prescriptions, title="Prescriptions"),
        st.Page(page_settings, title="Settings"),
    ]
    nav = st.navigation(pages)
    nav.run()


if __name__ == "__main__":
    _run_app()
else:
    try:
        _ctx = st.runtime.scriptrunner.get_script_run_ctx()
        if _ctx is not None:
            _run_app()
    except Exception:
        pass
