"""Streamlit dashboard for Orbit — robot learning failure analysis.

Provides five pages:
  1. Session Overview — summary stats, episode timeline, quick failure summary
  2. Distribution View — UMAP scatter, gap heatmap, cluster toggles
  3. Failure Analysis — cluster browser, representative frames, VLM descriptions
  4. Prescriptions — priority-ranked recommendations, export buttons
  5. Settings — detector thresholds, VLM key, embedding model, workspace bounds
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from orbit.dashboard.data_loader import (
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


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Entry point for ``orbit-dashboard`` and ``orbit dashboard`` CLI commands."""
    args = sys.argv[1:]
    data_dir = "./orbit_data"
    port = "8501"

    i = 0
    while i < len(args):
        if args[i] == "--data-dir" and i + 1 < len(args):
            data_dir = args[i + 1]
            i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            port = args[i + 1]
            i += 2
        else:
            i += 1

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        __file__,
        f"--server.port={port}",
        "--server.headless=true",
        "--",
        f"--data-dir={data_dir}",
    ]
    subprocess.run(cmd, check=True)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _parse_streamlit_args() -> str:
    """Extract ``--data-dir`` from Streamlit's ``--`` forwarded args."""
    for arg in sys.argv:
        if arg.startswith("--data-dir="):
            return arg.split("=", 1)[1]
    return "./orbit_data"


def _get_data_dir() -> str:
    """Return the data directory, preferring session state then CLI arg."""
    if "data_dir" not in st.session_state:
        st.session_state["data_dir"] = _parse_streamlit_args()
    return st.session_state["data_dir"]


def _ensure_episodes():
    """Load episodes into session state for the selected session."""
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


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar (shared across all pages)
# ═══════════════════════════════════════════════════════════════════════════


def render_sidebar() -> None:
    """Render the sidebar with session selector and data-dir input."""
    st.sidebar.title("Orbit Dashboard")
    st.sidebar.caption("Robot Learning Failure Debugger")
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


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: Session Overview
# ═══════════════════════════════════════════════════════════════════════════


def page_session_overview() -> None:
    """Session Overview — summary stats and episode timeline."""
    st.header("Session Overview")

    ep_dicts = _ensure_episodes()
    if not ep_dicts:
        st.info("Select a session from the sidebar to view episodes.")
        return

    episodes = deserialize_episodes(ep_dicts)
    df = episodes_to_summary_df(episodes)
    if df.empty:
        st.info("No episodes in this session.")
        return

    # --- Summary stats ---
    total = len(df)
    successes = int((df["outcome"] == "success").sum())
    success_rate = successes / total if total > 0 else 0
    avg_duration = df["duration"].dropna().mean() if not df["duration"].isna().all() else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Episodes", total)
    c2.metric("Success Rate", f"{success_rate:.0%}")
    c3.metric("Avg Duration", f"{avg_duration:.1f}s" if avg_duration else "N/A")
    c4.metric("Avg Frames", f"{df['num_frames'].mean():.0f}")

    # --- Episode timeline (horizontal bars) ---
    st.subheader("Episode Timeline")
    fig = go.Figure()
    for outcome, color in OUTCOME_COLORS.items():
        mask = df["outcome"] == outcome
        subset = df[mask]
        if subset.empty:
            continue
        fig.add_trace(
            go.Bar(
                y=subset["episode_id"],
                x=subset["num_frames"],
                orientation="h",
                name=outcome.capitalize(),
                marker_color=color,
                hovertemplate=(
                    "<b>%{y}</b><br>Frames: %{x}<br>Reward: %{customdata[0]:.2f}<br><extra></extra>"
                ),
                customdata=subset[["total_reward"]].values,
            )
        )
    fig.update_layout(
        xaxis_title="Number of Frames",
        yaxis_title="Episode",
        barmode="stack",
        height=max(300, 30 * total),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Quick failure summary ---
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


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: Distribution View
# ═══════════════════════════════════════════════════════════════════════════


def page_distribution_view() -> None:
    """Distribution View — UMAP scatter and gap heatmap."""
    st.header("Distribution View")

    ep_dicts = _ensure_episodes()
    if not ep_dicts:
        st.info("Select a session from the sidebar.")
        return

    episodes = deserialize_episodes(ep_dicts)
    data_dir = st.session_state.get("data_dir", "./orbit_data")

    viz_dir = Path(data_dir) / "orbit_viz"
    scatter_html = viz_dir / "embedding_space.html"
    heatmap_html = viz_dir / "gap_heatmap.html"

    # --- UMAP Scatter ---
    st.subheader("Embedding Space (UMAP)")

    if scatter_html.exists():
        st.radio("View", ["All data", "Failures only"], horizontal=True, key="umap_view")
        html_content = scatter_html.read_text()
        st.components.v1.html(html_content, height=750, scrolling=True)
        st.caption(f"Loaded from: `{scatter_html}`")
    else:
        st.info(
            "No pre-computed UMAP visualization found. "
            "Run the full analysis pipeline first to generate embeddings:\n\n"
            "```python\n"
            "from orbit.analyzer import EmbeddingAnalyzer\n"
            "analyzer = EmbeddingAnalyzer()\n"
            "report = analyzer.analyze(training_source, deployment_episodes)\n"
            "```"
        )

        # Fallback: simple outcome scatter using episode-level metrics
        df = episodes_to_summary_df(episodes)
        if len(df) > 1:
            st.subheader("Episode Metric Scatter (fallback)")
            fig = px.scatter(
                df,
                x="total_reward",
                y="num_frames",
                color="outcome",
                size="avg_action_mag",
                hover_data=["episode_id", "task_name"],
                color_discrete_map=OUTCOME_COLORS,
                title="Reward vs Frames (colored by outcome)",
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    # --- Gap Heatmap ---
    st.subheader("Gap Score Heatmap")

    if heatmap_html.exists():
        html_content = heatmap_html.read_text()
        st.components.v1.html(html_content, height=600, scrolling=True)
        st.caption(f"Loaded from: `{heatmap_html}`")
    else:
        st.info(
            "No pre-computed gap heatmap found. Run the embedding analyzer to generate gap scores."
        )

        # Fallback: reward bar per episode
        df = episodes_to_summary_df(episodes)
        if not df.empty:
            st.subheader("Reward per Episode (fallback)")
            fig = px.bar(
                df,
                x="episode_id",
                y="total_reward",
                color="outcome",
                color_discrete_map=OUTCOME_COLORS,
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: Failure Analysis
# ═══════════════════════════════════════════════════════════════════════════


def page_failure_analysis() -> None:
    """Failure Analysis — detector results, cluster browser, VLM descriptions."""
    st.header("Failure Analysis")

    ep_dicts = _ensure_episodes()
    if not ep_dicts:
        st.info("Select a session from the sidebar.")
        return

    episodes = deserialize_episodes(ep_dicts)

    # --- Detector results ---
    st.subheader("Heuristic Detection Results")
    detection_results = run_detector_pipeline(episodes)

    n_failures = sum(1 for r in detection_results if r["is_failure"])
    st.write(f"**{len(detection_results)}** episodes analyzed, **{n_failures}** failures detected.")

    # Failure reason distribution
    counts = get_failure_type_counts(detection_results)
    if counts:
        reason_df = pd.DataFrame([{"Detector": k, "Count": v} for k, v in counts.items()])
        fig = px.pie(
            reason_df,
            names="Detector",
            values="Count",
            title="Failure Type Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Failure details table
    failure_rows = []
    for r in detection_results:
        if r["is_failure"]:
            failure_rows.append(
                {
                    "Episode": r["episode_id"][:8],
                    "Probability": f"{r['failure_probability']:.0%}",
                    "Detections": len(r["detections"]),
                    "Details": "; ".join(d["description"] for d in r["detections"][:3]),
                }
            )
    if failure_rows:
        st.dataframe(pd.DataFrame(failure_rows), use_container_width=True)

    # --- Cluster browser ---
    st.subheader("Failure Clusters")

    data_dir = st.session_state.get("data_dir", "./orbit_data")
    cluster_file = Path(data_dir) / "orbit_viz" / "cluster_report.json"

    if cluster_file.exists():
        report = json.loads(cluster_file.read_text())
        clusters = report.get("clusters", [])
        if clusters:
            cluster_names = [
                f"Cluster {c['cluster_id']} ({c['size']} frames, gap={c['avg_gap_score']:.3f})"
                for c in clusters
            ]
            selected = st.selectbox(
                "Select cluster",
                range(len(cluster_names)),
                format_func=lambda i: cluster_names[i],
            )
            cluster = clusters[selected]

            c1, c2, c3 = st.columns(3)
            c1.metric("Frames", cluster["size"])
            c2.metric("Avg Gap Score", f"{cluster['avg_gap_score']:.3f}")
            temporal = cluster.get("temporal_distribution", {})
            c3.metric(
                "Peak Phase",
                max(temporal, key=temporal.get) if temporal else "N/A",
            )

            if temporal:
                st.bar_chart(pd.DataFrame([temporal]))

            rep_episodes = cluster.get("representative_episode_ids", [])
            if rep_episodes:
                st.write(f"Representative episodes: {', '.join(e[:8] for e in rep_episodes)}")

            _show_frame_images(episodes)
        else:
            st.info("No clusters found in the report.")
    else:
        st.info(
            "No cluster analysis data found. Run the embedding analyzer "
            "with `generate_viz=True` to generate cluster reports."
        )

    # --- VLM descriptions ---
    st.subheader("VLM Failure Descriptions")
    vlm_key = st.session_state.get("vlm_api_key", "")
    if not vlm_key:
        st.info(
            "Set a VLM API key in the Settings page to enable AI failure "
            "descriptions, or VLM descriptions will use local OpenCLIP."
        )

    if st.button("Generate VLM Descriptions"):
        with st.spinner("Analyzing failure frames with OpenCLIP..."):
            _generate_vlm_descriptions(episodes)


def _show_frame_images(episodes) -> None:
    """Show a few images from failure episodes (best-effort)."""
    failure_episodes = [e for e in episodes if e.outcome.value == "failure"]
    images_shown = 0
    max_images = 5
    cols = st.columns(max_images)
    for ep in failure_episodes:
        for i, frame in enumerate(ep.frames):
            if frame.image_path and Path(frame.image_path).exists():
                if images_shown < max_images:
                    cols[images_shown].image(
                        frame.image_path,
                        caption=f"Ep {str(ep.episode_id)[:6]} F{i}",
                        width=120,
                    )
                    images_shown += 1
        if images_shown >= max_images:
            break


def _generate_vlm_descriptions(episodes) -> None:
    """Run VLM failure descriptions using OpenCLIP zero-shot classification."""
    try:
        from orbit.logger.compat import episode_to_legacy
        from orbit.vlm.failure_describer import FailureDescriber

        describer = FailureDescriber()
        failure_eps = [ep for ep in episodes if ep.outcome.value == "failure"]
        if not failure_eps:
            st.info("No failure episodes to analyze.")
            return
        for ep in failure_eps[:3]:
            legacy = episode_to_legacy(ep)
            desc = describer.describe(legacy)
            st.write(f"**Episode {str(ep.episode_id)[:8]}**: {desc.description}")
    except Exception as e:
        st.error(f"VLM analysis failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4: Prescriptions
# ═══════════════════════════════════════════════════════════════════════════


def page_prescriptions() -> None:
    """Prescriptions — priority-ranked recommendations with export."""
    st.header("Prescriptions")

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

    # Priority-ranked table
    st.subheader("Recommendations")
    prescriptions = report["prescriptions"]
    if prescriptions:
        table_data = []
        for p in prescriptions:
            desc = p["description"]
            table_data.append(
                {
                    "Priority": p["priority"],
                    "Type": p["type"].replace("_", " ").title(),
                    "Title": p["title"],
                    "Confidence": f"{p['confidence']:.0%}",
                    "Description": desc[:120] + "..." if len(desc) > 120 else desc,
                }
            )
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        # Detail expanders
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

        # Export buttons
        st.subheader("Export")
        c1, c2, c3 = st.columns(3)

        json_str = json.dumps(prescriptions, indent=2)
        c1.download_button(
            "Download JSON",
            data=json_str,
            file_name="orbit_prescriptions.json",
            mime="application/json",
        )

        csv_buf = io.StringIO()
        writer = csv.DictWriter(
            csv_buf,
            fieldnames=[
                "priority",
                "type",
                "title",
                "description",
                "confidence",
                "evidence",
            ],
        )
        writer.writeheader()
        for p in prescriptions:
            writer.writerow(
                {
                    "priority": p["priority"],
                    "type": p["type"],
                    "title": p["title"],
                    "description": p["description"],
                    "confidence": p["confidence"],
                    "evidence": "; ".join(p["evidence"]),
                }
            )
        c2.download_button(
            "Download CSV",
            data=csv_buf.getvalue(),
            file_name="orbit_prescriptions.csv",
            mime="text/csv",
        )

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
        c3.download_button(
            "Download Markdown",
            data="\n".join(md_lines),
            file_name="orbit_prescriptions.md",
            mime="text/markdown",
        )

    # Progress tracking
    st.subheader("Progress Tracking")
    st.text_area(
        "Notes on new data collection",
        value=st.session_state.get("progress_notes", ""),
        key="progress_notes",
        placeholder="Track your progress collecting new data here...",
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6: Dataset Profiler
# ═══════════════════════════════════════════════════════════════════════════


def _traffic_light(score: float) -> str:
    """Return a traffic-light color string for a 0-1 score."""
    if score >= 0.7:
        return "#2ecc71"  # green
    if score >= 0.4:
        return "#f39c12"  # yellow
    return "#e74c3c"  # red


def _verdict(score: float) -> str:
    """Map a 0-1 score to a human verdict."""
    if score >= 0.7:
        return "Strong"
    if score >= 0.5:
        return "Adequate"
    if score >= 0.2:
        return "Weak"
    return "No Coverage"


def page_dataset_profiler() -> None:
    """Dataset Profiler — analyze coverage, capabilities, and quality."""
    st.header("Dataset Profiler")

    # ------------------------------------------------------------------
    # Section 1: Data Input
    # ------------------------------------------------------------------
    st.subheader("Data Input")

    input_mode = st.radio("Input source", ["Local Directory", "HuggingFace Hub"], horizontal=True)

    if input_mode == "Local Directory":
        data_path = st.text_input(
            "Dataset directory",
            value=st.session_state.get("data_dir", "./orbit_data"),
            key="profiler_data_path",
        )
    else:
        data_path = st.text_input(
            "HuggingFace repo ID",
            placeholder="lerobot/aloha_sim_insertion_human",
            key="profiler_hub_repo",
        )

    task_text = st.text_area(
        "Task descriptions (one per line)",
        placeholder="pick up cup\nopen drawer\nwipe surface",
        key="profiler_tasks",
    )

    # Comparison inputs
    with st.expander("Comparison Mode (optional)"):
        st.caption("Provide a second dataset to compare side-by-side.")
        compare_path = st.text_input("Second dataset directory", key="profiler_compare_path")

    col_run, col_status = st.columns([1, 3])
    run_clicked = col_run.button("Profile Dataset", type="primary")

    if run_clicked:
        if not data_path:
            st.error("Please provide a dataset path or repo ID.")
            return

        try:
            from orbit.profile.profiler import DatasetProfiler
            from orbit.profile.report import ProfileReporter
        except ImportError:
            st.error(
                "Profile dependencies not installed. Run: `pip install orbit-robotics[profile]`"
            )
            return

        tasks = [t.strip() for t in task_text.strip().splitlines() if t.strip()] or None

        with st.spinner("Profiling dataset — this may take several minutes..."):
            try:
                profiler = DatasetProfiler()
                if input_mode == "Local Directory":
                    profile_result = profiler.profile(data_path, task_descriptions=tasks)
                else:
                    profile_result = profiler.profile_from_hub(data_path, task_descriptions=tasks)

                reporter = ProfileReporter()
                dashboard_data = reporter.generate_dashboard_data(profile_result)

                st.session_state["profiler_profile"] = profile_result
                st.session_state["profiler_dashboard_data"] = dashboard_data
            except FileNotFoundError as e:
                st.error(f"Dataset not found: {e}")
                return
            except Exception as e:
                st.error(f"Profiling failed: {e}")
                return

        # Run comparison if a second path was provided
        if compare_path.strip():
            with st.spinner("Profiling second dataset for comparison..."):
                try:
                    profile_b = profiler.profile(compare_path.strip(), task_descriptions=tasks)
                    st.session_state["profiler_profile_b"] = profile_b
                    reporter_b = ProfileReporter()
                    st.session_state["profiler_dashboard_data_b"] = (
                        reporter_b.generate_dashboard_data(profile_b)
                    )
                except Exception as e:
                    st.warning(f"Comparison profiling failed: {e}")

        st.success("Profiling complete!")

    # Guard: nothing to show yet
    if "profiler_dashboard_data" not in st.session_state:
        st.info("Configure inputs above and click **Profile Dataset** to begin.")
        return

    data = st.session_state["profiler_dashboard_data"]
    profile_obj = st.session_state["profiler_profile"]

    # ------------------------------------------------------------------
    # Section 2: Overview
    # ------------------------------------------------------------------
    st.subheader("Overview")
    stats = data["summary_stats"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Episodes", stats["num_episodes"])
    c2.metric("Frames", stats["num_frames"])

    cov = stats["overall_coverage"]
    c3.metric("Coverage Score", f"{cov:.3f}")
    qual = stats["aggregate_quality"]
    c4.metric("Quality Score", f"{qual:.3f}")

    # Traffic lights
    tl1, tl2 = st.columns(2)
    tl1.markdown(
        f'<div style="display:inline-block;width:18px;height:18px;'
        f'border-radius:50%;background:{_traffic_light(cov)}"></div>'
        f" Coverage: **{cov:.3f}**",
        unsafe_allow_html=True,
    )
    tl2.markdown(
        f'<div style="display:inline-block;width:18px;height:18px;'
        f'border-radius:50%;background:{_traffic_light(qual)}"></div>'
        f" Quality: **{qual:.3f}**",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Section 3: Coverage Map
    # ------------------------------------------------------------------
    st.subheader("Coverage Map")
    umap_data = data.get("coverage_umap")
    if umap_data is not None:
        import numpy as np

        umap_arr = np.array(umap_data)
        umap_df = pd.DataFrame({"x": umap_arr[:, 0], "y": umap_arr[:, 1]})
        fig = px.scatter(
            umap_df,
            x="x",
            y="y",
            title="Embedding Space (UMAP)",
            opacity=0.5,
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback: bar chart of cluster sizes
        regions = profile_obj.coverage.dense_regions + profile_obj.coverage.sparse_regions
        if regions:
            cluster_df = pd.DataFrame(
                [
                    {
                        "Cluster": f"C{i}",
                        "Size": r.get("size", r.get("density", 0)),
                        "Type": "Dense" if r in profile_obj.coverage.dense_regions else "Sparse",
                    }
                    for i, r in enumerate(regions)
                ]
            )
            fig = px.bar(
                cluster_df,
                x="Cluster",
                y="Size",
                color="Type",
                color_discrete_map={"Dense": "#3498db", "Sparse": "#e74c3c"},
                title="Cluster Sizes",
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No coverage data available.")

    # ------------------------------------------------------------------
    # Section 4: Capability Table
    # ------------------------------------------------------------------
    st.subheader("Capability Scores")
    cap_bars = data.get("capability_bars", [])
    if cap_bars:
        cap_df = pd.DataFrame(cap_bars)
        cap_df["verdict"] = cap_df["score"].apply(_verdict)

        # Bar chart
        fig = px.bar(
            cap_df,
            x="task",
            y="score",
            color="score",
            color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
            range_color=[0, 1],
            title="Task Capability Scores",
        )
        fig.update_layout(template="plotly_white", xaxis_title="Task", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.dataframe(
            cap_df[["task", "score", "confidence", "verdict"]].rename(
                columns={
                    "task": "Task",
                    "score": "Score",
                    "confidence": "Confidence",
                    "verdict": "Verdict",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        # Expandable details
        for cap in profile_obj.capabilities:
            with st.expander(f"{cap.task_description} — {_verdict(cap.score)}"):
                st.write(f"**Score:** {cap.score:.3f} | **Confidence:** {cap.confidence:.3f}")
                st.write(f"**Supporting episodes:** {cap.supporting_episodes}")
                if cap.gap_description:
                    st.write(f"**Gap:** {cap.gap_description}")
    else:
        st.info("No tasks scored. Provide task descriptions to see capability scores.")

    # ------------------------------------------------------------------
    # Section 5: Quality Distribution
    # ------------------------------------------------------------------
    st.subheader("Quality Distribution")
    qual_data = data.get("quality_histogram", {})
    scores = qual_data.get("scores", [])
    if scores:
        fig = px.histogram(
            x=scores,
            nbins=20,
            labels={"x": "Quality Score", "y": "Count"},
            title="Per-Episode Quality Scores",
        )
        fig.add_vline(
            x=qual_data["mean"],
            line_dash="dash",
            line_color="green",
            annotation_text=f"Mean: {qual_data['mean']:.3f}",
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        qc1, qc2 = st.columns(2)
        qc1.metric("Mean Quality", f"{qual_data['mean']:.3f}")
        qc2.metric("Std Dev", f"{qual_data['std']:.3f}")

        low_q = profile_obj.quality.low_quality_episodes
        if low_q:
            st.warning(
                f"**{len(low_q)} low-quality episodes detected:** {low_q}\n\n"
                "These episodes may hurt policy performance. "
                "Consider removing or re-collecting."
            )
    else:
        st.info("No quality scores available.")

    # ------------------------------------------------------------------
    # Section 6: Prescriptions
    # ------------------------------------------------------------------
    st.subheader("Prescriptions")
    prescriptions = data.get("prescription_table", [])
    if prescriptions:
        rx_df = pd.DataFrame(prescriptions)
        display_cols = [
            c
            for c in [
                "priority",
                "task",
                "instruction",
                "estimated_demos",
                "current_capability",
                "target_capability",
            ]
            if c in rx_df.columns
        ]
        st.dataframe(rx_df[display_cols], use_container_width=True, hide_index=True)

        # Export buttons
        ec1, ec2, ec3 = st.columns(3)
        ec1.download_button(
            "Download JSON",
            data=json.dumps(prescriptions, indent=2),
            file_name="profiler_prescriptions.json",
            mime="application/json",
        )

        csv_buf = io.StringIO()
        if display_cols:
            writer = csv.DictWriter(csv_buf, fieldnames=display_cols)
            writer.writeheader()
            for p in prescriptions:
                writer.writerow({k: p.get(k, "") for k in display_cols})
        ec2.download_button(
            "Download CSV",
            data=csv_buf.getvalue(),
            file_name="profiler_prescriptions.csv",
            mime="text/csv",
        )

        md_lines = ["# Dataset Profiler — Prescriptions\n"]
        for p in prescriptions:
            md_lines.append(f"## [{p.get('priority', '?')}] {p.get('task', 'N/A')}")
            md_lines.append(p.get("instruction", "") + "\n")
        ec3.download_button(
            "Download Markdown",
            data="\n".join(md_lines),
            file_name="profiler_prescriptions.md",
            mime="text/markdown",
        )
    else:
        st.success("No prescriptions — dataset looks good!")

    # ------------------------------------------------------------------
    # Section 7: Comparison Mode
    # ------------------------------------------------------------------
    if "profiler_profile_b" in st.session_state:
        st.subheader("Dataset Comparison")
        profile_b = st.session_state["profiler_profile_b"]
        data_b = st.session_state["profiler_dashboard_data_b"]

        try:
            from orbit.profile.capability import CapabilityScorer

            scorer = CapabilityScorer()
            comp = scorer.compare_profiles(profile_obj, profile_b)
        except Exception as e:
            st.error(f"Comparison failed: {e}")
            return

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Overlap Score", f"{comp['overlap']:.3f}")
        cc2.metric("Combined Coverage", f"{comp['combined_coverage']:.3f}")
        cc3.metric(
            "Unique Tasks",
            f"A: {len(comp['unique_to_a'])} | B: {len(comp['unique_to_b'])}",
        )

        # Side-by-side stats
        stats_b = data_b["summary_stats"]
        side1, side2 = st.columns(2)
        with side1:
            st.markdown("**Dataset A**")
            st.write(f"Episodes: {stats['num_episodes']} | Frames: {stats['num_frames']}")
            st.write(f"Coverage: {stats['overall_coverage']:.3f}")
        with side2:
            st.markdown("**Dataset B**")
            st.write(f"Episodes: {stats_b['num_episodes']} | Frames: {stats_b['num_frames']}")
            st.write(f"Coverage: {stats_b['overall_coverage']:.3f}")

        if comp["unique_to_a"]:
            st.write(f"**A is strong where B is weak:** {', '.join(comp['unique_to_a'])}")
        if comp["unique_to_b"]:
            st.write(f"**B is strong where A is weak:** {', '.join(comp['unique_to_b'])}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5: Settings
# ═══════════════════════════════════════════════════════════════════════════


def page_settings() -> None:
    """Settings — configure detectors, VLM, embeddings, workspace."""
    st.header("Settings")

    # --- Detector thresholds ---
    st.subheader("Detector Thresholds")
    c1, c2 = st.columns(2)
    with c1:
        st.number_input(
            "Stall velocity threshold",
            value=st.session_state.get("stall_velocity_threshold", 0.001),
            format="%.4f",
            key="stall_velocity_threshold",
        )
        st.number_input(
            "Stall min frames",
            value=st.session_state.get("stall_min_frames", 10),
            min_value=1,
            key="stall_min_frames",
        )
        st.number_input(
            "Gripper closed threshold",
            value=st.session_state.get("gripper_closed_threshold", 0.3),
            format="%.2f",
            key="gripper_closed_threshold",
        )
    with c2:
        st.number_input(
            "Timeout max duration (s)",
            value=st.session_state.get("timeout_max_duration", 60.0),
            format="%.1f",
            key="timeout_max_duration",
        )
        st.number_input(
            "Timeout max frames",
            value=st.session_state.get("timeout_max_frames", 1000),
            min_value=1,
            key="timeout_max_frames",
        )
        st.number_input(
            "Min total reward threshold",
            value=st.session_state.get("min_reward_threshold", 0.0),
            format="%.2f",
            key="min_reward_threshold",
        )

    # --- VLM settings ---
    st.subheader("VLM Configuration")
    st.text_input(
        "VLM API Key",
        value=st.session_state.get("vlm_api_key", ""),
        type="password",
        key="vlm_api_key",
        help="Required for AI failure descriptions. OpenCLIP runs locally.",
    )
    st.selectbox(
        "VLM Model",
        ["ViT-B-32 (laion2b)", "ViT-L-14 (laion2b)", "ViT-H-14 (laion2b)"],
        key="vlm_model",
    )

    # --- Embedding model ---
    st.subheader("Embedding Model")
    st.selectbox(
        "Model",
        [
            "google/siglip-base-patch16-224 (recommended)",
            "ViT-B-32 (OpenCLIP)",
            "ViT-L-14 (OpenCLIP)",
        ],
        key="embedding_model",
    )
    st.selectbox("Device", ["cpu", "cuda", "mps"], key="embedding_device")

    # --- Workspace bounds ---
    st.subheader("Workspace Bounds (Joint Limits)")
    st.caption("Configure per-joint position limits for the OutOfBoundsDetector.")
    n_joints = st.number_input("Number of joints", value=6, min_value=1, max_value=12)

    lower_limits: list[float] = []
    upper_limits: list[float] = []
    cols = st.columns(int(n_joints))
    for j, col in enumerate(cols):
        with col:
            lower = st.number_input(
                f"J{j} lower",
                value=st.session_state.get(f"joint_{j}_lower", -3.14),
                format="%.3f",
                key=f"joint_{j}_lower",
            )
            upper = st.number_input(
                f"J{j} upper",
                value=st.session_state.get(f"joint_{j}_upper", 3.14),
                format="%.3f",
                key=f"joint_{j}_upper",
            )
            lower_limits.append(lower)
            upper_limits.append(upper)

    # --- Save settings ---
    st.divider()
    if st.button("Save Settings to YAML"):
        _save_settings_yaml(lower_limits, upper_limits)


def _save_settings_yaml(lower_limits: list[float], upper_limits: list[float]) -> None:
    """Save current settings to orbit_settings.yaml in the data dir."""
    import yaml

    data_dir = st.session_state.get("data_dir", "./orbit_data")
    settings = {
        "detectors": {
            "stall": {
                "velocity_threshold": st.session_state.get("stall_velocity_threshold", 0.001),
                "min_stall_frames": st.session_state.get("stall_min_frames", 10),
            },
            "gripper_drop": {
                "closed_threshold": st.session_state.get("gripper_closed_threshold", 0.3),
            },
            "timeout": {
                "max_duration_seconds": st.session_state.get("timeout_max_duration", 60.0),
                "max_frames": st.session_state.get("timeout_max_frames", 1000),
            },
            "reward_threshold": {
                "min_total_reward": st.session_state.get("min_reward_threshold", 0.0),
            },
            "out_of_bounds": {
                "joint_limits_lower": lower_limits,
                "joint_limits_upper": upper_limits,
            },
        },
        "vlm": {
            "model": st.session_state.get("vlm_model", "ViT-B-32 (laion2b)"),
        },
        "embedding": {
            "model": st.session_state.get("embedding_model", "google/siglip-base-patch16-224"),
            "device": st.session_state.get("embedding_device", "cpu"),
        },
    }

    out_path = Path(data_dir) / "orbit_settings.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.dump(settings, default_flow_style=False))
    st.success(f"Settings saved to `{out_path}`")


# ═══════════════════════════════════════════════════════════════════════════
# App entry (Streamlit execution)
# ═══════════════════════════════════════════════════════════════════════════


def _run_app() -> None:
    """Set up the multi-page Streamlit app."""
    st.set_page_config(
        page_title="Orbit Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_sidebar()

    pages = [
        st.Page(page_session_overview, title="Session Overview"),
        st.Page(page_distribution_view, title="Distribution View"),
        st.Page(page_failure_analysis, title="Failure Analysis"),
        st.Page(page_prescriptions, title="Prescriptions"),
        st.Page(page_dataset_profiler, title="Dataset Profiler"),
        st.Page(page_settings, title="Settings"),
    ]
    nav = st.navigation(pages)
    nav.run()


# Streamlit executes the module directly
if __name__ == "__main__":
    _run_app()
else:
    # When Streamlit imports the module via `streamlit run`
    try:
        _ctx = st.runtime.scriptrunner.get_script_run_ctx()
        if _ctx is not None:
            _run_app()
    except Exception:
        pass
