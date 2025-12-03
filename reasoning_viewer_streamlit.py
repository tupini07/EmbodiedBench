#!/usr/bin/env python3
"""
Streamlit viewer for EmbodiedBench reasoning comparisons across models.

Run:
    streamlit run reasoning_viewer_streamlit.py

Prerequisites:
1. Ingest logs:
    python -m sample_extractor.ingest --runs 20251109-total-gated-combined-5-rl-step-55temp0.6_maxTokens8192_rep1 --tasks EB-ALFRED
2. Score missing steps (populate scores table):
    python -m sample_extractor.rank_missing --runs 20251109-total-gated-combined-5-rl-step-55temp0.6_maxTokens8192_rep1 --limit 500
3. (Optional) Recompute divergence metrics:
    python -c "from sample_extractor.db import get_conn, compute_divergence; c=get_conn('outputs/steps_cache.sqlite'); compute_divergence(c)"

Features:
- Filter by task, subtask, run_ids, episode range, minimum model count
- Sort episodes/steps by divergence metrics (range/stddev) or max/min/avg quality
- Side-by-side model reasoning outputs with intermediate dimension scores
- Highlight best and worst final_quality
- Flag indicators (hallucination, overly verbose)
- Export selected comparisons to JSON
"""

import json
import math
import sqlite3
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

# Robust DB path resolution without requiring secrets.toml.
try:
    DB_PATH = st.secrets.get("embodiedbench_db", "outputs/steps_cache.sqlite")  # type: ignore
except Exception:
    DB_PATH = os.getenv("EMBODIEDBENCH_DB", "outputs/steps_cache.sqlite")

DIMENSIONS = [
    "tightness",
    "task_relevance",
    "logical_progression",
    "fidelity",
    "efficiency",
    "insight",
    "structure",
    "robustness",
]

# -----------------------
# Data access
# -----------------------

def _get_conn():
    return sqlite3.connect(DB_PATH)

def fetch_tasks(conn) -> List[str]:
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT task FROM steps ORDER BY task")
    return [r[0] for r in cur.fetchall()]

def fetch_subtasks(conn, task: str) -> List[str]:
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT subtask FROM steps WHERE task=? ORDER BY subtask", (task,))
    return [r[0] for r in cur.fetchall()]

def fetch_run_ids(conn, task: str, subtask: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT run_id FROM steps WHERE task=? AND subtask=? ORDER BY run_id",
        (task, subtask),
    )
    return [r[0] for r in cur.fetchall()]

def fetch_episode_bounds(conn, task: str, subtask: str, run_ids: List[str]) -> Tuple[int, int]:
    cur = conn.cursor()
    if run_ids:
        q = ",".join("?" for _ in run_ids)
        cur.execute(
            f"""
            SELECT MIN(episode), MAX(episode) FROM steps
            WHERE task=? AND subtask=? AND run_id IN ({q})
            """,
            (task, subtask, *run_ids),
        )
    else:
        cur.execute(
            "SELECT MIN(episode), MAX(episode) FROM steps WHERE task=? AND subtask=?",
            (task, subtask),
        )
    row = cur.fetchone()
    return (row[0] or 0, row[1] or 0)

def fetch_scores(conn, task: str, subtask: str, run_ids: List[str], ep_lo: int, ep_hi: int):
    cur = conn.cursor()
    params = [task, subtask, ep_lo, ep_hi]
    run_filter_clause = ""
    if run_ids:
        run_filter_clause = "AND s.run_id IN ({})".format(",".join("?" for _ in run_ids))
        params.extend(run_ids)
    cur.execute(
        f"""
        SELECT
            s.run_id, s.model_name, s.task, s.subtask,
            sc.episode, sc.step,
            sc.final_quality,
            sc.tightness, sc.task_relevance, sc.logical_progression, sc.fidelity,
            sc.efficiency, sc.insight, sc.structure, sc.robustness,
            sc.explanation, sc.hallucination, sc.overly_verbose,
            sc.raw_output_len,
            s.prompt,
            s.raw_output, s.retries_before_success, s.log_path
        FROM scores sc
        JOIN steps s
          ON sc.run_id = s.run_id
         AND sc.model_name = s.model_name
         AND sc.episode = s.episode
         AND sc.step = s.step
        WHERE s.task=? AND s.subtask=?
          AND sc.episode BETWEEN ? AND ?
          {run_filter_clause}
        ORDER BY sc.episode, sc.step, s.model_name
        """,
        params,
    )
    cols = [
        "run_id", "model_name", "task", "subtask",
        "episode", "step", "final_quality",
        *DIMENSIONS, "explanation", "hallucination", "overly_verbose",
        "raw_output_len", "prompt", "raw_output", "retries_before_success", "log_path",
    ]
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)

def fetch_divergence(conn, task: str, ep_lo: int, ep_hi: int):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT run_group, episode, step, model_count,
               max_quality, min_quality, quality_range, stddev_quality
        FROM divergence_metrics
        WHERE run_group LIKE ? AND episode BETWEEN ? AND ?
        ORDER BY episode, step
        """,
        (f"{task}|%", ep_lo, ep_hi),
    )
    return pd.DataFrame(
        cur.fetchall(),
        columns=[
            "run_group", "episode", "step", "model_count",
            "max_quality", "min_quality", "quality_range", "stddev_quality",
        ],
    )

# -----------------------
# Utility
# -----------------------

def color_score(v: int, max_v: int = 10) -> str:
    if v is None:
        return "background-color:#eee;"
    ratio = v / max_v
    # green to red gradient (invert so high = green)
    r = int(255 * (1 - ratio))
    g = int(200 * ratio)
    b = 80
    return f"background-color:rgb({r},{g},{b},0.65);"

def format_flags(row) -> str:
    flags = []
    if row["hallucination"]:
        flags.append("⚠️ hallucination")
    if row["overly_verbose"]:
        flags.append("🗣 verbose")
    return ", ".join(flags) if flags else "—"

def build_step_groups(df: pd.DataFrame) -> Dict[Tuple[int, int], pd.DataFrame]:
    groups: Dict[Tuple[int, int], pd.DataFrame] = {}
    for (ep, st), subdf in df.groupby(["episode", "step"]):
        groups[(ep, st)] = subdf
    return groups

def extract_episode_instruction(log_path: str, episode: int) -> str:
    """Extract the instruction for a given episode from the log file."""
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            current_episode = -1
            for line in f:
                # Check for episode markers in planner output
                if f"Episode:{episode}" in line:
                    current_episode = episode
                # Look for instruction line when we're at the right episode
                if current_episode == episode and line.strip().startswith("Instruction:"):
                    return line.strip().replace("Instruction:", "").strip()
    except (FileNotFoundError, IOError):
        return ""
    return ""

# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(page_title="EmbodiedBench Reasoning Viewer", layout="wide")
st.title("🧠 EmbodiedBench Reasoning Quality Viewer")

conn = _get_conn()

tasks = fetch_tasks(conn)
if not tasks:
    st.error("No tasks found. Ensure ingestion and scoring have been run.")
    st.stop()

task = st.sidebar.selectbox("Task", tasks, index=0)
subtasks = fetch_subtasks(conn, task)
subtask = st.sidebar.selectbox("Subtask", subtasks, index=0)

run_ids_all = fetch_run_ids(conn, task, subtask)
selected_runs = st.sidebar.multiselect("Run IDs (optional filter)", run_ids_all, default=run_ids_all[:5])

ep_min, ep_max = fetch_episode_bounds(conn, task, subtask, selected_runs)
episode_range = st.sidebar.slider("Episode range", ep_min, ep_max, (ep_min, ep_max), step=1)

# Step filter
st.sidebar.markdown("**Step Filter**")
filter_specific_steps = st.sidebar.checkbox("Filter by specific steps", value=False)
specific_steps = []
if filter_specific_steps:
    step_input = st.sidebar.text_input(
        "Steps (comma-separated, e.g., 0,1,5)",
        value="0",
        help="Enter step numbers separated by commas"
    )
    try:
        specific_steps = [int(s.strip()) for s in step_input.split(",") if s.strip()]
    except ValueError:
        st.sidebar.error("Invalid step numbers. Use comma-separated integers.")
        specific_steps = []

min_models = st.sidebar.number_input("Minimum model count per step", min_value=1, max_value=20, value=2)

# Target model gap analysis
st.sidebar.markdown("---")
st.sidebar.markdown("**Gap Analysis**")
target_run_id = st.sidebar.selectbox(
    "Target Run ID (for gap sorting)",
    options=selected_runs if selected_runs else run_ids_all,
    index=0 if (selected_runs or run_ids_all) else None,
    help="Select the model to compare against others. Gap = target_score - avg(others)"
)
st.sidebar.markdown("---")

sort_mode = st.sidebar.selectbox(
    "Sort steps by",
    [
        "episode_step",
        "quality_range_desc",
        "stddev_desc",
        "max_quality_desc",
        "min_quality_asc",
        "avg_quality_desc",
        "target_gap_desc",
        "target_gap_asc",
    ],
    index=1,
)

show_raw = st.sidebar.checkbox("Show raw outputs", value=True)
show_prompt_stats = st.sidebar.checkbox("Show explanation / flags", value=True)
limit_steps = st.sidebar.number_input("Max steps shown", min_value=1, max_value=2000, value=200, step=10)

refresh = st.sidebar.button("🔄 Refresh")

if refresh:
    # Streamlit >=1.32 uses st.rerun (experimental_rerun removed)
    st.rerun()

df_scores = fetch_scores(conn, task, subtask, selected_runs, episode_range[0], episode_range[1])
if df_scores.empty:
    st.warning("No scored steps in selection.")
    st.stop()

# Compute per-step divergence (local if divergence_metrics not present or incomplete)
divergence_df = fetch_divergence(conn, task, episode_range[0], episode_range[1])
have_divergence = not divergence_df.empty
if have_divergence:
    divergence_map = {
        (row.episode, row.step): row for row in divergence_df.itertuples()
    }
else:
    # local computation
    divergence_map = {}
    for (ep, stepd), g in df_scores.groupby(["episode", "step"]):
        quals = g["final_quality"].fillna(0).tolist()
        if not quals:
            continue
        model_count = len(quals)
        max_q = max(quals)
        min_q = min(quals)
        rng = max_q - min_q
        mean = sum(quals) / model_count
        stddev = math.sqrt(sum((q - mean) ** 2 for q in quals) / model_count) if model_count else 0.0
        divergence_map[(ep, stepd)] = {
            "episode": ep,
            "step": stepd,
            "model_count": model_count,
            "max_quality": max_q,
            "min_quality": min_q,
            "quality_range": rng,
            "stddev_quality": stddev,
        }

step_groups = build_step_groups(df_scores)

# Build list of step descriptors for sorting/filtering
step_rows = []
for (ep, step_idx), g in step_groups.items():
    # Apply step filter if enabled
    if filter_specific_steps and specific_steps and step_idx not in specific_steps:
        continue
    
    model_count = g["run_id"].nunique()
    if model_count < min_models:
        continue
    div = divergence_map.get((ep, step_idx))
    if div is None:
        continue
    avg_quality = g["final_quality"].mean()
    # Support divergence_map values as dict (local compute) or namedtuple (DB fetch).
    if hasattr(div, "max_quality"):
        max_quality = div.max_quality
        min_quality = div.min_quality
        quality_range = div.quality_range
        stddev_quality = div.stddev_quality
    else:
        max_quality = div["max_quality"]
        min_quality = div["min_quality"]
        quality_range = div["quality_range"]
        stddev_quality = div["stddev_quality"]

    # Compute gap for target model
    target_gap = 0.0
    if target_run_id:
        target_rows = g[g["run_id"] == target_run_id]
        if not target_rows.empty:
            target_score = target_rows["final_quality"].iloc[0]
            other_scores = g[g["run_id"] != target_run_id]["final_quality"]
            if not other_scores.empty:
                avg_others = other_scores.mean()
                target_gap = float(target_score - avg_others)

    step_rows.append({
        "episode": ep,
        "step": step_idx,
        "model_count": model_count,
        "max_quality": max_quality,
        "min_quality": min_quality,
        "quality_range": quality_range,
        "stddev_quality": stddev_quality,
        "avg_quality": avg_quality,
        "target_gap": target_gap,
    })

steps_df = pd.DataFrame(step_rows)
if steps_df.empty:
    st.warning("No steps after filters (model count threshold may be too high).")
    st.stop()

if sort_mode == "episode_step":
    steps_df = steps_df.sort_values(["episode", "step"])
elif sort_mode == "quality_range_desc":
    steps_df = steps_df.sort_values(["quality_range", "stddev_quality"], ascending=[False, False])
elif sort_mode == "stddev_desc":
    steps_df = steps_df.sort_values(["stddev_quality", "quality_range"], ascending=[False, False])
elif sort_mode == "max_quality_desc":
    steps_df = steps_df.sort_values(["max_quality"], ascending=False)
elif sort_mode == "min_quality_asc":
    steps_df = steps_df.sort_values(["min_quality"], ascending=True)
elif sort_mode == "avg_quality_desc":
    steps_df = steps_df.sort_values(["avg_quality"], ascending=False)
elif sort_mode == "target_gap_desc":
    steps_df = steps_df.sort_values(["target_gap"], ascending=False)
elif sort_mode == "target_gap_asc":
    steps_df = steps_df.sort_values(["target_gap"], ascending=True)

# Display summary table
st.subheader("Step Divergence Summary")
if target_run_id and sort_mode in ["target_gap_desc", "target_gap_asc"]:
    st.info(f"🎯 Gap Analysis: Showing steps sorted by `{target_run_id}` performance gap vs others")
st.dataframe(
    steps_df.head(limit_steps),
    width="stretch",
)

# Detailed view
st.subheader("Detailed Reasoning Comparison")

export_buffer: List[Dict] = []

shown = 0
for row in steps_df.itertuples():
    if shown >= limit_steps:
        break
    ep, stp = row.episode, row.step
    g = step_groups[(ep, stp)]
    
    # Deduplicate by run_id, keeping the entry with highest final_quality
    # This ensures we only show one column per run_id
    g_deduped = g.sort_values("final_quality", ascending=False).drop_duplicates(subset=["run_id"], keep="first")
    
    models_sorted = g_deduped.sort_values("final_quality", ascending=False)
    max_q = models_sorted["final_quality"].max()
    min_q = models_sorted["final_quality"].min()
    
    # Get task name from the first row (all rows in this group have the same task)
    task_name = g["task"].iloc[0] if not g.empty else task
    
    # Build expander title with gap info if using gap sorting
    gap_str = ""
    if hasattr(row, "target_gap") and sort_mode in ["target_gap_desc", "target_gap_asc"]:
        gap_val = row.target_gap
        gap_indicator = "🔺" if gap_val > 0 else ("🔻" if gap_val < 0 else "➖")
        gap_str = f" | gap={gap_indicator}{gap_val:+.1f}"

    expander = st.expander(
        f"**{task_name}** | Episode {ep} Step {stp} | models={row.model_count} "
        f"range={row.quality_range} stddev={row.stddev_quality:.2f} avg={row.avg_quality:.2f}{gap_str}",
        expanded=False,
    )

    with expander:
        # Extract and display episode instruction (once per step, not per model)
        log_path = g["log_path"].iloc[0] if "log_path" in g.columns and not g.empty else ""
        if log_path:
            instruction = extract_episode_instruction(log_path, ep)
            if instruction:
                st.info(f"📋 **Episode Instruction:** {instruction}")
        
        cols = st.columns(len(models_sorted))
        export_record_step = {
            "episode": int(ep),
            "step": int(stp),
            "models": [],
            "metrics": {
                "model_count": int(row.model_count),
                "quality_range": int(row.quality_range),
                "stddev_quality": float(row.stddev_quality),
                "avg_quality": float(row.avg_quality),
            },
        }
        for idx, m in enumerate(models_sorted.itertuples(index=False)):
            with cols[idx]:
                fq_raw = getattr(m, "final_quality", 0)
                try:
                    fq = int(fq_raw)
                except (TypeError, ValueError):
                    fq = 0
                max_numeric = int(max_q) if pd.notna(max_q) else 0
                min_numeric = int(min_q) if pd.notna(min_q) else 0
                header_color = "🟢" if fq == max_numeric and fq > 0 else ("🔴" if fq == min_numeric and fq > 0 else "⚪")
                model_name = getattr(m, "model_name", "(unknown)")
                run_id_display = getattr(m, "run_id", "(run_id?)")
                st.markdown(f"### {header_color} `{run_id_display}` (final={fq})")

                # Attempt to load and display step image (path pattern):
                # {image_root}/{task_dir}/{model_name}_{run_id}/base/images/episode_{ep}/episode_{ep}_step_{stp}.png
                image_root = os.getenv("EMB_IMAGE_ROOT", "running")
                task_dir_map = {
                    "EB-ALFRED": "eb_alfred",
                    "EB-Habitat": "eb_habitat",
                    "EB-Navigation": "eb_navigation",
                }
                task_dir = task_dir_map.get(task, task.lower())
                run_id_val = getattr(m, "run_id", (selected_runs[0] if selected_runs else ""))
                image_path = f"{image_root}/{task_dir}/{model_name}_{run_id_val}/base/images/episode_{ep}/episode_{ep}_step_{stp}.png"
                if os.path.exists(image_path):
                    st.image(image_path, caption=f"Episode {ep} Step {stp}", width="stretch")

                dim_table = []
                for dim in DIMENSIONS:
                    val_raw = getattr(m, dim, 0)
                    try:
                        val = int(val_raw)
                    except (TypeError, ValueError):
                        val = 0
                    dim_table.append((dim, val))
                dim_df = pd.DataFrame(dim_table, columns=["dimension", "score"])
                styled = dim_df.style.apply(
                    lambda col: [color_score(v) for v in col] if col.name == "score" else [""] * len(col),
                    axis=0,
                )
                st.dataframe(styled, hide_index=True, width="stretch")

                halluc = bool(getattr(m, "hallucination", False))
                overly = bool(getattr(m, "overly_verbose", False))
                retries_val = getattr(m, "retries_before_success", 0)
                explanation_text = getattr(m, "explanation", "")
                raw_text = str(getattr(m, "raw_output", ""))
                prompt_text = str(getattr(m, "prompt", "") or "").strip()
                if prompt_text:
                    with st.expander("Prompt", expanded=False):
                        st.code(prompt_text, language="text", wrap_lines=True)

                if show_prompt_stats:
                    st.caption(
                        f"Retries before success: {int(retries_val or 0)} | Flags: {format_flags({'hallucination': halluc, 'overly_verbose': overly})}"
                    )
                    if explanation_text:
                        st.write(f"Explanation: _{explanation_text}_")

                if show_raw:
                    st.code(raw_text, language="text", wrap_lines=True)

                export_record_step["models"].append({
                    "run_id": run_id_val,
                    "final_quality": fq,
                    "intermediate_scores": {d: int(getattr(m, d, 0) or 0) for d in DIMENSIONS},
                    "explanation": explanation_text,
                    "flags": {"hallucination": halluc, "overly_verbose": overly},
                    "prompt": prompt_text,
                    "raw_output": raw_text,
                    "retries_before_success": int(retries_val or 0),
                })
        export_buffer.append(export_record_step)
    shown += 1

st.markdown("---")
st.subheader("Export")
if st.button("Export shown steps JSON"):
    st.download_button(
        "Download JSON",
        data=json.dumps({"steps": export_buffer}, indent=2),
        file_name="embodiedbench_reasoning_export.json",
        mime="application/json",
    )

st.info("Tip: Increase limit or adjust sort to surface high-divergence reasoning examples quickly.")
