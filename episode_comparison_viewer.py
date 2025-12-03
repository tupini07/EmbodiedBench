#!/usr/bin/env python3
"""
Streamlit viewer for episode-level success/failure comparisons.

Shows episodes where target model succeeds and baseline fails, with side-by-side
comparison of best target reasoning vs worst baseline reasoning.

Run:
    streamlit run episode_comparison_viewer.py

Prerequisites:
1. Ingest logs and score steps (existing pipeline)
2. Ingest episode outcomes:
    python -m sample_extractor.ingest_outcomes --base-dir running --tasks eb_alfred eb_habitat
3. Generate comparisons:
    python -m sample_extractor.generate_comparisons \
        --target-runs "20251109-*" \
        --baseline-runs "Qwen*" \
        --pattern success_vs_failure
"""

import os
import sqlite3
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

# DB path
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

def _get_conn():
    return sqlite3.connect(DB_PATH)

def fetch_tasks(conn) -> List[str]:
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT task FROM comparison_candidates ORDER BY task")
    return [r[0] for r in cur.fetchall()]

def fetch_run_pairs(conn, task: str) -> List[tuple]:
    """Get unique (target_run, baseline_run) pairs for a task."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT target_run_id, baseline_run_id
        FROM comparison_candidates
        WHERE task = ?
        ORDER BY target_run_id, baseline_run_id
        """,
        (task,),
    )
    return cur.fetchall()

def fetch_comparisons(
    conn,
    task: str,
    target_run: Optional[str] = None,
    baseline_run: Optional[str] = None,
    min_contrast: float = 0.0,
) -> pd.DataFrame:
    """Fetch comparison candidates with filters."""
    cur = conn.cursor()
    
    params: List = [task]
    filters = ["task = ?"]
    
    if target_run:
        filters.append("target_run_id = ?")
        params.append(target_run)
    if baseline_run:
        filters.append("baseline_run_id = ?")
        params.append(baseline_run)
    if min_contrast > 0:
        filters.append("contrast_score >= ?")
        params.append(min_contrast)
    
    where_clause = " AND ".join(filters)
    
    cur.execute(
        f"""
        SELECT 
            id, task, episode, target_run_id, baseline_run_id,
            target_best_step, target_best_quality,
            baseline_worst_step, baseline_worst_quality,
            baseline_has_hallucination, contrast_score
        FROM comparison_candidates
        WHERE {where_clause}
        ORDER BY contrast_score DESC
        """,
        params,
    )
    
    cols = [
        "id", "task", "episode", "target_run_id", "baseline_run_id",
        "target_best_step", "target_best_quality",
        "baseline_worst_step", "baseline_worst_quality",
        "baseline_has_hallucination", "contrast_score",
    ]
    
    return pd.DataFrame(cur.fetchall(), columns=cols)

def fetch_step_details(conn, run_id: str, episode: int, step: int) -> Optional[Dict]:
    """Fetch full step details including scores and reasoning."""
    cur = conn.cursor()
    
    # Get step data
    cur.execute(
        """
        SELECT s.prompt, s.raw_output, s.retries_before_success,
               sc.final_quality, sc.explanation, sc.hallucination, sc.overly_verbose,
               sc.tightness, sc.task_relevance, sc.logical_progression, sc.fidelity,
               sc.efficiency, sc.insight, sc.structure, sc.robustness
        FROM steps s
        LEFT JOIN scores sc ON sc.run_id = s.run_id 
            AND sc.episode = s.episode 
            AND sc.step = s.step
        WHERE s.run_id = ? AND s.episode = ? AND s.step = ?
        LIMIT 1
        """,
        (run_id, episode, step),
    )
    
    row = cur.fetchone()
    if not row:
        return None
    
    return {
        "prompt": row[0],
        "raw_output": row[1],
        "retries": row[2],
        "final_quality": row[3] or 0,
        "explanation": row[4] or "",
        "hallucination": bool(row[5]),
        "overly_verbose": bool(row[6]),
        "dimensions": {
            "tightness": row[7] or 0,
            "task_relevance": row[8] or 0,
            "logical_progression": row[9] or 0,
            "fidelity": row[10] or 0,
            "efficiency": row[11] or 0,
            "insight": row[12] or 0,
            "structure": row[13] or 0,
            "robustness": row[14] or 0,
        },
    }

def fetch_episode_outcome(conn, run_id: str, task: str, episode: int) -> Optional[Dict]:
    """Fetch episode outcome details."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT task_success, task_progress, num_invalid_actions, 
               num_steps, instruction
        FROM episode_outcomes
        WHERE run_id = ? AND task = ? AND episode = ?
        """,
        (run_id, task, episode),
    )
    
    row = cur.fetchone()
    if not row:
        return None
    
    return {
        "task_success": row[0],
        "task_progress": row[1],
        "num_invalid_actions": row[2],
        "num_steps": row[3],
        "instruction": row[4],
    }

def color_score(v: int, max_v: int = 10) -> str:
    """Color coding for scores."""
    if v is None:
        return "background-color:#eee;"
    ratio = v / max_v
    r = int(255 * (1 - ratio))
    g = int(200 * ratio)
    b = 80
    return f"background-color:rgb({r},{g},{b},0.65);"

def get_image_path(task: str, run_id: str, episode: int, step: int) -> Optional[str]:
    """Construct image path for a step."""
    image_root = os.getenv("EMB_IMAGE_ROOT", "running")
    task_dir_map = {
        "EB-ALFRED": "eb_alfred",
        "EB-Habitat": "eb_habitat",
        "EB-Navigation": "eb_navigation",
    }
    task_dir = task_dir_map.get(task, task.lower())
    
    # The run_id already includes the full directory name
    image_path = f"{image_root}/{task_dir}/{run_id}/base/images/episode_{episode}/episode_{episode}_step_{step}.png"
    
    if os.path.exists(image_path):
        return image_path
    return None

# Streamlit UI
st.set_page_config(page_title="Episode Comparison Viewer", layout="wide")
st.title("🎯 Episode Success/Failure Comparison Viewer")

conn = _get_conn()

# Check if we have data
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM comparison_candidates")
total_comparisons = cur.fetchone()[0]

if total_comparisons == 0:
    st.error("""
    No comparison candidates found. Please run:
    1. `python -m sample_extractor.ingest_outcomes --base-dir running --tasks eb_alfred eb_habitat`
    2. `python -m sample_extractor.generate_comparisons --target-runs "YOUR_MODEL*" --baseline-runs "Qwen*"`
    """)
    st.stop()

st.sidebar.markdown(f"**Total comparisons:** {total_comparisons}")

# Filters
tasks = fetch_tasks(conn)
if not tasks:
    st.error("No tasks found in comparison candidates.")
    st.stop()

task = st.sidebar.selectbox("Task", tasks, index=0)

# Get run pairs for selected task
run_pairs = fetch_run_pairs(conn, task)
if not run_pairs:
    st.warning(f"No comparisons for task: {task}")
    st.stop()

# Extract unique target and baseline runs
target_runs = sorted(set(p[0] for p in run_pairs))
baseline_runs = sorted(set(p[1] for p in run_pairs))

target_run = st.sidebar.selectbox("Target Model Run", target_runs, index=0)
baseline_run = st.sidebar.selectbox("Baseline Model Run", baseline_runs, index=0)

min_contrast = st.sidebar.slider(
    "Minimum Contrast Score",
    min_value=0.0,
    max_value=50.0,
    value=5.0,
    step=1.0,
)

limit = st.sidebar.number_input("Max episodes shown", min_value=1, max_value=200, value=50, step=10)

show_images = st.sidebar.checkbox("Show step images", value=True)
show_raw = st.sidebar.checkbox("Show raw outputs", value=False)

# Fetch comparisons
df = fetch_comparisons(conn, task, target_run, baseline_run, min_contrast)

if df.empty:
    st.warning("No comparisons match the current filters.")
    st.stop()

st.subheader(f"Found {len(df)} Comparisons")
st.caption(f"Target: `{target_run}` vs Baseline: `{baseline_run}`")

# Display comparisons
for idx, row in df.head(limit).iterrows():
    episode = row["episode"]
    contrast_score = row["contrast_score"]
    
    target_step = int(row["target_best_step"])
    target_quality = int(row["target_best_quality"])
    baseline_step = int(row["baseline_worst_step"])
    baseline_quality = int(row["baseline_worst_quality"])
    baseline_halluc = bool(row["baseline_has_hallucination"])
    
    # Fetch episode outcomes
    target_outcome = fetch_episode_outcome(conn, str(target_run), task, episode)
    baseline_outcome = fetch_episode_outcome(conn, str(baseline_run), task, episode)
    
    instruction = target_outcome["instruction"] if target_outcome else ""
    
    # Fetch step details
    target_details = fetch_step_details(conn, str(target_run), episode, target_step)
    baseline_details = fetch_step_details(conn, str(baseline_run), episode, baseline_step)
    
    if not target_details or not baseline_details:
        continue
    
    # Episode header
    with st.expander(
        f"**Episode {episode}** | Contrast: {contrast_score:.1f} | "
        f"Target Q: {target_quality} vs Baseline Q: {baseline_quality}"
        f"{' 🚨' if baseline_halluc else ''}",
        expanded=False,
    ):
        # Instruction
        if instruction:
            st.info(f"📋 **Task:** {instruction}")
        
        # Outcome summary
        col_summary1, col_summary2 = st.columns(2)
        
        with col_summary1:
            st.markdown("### 🟢 Target Model")
            if target_outcome:
                success_pct = target_outcome["task_success"] * 100
                st.markdown(f"**Success:** {success_pct:.0f}% | **Progress:** {target_outcome['task_progress']*100:.0f}%")
                st.markdown(f"**Steps:** {target_outcome['num_steps']} | **Invalid Actions:** {target_outcome['num_invalid_actions']}")
        
        with col_summary2:
            st.markdown("### 🔴 Baseline Model")
            if baseline_outcome:
                success_pct = baseline_outcome["task_success"] * 100
                st.markdown(f"**Success:** {success_pct:.0f}% | **Progress:** {baseline_outcome['task_progress']*100:.0f}%")
                st.markdown(f"**Steps:** {baseline_outcome['num_steps']} | **Invalid Actions:** {baseline_outcome['num_invalid_actions']}")
        
        st.markdown("---")
        
        # Side-by-side step comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 🟢 Best Step: {target_step} (Quality: {target_quality}/10)")
            
            # Image
            if show_images:
                img_path = get_image_path(task, str(target_run), episode, target_step)
                if img_path:
                    st.image(img_path, caption=f"Step {target_step}", use_container_width=True)
            
            # Dimensions
            dim_data = [(k, v) for k, v in target_details["dimensions"].items()]
            dim_df = pd.DataFrame(dim_data, columns=["Dimension", "Score"])
            styled = dim_df.style.apply(
                lambda col: [color_score(v) for v in col] if col.name == "Score" else [""] * len(col),
                axis=0,
            )
            st.dataframe(styled, hide_index=True, use_container_width=True)
            
            # Explanation
            if target_details["explanation"]:
                st.caption(f"**Explanation:** {target_details['explanation']}")
            
            # Raw output
            if show_raw and target_details["raw_output"]:
                with st.expander("Raw Output"):
                    st.code(target_details["raw_output"], language="text")
        
        with col2:
            halluc_marker = " ⚠️ HALLUCINATION" if baseline_details["hallucination"] else ""
            st.markdown(f"### 🔴 Worst Step: {baseline_step} (Quality: {baseline_quality}/10){halluc_marker}")
            
            # Image
            if show_images:
                img_path = get_image_path(task, str(baseline_run), episode, baseline_step)
                if img_path:
                    st.image(img_path, caption=f"Step {baseline_step}", use_container_width=True)
            
            # Dimensions
            dim_data = [(k, v) for k, v in baseline_details["dimensions"].items()]
            dim_df = pd.DataFrame(dim_data, columns=["Dimension", "Score"])
            styled = dim_df.style.apply(
                lambda col: [color_score(v) for v in col] if col.name == "Score" else [""] * len(col),
                axis=0,
            )
            st.dataframe(styled, hide_index=True, use_container_width=True)
            
            # Explanation
            if baseline_details["explanation"]:
                st.caption(f"**Explanation:** {baseline_details['explanation']}")
            
            # Flags
            flags = []
            if baseline_details["hallucination"]:
                flags.append("⚠️ Hallucination")
            if baseline_details["overly_verbose"]:
                flags.append("🗣 Overly verbose")
            if flags:
                st.warning(" | ".join(flags))
            
            # Raw output
            if show_raw and baseline_details["raw_output"]:
                with st.expander("Raw Output"):
                    st.code(baseline_details["raw_output"], language="text")

st.markdown("---")
st.info("💡 **Tip:** Adjust the minimum contrast score to filter for the most compelling examples. Higher scores indicate clearer success/failure contrasts.")
