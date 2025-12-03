# Episode-Based Comparison Workflow

This workflow helps identify compelling qualitative examples where your target model succeeds and baseline models fail.

## Overview

Instead of comparing step-by-step (which fails when trajectories diverge), this approach:

1. **Identifies success/failure episodes**: Episodes where target model succeeds (task_success=1.0) and baseline fails (task_success<1.0)
2. **Finds best moments**: For each comparison, selects the best reasoning step from target and worst/failure step from baseline
3. **Ranks by contrast**: Computes contrast scores based on quality gaps, hallucinations, and invalid actions
4. **Displays side-by-side**: Shows episode outcomes and key reasoning moments in an easy-to-navigate viewer

## Quick Start

### 1. Ingest Episode Outcomes

Parse the `episode_*_final_res.json` files from your runs:

```bash
python -m sample_extractor.ingest_outcomes \
    --base-dir running \
    --tasks eb_alfred eb_habitat
```

This populates the `episode_outcomes` table with task success rates, progress, and instructions.

### 2. Generate Comparison Candidates

Find episodes where target succeeds and baseline fails:

```bash
python -m sample_extractor.generate_comparisons \
    --target-runs "20251109-total-gated*" \
    --baseline-runs "Qwen*" \
    --pattern success_vs_failure \
    --min-contrast 5.0
```

**Parameters:**
- `--target-runs`: Glob pattern for your model's run_ids
- `--baseline-runs`: Glob pattern for baseline model run_ids  
- `--pattern`: `success_vs_failure` (target succeeds, baseline fails) or `all`
- `--min-contrast`: Minimum contrast score to include (optional)
- `--tasks`: Filter by specific tasks like `EB-ALFRED` (optional)

This generates comparison candidates with contrast scores based on:
- Quality gap between target's best and baseline's worst step
- Presence of hallucinations in baseline (+10 points)
- Number of invalid actions by baseline (+2 points each)
- Success differential bonus (+5 points)

### 3. View Comparisons

Launch the Streamlit viewer:

```bash
streamlit run episode_comparison_viewer.py
```

The viewer shows:
- **Filters**: Task, target/baseline runs, minimum contrast score
- **Episode list**: Sorted by contrast score (highest = most compelling)
- **Side-by-side comparison**: 
  - Target model's best reasoning step (highest quality)
  - Baseline model's failure step (hallucination or lowest quality)
  - Episode outcomes (success %, progress, invalid actions)
  - Task instruction
  - Step images (if available)
  - Quality dimension scores
  - Raw reasoning outputs

## Database Schema

### episode_outcomes
Stores episode-level results from `episode_*_final_res.json`:
```sql
CREATE TABLE episode_outcomes (
    run_id TEXT,
    task TEXT,
    episode INTEGER,
    task_success REAL,
    task_progress REAL,
    num_invalid_actions INTEGER,
    num_steps INTEGER,
    instruction TEXT,
    PRIMARY KEY(run_id, task, episode)
);
```

### comparison_candidates
Stores identified success/failure contrasts:
```sql
CREATE TABLE comparison_candidates (
    task TEXT,
    episode INTEGER,
    target_run_id TEXT,
    baseline_run_id TEXT,
    target_best_step INTEGER,
    target_best_quality INTEGER,
    baseline_worst_step INTEGER,
    baseline_worst_quality INTEGER,
    baseline_has_hallucination INTEGER,
    contrast_score REAL,
    PRIMARY KEY(task, episode, target_run_id, baseline_run_id)
);
```

## Complete Pipeline

Assuming you've already ingested logs and scored steps:

```bash
# 1. Ingest episode outcomes (once per data update)
python -m sample_extractor.ingest_outcomes \
    --base-dir running \
    --tasks eb_alfred eb_habitat

# 2. Generate comparisons (can re-run with different filters)
python -m sample_extractor.generate_comparisons \
    --target-runs "YOUR_MODEL*" \
    --baseline-runs "Qwen*" \
    --pattern success_vs_failure \
    --min-contrast 10.0 \
    --tasks EB-ALFRED

# 3. View and export examples
streamlit run episode_comparison_viewer.py
```

## Tips for Finding Compelling Examples

1. **Start with high contrast scores**: Use `--min-contrast 10.0` or higher to filter for the clearest differences

2. **Look for hallucinations**: Baseline hallucinations are automatically prioritized and flagged with 🚨

3. **Check invalid actions**: Episodes where baseline has many invalid actions show clear failure modes

4. **Adjust filters in viewer**: The Streamlit UI lets you interactively explore different model pairs and contrast thresholds

5. **Export key examples**: Take screenshots or export raw outputs for paper figures

## Advantages Over Step-by-Step Comparison

✅ **Handles trajectory divergence**: Doesn't require same step numbers  
✅ **Focuses on outcomes**: Success vs failure, not arbitrary step alignment  
✅ **Surfaces clear contrasts**: Automatically finds best target moments vs worst baseline moments  
✅ **Efficient exploration**: Sorted by contrast score, most compelling examples first  
✅ **Paper-ready**: Shows clear "our model works, baseline fails" narratives  

## Example Output

```
Episode 42 | Contrast: 23.5 | Target Q: 9 vs Baseline Q: 2 🚨

Task: "Rinse off a ladle and move it to the table"

Target Model ✅
  Success: 100% | Progress: 100%
  Best Step 8: "I can see the ladle in the sink. I'll pick it up..."
  Quality: 9/10 | Fidelity: 9 | Task Relevance: 10

Baseline Model ❌
  Success: 0% | Progress: 33%
  Failure Step 5: "The ladle is on the table, I'll pick it up..."
  Quality: 2/10 | Fidelity: 1 | ⚠️ Hallucination
  (Object not actually visible - hallucination)
```

## Troubleshooting

**No comparisons found:**
- Ensure episode outcomes are ingested: `python -m sample_extractor.ingest_outcomes`
- Check run_id patterns match your data
- Lower `--min-contrast` threshold
- Verify scores exist for the episodes (run `sample_extractor.rank_missing` if needed)

**Type errors in viewer:**
- Update to latest code with type fixes

**Missing images:**
- Set `EMB_IMAGE_ROOT` environment variable if images are in non-standard location
- Images should be at: `{root}/{task_dir}/{model}_{run_id}/base/images/episode_{N}/episode_{N}_step_{M}.png`
