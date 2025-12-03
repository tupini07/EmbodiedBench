#!/usr/bin/env python3
"""
Script to print summary tables of EmbodiedBench results.
Reads all summary.json files from the running/ directory and organizes them by environment.

Requirements:
    pip install tabulate
"""

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tabulate import tabulate
import re
import statistics

import pandas as pd
import yaml
from datetime import datetime


def load_experiment_short_names(config_path: Optional[Path] = None) -> Dict[str, str]:
    """Load short_name mappings from experiment_specs.yaml.
    
    Returns a dictionary mapping prefix -> short_name for experiments that define it.
    """
    if config_path is None:
        script_dir = Path(__file__).parent
        config_path = script_dir / "run_configs" / "experiment_specs.yaml"
    
    if not config_path.exists():
        return {}
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        short_names = {}
        experiments = config.get("experiments", [])
        for exp in experiments:
            prefix = exp.get("prefix", "")
            short_name = exp.get("short_name")
            if prefix and short_name:
                short_names[prefix] = short_name
        
        return short_names
    except Exception as e:
        print(f"Warning: Could not load experiment_specs.yaml: {e}")
        return {}


def apply_short_name(exp_name: str, short_name_map: Dict[str, str]) -> str:
    """Apply short_name mapping if the experiment name starts with a known prefix.
    
    If the experiment name starts with a prefix that has a short_name defined,
    replace the prefix with the short_name. Otherwise, return the original name.
    
    Handles two cases:
    1. Direct match: exp_name starts with prefix
    2. Model-prefixed match: exp_name has "Qwen2.5-VL-7B-Instruct_" + prefix
    
    Adds "__" separator between short_name and remainder if remainder exists.
    """
    # Try direct match first
    for prefix, short_name in short_name_map.items():
        if exp_name.startswith(prefix):
            # Replace the prefix with the short_name
            remainder = exp_name[len(prefix):]
            if remainder:
                return short_name + "__" + remainder
            return short_name
    
    # Try with common model prefix stripped
    model_prefix = "Qwen2.5-VL-7B-Instruct_"
    if exp_name.startswith(model_prefix):
        exp_without_model = exp_name[len(model_prefix):]
        for prefix, short_name in short_name_map.items():
            if exp_without_model.startswith(prefix):
                # Replace the prefix with the short_name, keep model prefix
                remainder = exp_without_model[len(prefix):]
                if remainder:
                    return model_prefix + short_name + "__" + remainder
                return model_prefix + short_name
    
    return exp_name


def load_summary_json(file_path: Path) -> Dict[str, Any]:
    """Load and parse a summary.json file.
    
    Normalizes key names across different environments:
    - 'success_rate' (eb_manipulation) -> 'task_success'
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    
    # Normalize key names for consistency across environments
    if 'success_rate' in data and 'task_success' not in data:
        data['task_success'] = data['success_rate']
    
    return data


def parse_results_structure(running_dir: Path) -> Dict[str, Dict[str, Dict[str, Path]]]:
    """
    Parse the running directory structure.
    
    Handles two directory structures:
    1. Standard: running/{env_name}/{exp_name}/{dimension}/results/summary.json
    2. Nested (eb_manipulation): running/{env_name}/{model_name}/{exp_name}/{dimension}/results/summary.json

    Returns:
        Dict with structure: {env_name: {experiment_name: {dimension: summary_json_path}}}
    """
    # Nested mapping: env -> experiment -> dimension -> summary path
    results: Dict[str, Dict[str, Dict[str, Path]]] = defaultdict(lambda: defaultdict(dict))  # type: ignore

    # Iterate through each environment directory
    for env_dir in running_dir.iterdir():
        if not env_dir.is_dir() or env_dir.name.startswith("."):
            continue

        env_name = env_dir.name

        # Iterate through each experiment directory (or model directory for nested structure)
        for exp_dir in env_dir.iterdir():
            if not exp_dir.is_dir() or exp_dir.name.startswith("."):
                continue

            # Check if this directory contains dimension folders directly
            # or if it's a model directory that contains experiment folders
            has_results = False
            subdirs = list(exp_dir.iterdir())
            
            # Check if any subdirectory has results/summary.json
            for subdir in subdirs:
                if not subdir.is_dir():
                    continue
                test_summary = subdir / "results" / "summary.json"
                test_summary_all = subdir / "results" / "summary_all.json"
                if test_summary.exists() or test_summary_all.exists():
                    has_results = True
                    break
            
            if has_results:
                # Standard structure: exp_dir contains dimensions directly
                exp_name = exp_dir.name
                for dim_dir in exp_dir.iterdir():
                    if not dim_dir.is_dir() or dim_dir.name.startswith("."):
                        continue

                    dimension = dim_dir.name
                    summary_path = dim_dir / "results" / "summary.json"
                    summary_all_path = dim_dir / "results" / "summary_all.json"

                    # Try both summary.json and summary_all.json
                    if summary_path.exists():
                        results[env_name][exp_name][dimension] = summary_path
                    elif summary_all_path.exists():
                        results[env_name][exp_name][dimension] = summary_all_path
            else:
                # Nested structure: exp_dir is a model directory containing experiments
                model_prefix = exp_dir.name
                for nested_exp_dir in exp_dir.iterdir():
                    if not nested_exp_dir.is_dir() or nested_exp_dir.name.startswith("."):
                        continue
                    
                    # Construct experiment name with model prefix
                    exp_name = f"{model_prefix}_{nested_exp_dir.name}"
                    
                    # Iterate through each dimension directory
                    for dim_dir in nested_exp_dir.iterdir():
                        if not dim_dir.is_dir() or dim_dir.name.startswith("."):
                            continue

                        dimension = dim_dir.name
                        summary_path = dim_dir / "results" / "summary.json"
                        summary_all_path = dim_dir / "results" / "summary_all.json"

                        # Try both summary.json and summary_all.json
                        if summary_path.exists():
                            results[env_name][exp_name][dimension] = summary_path
                        elif summary_all_path.exists():
                            results[env_name][exp_name][dimension] = summary_all_path

    return results


def format_value(value: Any) -> str:
    """Format a value for display in the table."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        # Format with 3 decimal places
        return f"{value:.3f}"
    return str(value)


def print_environment_table(
    env_name: str, experiments_data: Dict[str, Dict[str, Path]]
):
    """Print a table for a single environment."""
    print(f"\n## Environment: {env_name.upper()}\n")

    if not experiments_data:
        print("No results found for this environment.\n")
        return

    # Get all unique dimensions and metrics
    all_dimensions = set()
    all_metrics = set()

    for exp_name, dimensions in experiments_data.items():
        all_dimensions.update(dimensions.keys())
        for dim_path in dimensions.values():
            data = load_summary_json(dim_path)
            all_metrics.update(data.keys())

    all_dimensions = sorted(all_dimensions)
    all_metrics = sorted(all_metrics)

    # Print table for each dimension
    for dimension in all_dimensions:
        print(f"\n{'-'*100}")
        print(f"Dimension: {dimension}")
        print(f"{'-'*100}")

        # Collect data for this dimension
        table_data = []
        for exp_name in sorted(experiments_data.keys()):
            if dimension in experiments_data[exp_name]:
                summary_path = experiments_data[exp_name][dimension]
                data = load_summary_json(summary_path)
                table_data.append((exp_name, data))

        if not table_data:
            print("No results for this dimension.\n")
            continue

        # Determine which metrics are present in this dimension
        dimension_metrics = set()
        for _, data in table_data:
            dimension_metrics.update(data.keys())
        dimension_metrics = sorted(dimension_metrics)

        # Use tabulate for better formatting
        headers = ["Experiment"] + dimension_metrics
        rows = []
        for exp_name, data in table_data:
            row = [exp_name] + [
                format_value(data.get(metric)) for metric in dimension_metrics
            ]
            rows.append(row)
        print(tabulate(rows, headers=headers, tablefmt="pipe"))

        print()


def print_compact_table(env_name: str, experiments_data: Dict[str, Dict[str, Path]]):
    """Print a compact table showing key metrics for all dimensions of an environment."""
    print(f"\n## Environment: {env_name.upper()} - COMPACT VIEW\n")

    if not experiments_data:
        print("No results found for this environment.\n")
        return

    # Define key metrics to show (adjust based on what's most important)
    key_metrics = [
        "task_success",
        "task_progress",
        "num_steps",
        "planner_output_error",
        "planner_output_error_ratio",
        "num_invalid_action_ratio",
        "episode_elapsed_seconds",
    ]

    # Get all dimensions
    all_dimensions = set()
    for dimensions in experiments_data.values():
        all_dimensions.update(dimensions.keys())
    all_dimensions = sorted(all_dimensions)

    # For each experiment, create a row with all dimensions
    for exp_name in sorted(experiments_data.keys()):
        print(f"\nExperiment: {exp_name}")
        print(f"{'-'*150}")

        # Use tabulate for better formatting
        headers = ["Dimension"] + key_metrics
        rows = []
        for dimension in all_dimensions:
            if dimension in experiments_data[exp_name]:
                summary_path = experiments_data[exp_name][dimension]
                data = load_summary_json(summary_path)
                row = [dimension] + [
                    format_value(data.get(metric)) for metric in key_metrics
                ]
                rows.append(row)
        print(tabulate(rows, headers=headers, tablefmt="pipe"))

        print()


def print_summary_table(env_name: str, experiments_data: Dict[str, Dict[str, Path]], all_experiments: Optional[Set[str]] = None):
    """Print a summary table showing task_success for all experiments and dimensions, with average invalid action ratio."""
    print(f"\n## Environment: {env_name.upper()} - SUMMARY (Task Success)\n")

    # Get all dimensions across all experiments in this environment
    all_dimensions = set()
    for dimensions in experiments_data.values():
        all_dimensions.update(dimensions.keys())
    all_dimensions = sorted(all_dimensions)

    # If no dimensions found for this environment, but we have all_experiments, still show them
    if not all_dimensions and not all_experiments:
        print("No results found for this environment.\n")
        return
    
    if not all_dimensions:
        print("No dimensions found for this environment.\n")
        return

    # Use all_experiments if provided, otherwise fall back to experiments_data.keys()
    experiments_to_show = all_experiments if all_experiments else set(experiments_data.keys())
    
    # Use tabulate for better formatting
    headers = ["Experiment"] + all_dimensions + ["Avg Invalid Action Ratio"]
    rows = []
    for exp_name in sorted(experiments_to_show):
        row = [exp_name]
        invalid_ratios = []
        for dimension in all_dimensions:
            if exp_name in experiments_data and dimension in experiments_data[exp_name]:
                summary_path = experiments_data[exp_name][dimension]
                data = load_summary_json(summary_path)
                task_success = data.get("task_success")
                row.append(format_value(task_success))
                
                # Collect invalid action ratios for averaging
                invalid_ratio = data.get("num_invalid_action_ratio")
                if invalid_ratio is not None and not math.isnan(invalid_ratio):
                    invalid_ratios.append(invalid_ratio)
            else:
                row.append("-")
        
        # Calculate and add average invalid action ratio
        if invalid_ratios:
            avg_invalid_ratio = sum(invalid_ratios) / len(invalid_ratios)
            row.append(format_value(avg_invalid_ratio))
        else:
            row.append("N/A")
        
        rows.append(row)
    print(tabulate(rows, headers=headers, tablefmt="pipe"))

    print()


def base_experiment_name(exp_name: str) -> str:
    r"""Strip trailing _rep\d+ from an experiment name to get the base experiment identifier.

    If no repetition suffix present, returns the original name.
    """
    return re.sub(r"_rep\d+$", "", exp_name)


def compute_mean_std(values: List[float]) -> Optional[tuple]:
    """Return (mean, std) for a list of numeric values. Uses sample std (statistics.stdev).

    Returns None if list is empty. If only one value, std is 0.0.
    """
    if not values:
        return None
    if len(values) == 1:
        return (values[0], 0.0)
    try:
        return (statistics.mean(values), statistics.stdev(values))
    except statistics.StatisticsError:
        # Fallback in pathological cases
        return (statistics.mean(values), 0.0)


def format_mean_std(pairs: Optional[tuple], percent: bool = True) -> str:
    """Format mean ± std as percentage with two decimals, e.g. 4.67% ± 1.15%.

    If pairs is None -> '-'. If percent False, show raw with 3 decimals.
    """
    if pairs is None:
        return "-"
    mean, std = pairs
    if percent:
        return f"{mean*100:.2f}% ± {std*100:.2f}%"
    else:
        return f"{mean:.3f} ± {std:.3f}"


def print_aggregated_summary_tables(env_name: str, experiments_data: Dict[str, Dict[str, Path]], short_name_map: Optional[Dict[str, str]] = None):
    """Reworked aggregated summary tables.

    New logic:
      1. Build a flat record list of (experiment, cleaned_group, repetition, dimension, task_success, invalid_ratio).
    2. Cleaned group name strips the common leading model prefix and trailing _rep<digits> using regex:
        ^(?:Qwen2\\.5-VL-7B-Instruct_)|_rep\\d+$
         (Designed per user instruction; applies sequentially to remove prefix and repetition suffix.)
      3. For each repetition (original experiment name) compute its average invalid ratio across available dimensions.
      4. Aggregate across repetitions within the same cleaned group: for each dimension compute mean ± std of task_success; for invalid ratio compute mean ± std of the per-repetition averages.
      5. Format task_success as percentage (mean*100 with 2 decimals for mean and std) and invalid ratio as raw value with 3 decimals.

    If pandas is available we use it for clearer grouping; otherwise we fall back to pure-python collections.
    """
    if short_name_map is None:
        short_name_map = {}

    # Collect all dimensions for column ordering
    all_dimensions: Set[str] = set()
    for dims in experiments_data.values():
        all_dimensions.update(dims.keys())
    dimensions_sorted = sorted(all_dimensions)
    if not dimensions_sorted:
        return

    # Regex for cleaning experiment identifiers
    # Use raw string to avoid invalid escape sequence warnings.
    clean_pattern = re.compile(r'^(?:Qwen2\.5-VL-7B-Instruct_)|_rep\d+$')

    def clean_experiment(exp: str) -> str:
        # First apply short_name if available
        exp_with_short = apply_short_name(exp, short_name_map)
        # Then apply the cleaning pattern
        return clean_pattern.sub('', exp_with_short)

    # Build flat records
    records: List[Dict[str, Any]] = []
    for exp_name, dims in experiments_data.items():
        cleaned = clean_experiment(exp_name)
        rep_match = re.search(r'(_rep\d+)$', exp_name)
        rep_id = rep_match.group(1) if rep_match else '_rep0'
        for dimension, path in dims.items():
            data = load_summary_json(path)
            ts = data.get('task_success')
            invalid_ratio = data.get('num_invalid_action_ratio')
            records.append({
                'Experiment': exp_name,
                'Group': cleaned,
                'Repetition': rep_id,
                'Dimension': dimension,
                'task_success': ts if isinstance(ts, (int, float)) and not (isinstance(ts, float) and math.isnan(ts)) else None,
                'num_invalid_action_ratio': invalid_ratio if isinstance(invalid_ratio, (int, float)) and not (isinstance(invalid_ratio, float) and math.isnan(invalid_ratio)) else None,
            })

    if not records:
        return

    headers = ["Experiment"] + dimensions_sorted + ["Avg Invalid Action Ratio"]
    rows: List[List[str]] = []

    # Pandas path
    # pandas is available in this branch; assert for type checkers
    assert pd is not None
    df = pd.DataFrame(records)  # type: ignore[attr-defined]

    # Task success aggregation including count of valid repetitions
    ts_stats_df = (
        df.groupby(['Group', 'Dimension'])['task_success']
            .agg(['count', 'mean', 'std'])
            .reset_index()
    )
    # Fix std for single value
    ts_stats_df['std'] = ts_stats_df.apply(
        lambda r: 0.0 if r['count'] == 1 or pd.isna(r['std']) else r['std'], axis=1
    )
    ts_stats: Dict[Tuple[str, str], Tuple[int, float, float]] = {
        (row.Group, row.Dimension): (int(row['count']), float(row['mean']), float(row['std']))
        for _, row in ts_stats_df.iterrows() if not math.isnan(row['mean'])
    }

    # Invalid action ratio per repetition (average across its dimensions)
    rep_invalid = (
        df.dropna(subset=['num_invalid_action_ratio'])
            .groupby(['Group', 'Repetition'])['num_invalid_action_ratio']
            .mean()
            .reset_index()
    )
    # Aggregate invalid ratio across repetitions
    invalid_stats_df = (
        rep_invalid.groupby('Group')['num_invalid_action_ratio']
                    .agg(['count', 'mean', 'std'])
                    .reset_index()
    )
    invalid_stats_df['std'] = invalid_stats_df.apply(
        lambda r: 0.0 if r['count'] == 1 or pd.isna(r['std']) else r['std'], axis=1
    )
    invalid_stats: Dict[str, Tuple[int, float, float]] = {
        row['Group']: (int(row['count']), float(row['mean']), float(row['std']))
        for _, row in invalid_stats_df.iterrows() if not math.isnan(row['mean'])
    }

    groups = sorted(df['Group'].unique())
    for group in groups:
        row_cells: List[str] = [group]
        for dim in dimensions_sorted:
            key = (group, dim)
            if key in ts_stats:
                count_valid, mean, std = ts_stats[key]
                if count_valid == 0:
                    row_cells.append('no data')
                else:
                    # Show mean ± std regardless of how many reps are present
                    row_cells.append(f"{mean*100:.2f}% ± {std*100:.2f}%")
            else:
                # Entire dimension absent for this group
                row_cells.append('-')
        if group in invalid_stats:
            inv_count, inv_mean, inv_std = invalid_stats[group]
            if inv_count == 0:
                row_cells.append('no data')
            else:
                # Show mean ± std regardless of how many reps are present
                row_cells.append(f"{inv_mean:.3f} ± {inv_std:.3f}")
        else:
            row_cells.append('-')
        rows.append(row_cells)

    print(f"\n## Environment: {env_name.upper()} - AGGREGATED SUMMARY (Reworked Mean ± Std of Task Success)\n" )
    print(tabulate(rows, headers=headers, tablefmt='pipe'))
    print()


def print_combined_summary_table(all_results: Dict[str, Dict[str, Dict[str, Path]]], short_name_map: Optional[Dict[str, str]] = None):
    """Print a single combined table with all environments, experiments, and dimensions.
    
    Creates one large table with columns:
    - Environment
    - Experiment
    - All dimensions (union across all environments)
    - Avg Invalid Action Ratio
    
    Missing dimensions for specific environments are shown as empty cells.
    """
    if short_name_map is None:
        short_name_map = {}
    
    print("## ALL ENVIRONMENTS - COMBINED SUMMARY\n")
    
    # Collect all unique dimensions across all environments
    all_dimensions: Set[str] = set()
    for env_data in all_results.values():
        for exp_data in env_data.values():
            all_dimensions.update(exp_data.keys())
    
    dimensions_sorted = sorted(all_dimensions)
    
    # Build headers: Environment, Experiment, all dimensions, Avg Invalid Action Ratio
    headers = ["Environment", "Experiment"] + dimensions_sorted + ["Avg Invalid Action Ratio"]
    
    # Collect all rows
    rows: List[List[str]] = []
    
    for env_name in sorted(all_results.keys()):
        experiments_data = all_results[env_name]
        
        for exp_name in sorted(experiments_data.keys()):
            # Apply short_name mapping if available
            display_name = apply_short_name(exp_name, short_name_map)
            row = [env_name, display_name]
            
            invalid_ratios = []
            
            # Add data for each dimension
            for dimension in dimensions_sorted:
                if dimension in experiments_data[exp_name]:
                    summary_path = experiments_data[exp_name][dimension]
                    data = load_summary_json(summary_path)
                    task_success = data.get("task_success")
                    row.append(format_value(task_success))
                    
                    # Collect invalid action ratios for averaging
                    invalid_ratio = data.get("num_invalid_action_ratio")
                    if invalid_ratio is not None and not math.isnan(invalid_ratio):
                        invalid_ratios.append(invalid_ratio)
                else:
                    # Empty cell for missing dimension
                    row.append("")
            
            # Calculate and add average invalid action ratio
            if invalid_ratios:
                avg_invalid_ratio = sum(invalid_ratios) / len(invalid_ratios)
                row.append(format_value(avg_invalid_ratio))
            else:
                row.append("")
            
            rows.append(row)
    
    print(tabulate(rows, headers=headers, tablefmt="pipe"))
    print()


def parse_aggregated_table_from_text(text: str, env_name: str) -> Dict[str, Dict[str, Optional[float]]]:
    """Parse an aggregated summary table from text output.
    
    Returns a dict mapping experiment_name -> {dimension: task_success_mean, ...}
    Extracts the mean from "XX.XX% ± YY.YY%" format.
    For dimensions with "X items missing", stores None but still includes the dimension key.
    """
    results: Dict[str, Dict[str, Optional[float]]] = {}
    
    # Find the section for this environment
    env_marker = f"## Environment: {env_name.upper()} - AGGREGATED SUMMARY"
    if env_marker not in text:
        return results
    
    # Extract the table section
    start_idx = text.find(env_marker)
    next_section = text.find("\n## ", start_idx + 1)
    if next_section == -1:
        section = text[start_idx:]
    else:
        section = text[start_idx:next_section]
    
    # Find the table (starts after the header line with |:---|)
    lines = section.split('\n')
    in_table = False
    headers: List[str] = []
    
    for line in lines:
        if not line.strip() or not line.startswith('|'):
            continue
            
        if '|:--' in line or '|---' in line:
            in_table = True
            continue
        
        if not in_table:
            # This is the header row
            headers = [col.strip() for col in line.split('|')[1:-1]]
            continue
        
        # Parse data row
        cols = [col.strip() for col in line.split('|')[1:-1]]
        if len(cols) < 2:
            continue
        
        exp_name = cols[0]
        exp_data: Dict[str, Optional[float]] = {}
        
        # Parse each dimension column (skip "Experiment" and "Avg Invalid Action Ratio")
        for i, header in enumerate(headers[1:], start=1):
            if header == "Avg Invalid Action Ratio":
                break
            
            if i >= len(cols):
                continue
                
            value_str = cols[i]
            
            # Try to extract mean from "XX.XX% ± YY.YY%" format
            if '%' in value_str:
                mean_match = re.search(r'([\d.]+)%', value_str)
                if mean_match:
                    exp_data[header] = float(mean_match.group(1)) / 100.0
                else:
                    exp_data[header] = None
            elif value_str in ('-', 'N/A', 'no data', ''):
                exp_data[header] = None
            else:
                try:
                    exp_data[header] = float(value_str)
                except ValueError:
                    exp_data[header] = None
        
        if exp_data:
            results[exp_name] = exp_data
    
    return results


def main():
    """Main function to process and display all results."""
    import sys
    from io import StringIO
    
    # Get the running directory
    script_dir = Path(__file__).parent
    running_dir = script_dir.parent / "running"
    results_file = script_dir.parent / "results_pp.txt"

    if not running_dir.exists():
        print(f"Error: Running directory not found at {running_dir}")
        return

    # Read old results file BEFORE generating new output
    old_file_exists = results_file.exists()
    old_file_content: Optional[str] = None
    old_file_mtime: Optional[datetime] = None
    if old_file_exists:
        old_file_mtime = datetime.fromtimestamp(results_file.stat().st_mtime)
        with open(results_file, 'r') as f:
            old_file_content = f.read()
    
    # Capture all output to a string buffer
    output_buffer = StringIO()
    original_stdout = sys.stdout
    sys.stdout = output_buffer
    
    try:
        print(f"Scanning results from: {running_dir}")

        # Load short_name mappings from experiment_specs.yaml
        short_name_map = load_experiment_short_names()
        if short_name_map:
            print(f"Loaded {len(short_name_map)} short_name mappings from experiment_specs.yaml")

        # Parse the results structure
        results = parse_results_structure(running_dir)

        if not results:
            print("No results found!")
            return

        # Print summary of what was found
        print(f"\nFound results for {len(results)} environments:")
        for env_name in sorted(results.keys()):
            num_experiments = len(results[env_name])
            print(f"  - {env_name}: {num_experiments} experiment(s)")

        # Compare with previous results if available
        if old_file_exists and old_file_content is not None and old_file_mtime is not None:
            print("\n# CHANGES FROM PREVIOUS RUN\n")
            print(f"Comparing with previous results from: {old_file_mtime.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Track overall changes
            total_new_experiments = 0
            total_new_dimensions = 0
            total_improved = 0
            total_regressed = 0
            
            # Process each environment
            for env_name in sorted(results.keys()):
                # Parse old results for this environment
                old_env_results = parse_aggregated_table_from_text(old_file_content, env_name)
                
                if not old_env_results:
                    print(f"## Environment: {env_name.upper()}\n")
                    print(f"This environment is new! Found {len(results[env_name])} experiments.\n")
                    total_new_experiments += len(results[env_name])
                    continue
                
                # Build new results structure by aggregating repetitions
                clean_pattern = re.compile(r'^(?:Qwen2\.5-VL-7B-Instruct_)|_rep\d+$')
                
                def clean_experiment(exp: str) -> str:
                    exp_with_short = apply_short_name(exp, short_name_map)
                    return clean_pattern.sub('', exp_with_short)
                
                # Collect data grouped by cleaned experiment name
                new_grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
                
                for exp_name, dims in results[env_name].items():
                    cleaned = clean_experiment(exp_name)
                    for dimension, path in dims.items():
                        data = load_summary_json(path)
                        ts = data.get('task_success')
                        if isinstance(ts, (int, float)) and not (isinstance(ts, float) and math.isnan(ts)):
                            new_grouped[cleaned][dimension].append(ts)
                
                # Compute means for new results
                new_env_results: Dict[str, Dict[str, float]] = {}
                for exp_name, dims in new_grouped.items():
                    new_env_results[exp_name] = {}
                    for dim, values in dims.items():
                        if values:
                            new_env_results[exp_name][dim] = statistics.mean(values)
                
                # Compare experiments
                env_changes: List[str] = []
                env_new_experiments: List[str] = []
                env_new_dimensions: List[Tuple[str, str]] = []
                env_improvements: List[Tuple[str, str, float, float]] = []
                env_regressions: List[Tuple[str, str, float, float]] = []
                
                # Check for new experiments
                for exp_name in new_env_results:
                    if exp_name not in old_env_results:
                        env_new_experiments.append(exp_name)
                        total_new_experiments += 1
                
                # Check for changes in existing experiments
                for exp_name in sorted(set(old_env_results.keys()) & set(new_env_results.keys())):
                    old_dims = old_env_results[exp_name]
                    new_dims = new_env_results[exp_name]
                    
                    # Check for new dimensions
                    for dim in new_dims:
                        # Only report as "new" if the dimension didn't exist in old results at all
                        # Don't report if it existed but was None (incomplete)
                        if dim not in old_dims:
                            env_new_dimensions.append((exp_name, dim))
                            total_new_dimensions += 1
                        elif new_dims[dim] is not None and old_dims[dim] is not None:
                            # Compare values (both guaranteed non-None here)
                            old_val = old_dims[dim]
                            new_val = new_dims[dim]
                            assert old_val is not None and new_val is not None  # Type narrowing
                            diff = new_val - old_val
                            
                            # Consider significant changes (> 1% absolute difference)
                            if abs(diff) > 0.01:
                                if diff > 0:
                                    env_improvements.append((exp_name, dim, old_val, new_val))
                                    total_improved += 1
                                else:
                                    env_regressions.append((exp_name, dim, old_val, new_val))
                                    total_regressed += 1
                
                # Print environment changes if any
                if env_new_experiments or env_new_dimensions or env_improvements or env_regressions:
                    print(f"## Environment: {env_name.upper()}\n")
                    
                    if env_new_experiments:
                        print(f"### New Experiments ({len(env_new_experiments)})\n")
                        for exp in sorted(env_new_experiments):
                            dims = sorted(new_env_results[exp].keys())
                            print(f"- **{exp}** with dimensions: {', '.join(dims)}")
                        print()
                    
                    if env_new_dimensions:
                        print(f"### New Dimensions for Existing Experiments ({len(env_new_dimensions)})\n")
                        dim_by_exp: Dict[str, List[str]] = defaultdict(list)
                        for exp, dim in env_new_dimensions:
                            dim_by_exp[exp].append(dim)
                        for exp in sorted(dim_by_exp.keys()):
                            dims = sorted(dim_by_exp[exp])
                            print(f"- **{exp}**: {', '.join(dims)}")
                        print()
                    
                    if env_improvements:
                        print(f"### Improvements (>{1}% absolute increase) ({len(env_improvements)})\n")
                        for exp, dim, old_val, new_val in sorted(env_improvements, key=lambda x: x[3]-x[2], reverse=True):
                            diff = new_val - old_val
                            print(f"- **{exp}** [{dim}]: {old_val*100:.2f}% → {new_val*100:.2f}% (+{diff*100:.2f}%)")
                        print()
                    
                    if env_regressions:
                        print(f"### Regressions (>{1}% absolute decrease) ({len(env_regressions)})\n")
                        for exp, dim, old_val, new_val in sorted(env_regressions, key=lambda x: x[2]-x[3], reverse=True):
                            diff = new_val - old_val
                            print(f"- **{exp}** [{dim}]: {old_val*100:.2f}% → {new_val*100:.2f}% ({diff*100:.2f}%)")
                        print()
            
            # Print summary
            print("## Summary of Changes\n")
            print(f"- **New experiments**: {total_new_experiments}")
            print(f"- **New dimensions**: {total_new_dimensions}")
            print(f"- **Improvements**: {total_improved}")
            print(f"- **Regressions**: {total_regressed}")
            print()
        else:
            print("\n# CHANGES FROM PREVIOUS RUN\n")
            print("No previous results file found. This is the first run.\n")

        # # Print detailed tables for each environment
        # print("\n" + "=" * 150)
        # print("DETAILED TABLES BY DIMENSION")
        # print("=" * 150)

        # for env_name in sorted(results.keys()):
        #     print_environment_table(env_name, results[env_name])

        # # Print compact tables
        # print("\n" + "=" * 150)
        # print("COMPACT TABLES - KEY METRICS")
        # print("=" * 150)

        # for env_name in sorted(results.keys()):
        #     print_compact_table(env_name, results[env_name])

        # Print aggregated mean ± std tables
        print("\n# AGGREGATED SUMMARY TABLES - MEAN ± STD OF TASK SUCCESS\n")
        for env_name in sorted(results.keys()):
            print_aggregated_summary_tables(env_name, results[env_name], short_name_map)

        # Print combined summary table (all environments in one table)
        print("\n# COMBINED SUMMARY TABLE - ALL ENVIRONMENTS\n")
        print_combined_summary_table(results, short_name_map)
        
    finally:
        # Restore stdout
        sys.stdout = original_stdout
    
    # Get the generated output
    output_text = output_buffer.getvalue()
    
    # Write to file
    with open(results_file, 'w') as f:
        f.write(output_text)
    
    # Also print to stdout so user can see it
    print(output_text)
    
    # # Print individual summary tables (task_success with average invalid action ratio)
    # print("\n# INDIVIDUAL SUMMARY TABLES - TASK SUCCESS WITH AVG INVALID ACTION RATIO\n")

    # # Collect all experiment names across all environments
    # all_experiments = set()
    # for env_data in results.values():
    #     all_experiments.update(env_data.keys())

    # for env_name in sorted(results.keys()):
    #     print_summary_table(env_name, results[env_name], all_experiments)



if __name__ == "__main__":
    main()
