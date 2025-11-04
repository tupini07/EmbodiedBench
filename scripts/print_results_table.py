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
from typing import Any, Dict, List, Optional, Set

from tabulate import tabulate
import re
import statistics


def load_summary_json(file_path: Path) -> Dict[str, Any]:
    """Load and parse a summary.json file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def parse_results_structure(running_dir: Path) -> Dict[str, Dict[str, Dict[str, Path]]]:
    """
    Parse the running directory structure.

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

        # Iterate through each experiment directory
        for exp_dir in env_dir.iterdir():
            if not exp_dir.is_dir() or exp_dir.name.startswith("."):
                continue

            exp_name = exp_dir.name

            # Iterate through each dimension directory (base, common_sense, etc.)
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
    print(f"\n{'='*100}")
    print(f"Environment: {env_name.upper()}")
    print(f"{'='*100}\n")

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
    print(f"\n{'='*150}")
    print(f"Environment: {env_name.upper()} - COMPACT VIEW")
    print(f"{'='*150}\n")

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
    print(f"\n{'='*150}")
    print(f"Environment: {env_name.upper()} - SUMMARY (Task Success)")
    print(f"{'='*150}\n")

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


def print_aggregated_summary_tables(env_name: str, experiments_data: Dict[str, Dict[str, Path]]):
    """Print aggregated summary table: groups repetitions and shows mean ± std for task_success per dimension and invalid action ratio.

        Strategy:
            1. Derive base experiment names (strip _repX)
            2. For each base experiment & dimension collect task_success values
            3. Also collect num_invalid_action_ratio per dimension then average per repetition, then aggregate those averages across repetitions.
                 (Alternatively average over all dimension ratios per repetition then aggregate; we mirror existing summary which averages across dimensions.)
    We will mimic existing summary's average invalid action ratio: for each repetition (exp name with rep suffix) we compute the average of available dimension ratios; then we aggregate across reps for same base.
    """
    # Collect dimensions across this environment
    all_dimensions_set: Set[str] = set()
    for dims in experiments_data.values():
        all_dimensions_set.update(dims.keys())
    all_dimensions_list = sorted(all_dimensions_set)
    if not all_dimensions_list:
        return

    # Map: base_name -> dimension -> list of task_success values
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    # Map: base_name -> list of avg invalid action ratio per repetition
    grouped_invalid: Dict[str, List[float]] = defaultdict(list)

    # Temporary: base_name -> rep_id -> list of invalid ratios to compute per-rep average
    per_rep_invalid: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for exp_name, dims in experiments_data.items():
        base_name = base_experiment_name(exp_name)
        rep_match = re.search(r"(_rep\d+)$", exp_name)
        rep_id = rep_match.group(1) if rep_match else "_rep0"  # default single rep
        invalid_values_this_rep: List[float] = []
        for dimension, path in dims.items():
            data = load_summary_json(path)
            ts = data.get("task_success")
            if isinstance(ts, (int, float)) and not (isinstance(ts, float) and math.isnan(ts)):
                grouped[base_name][dimension].append(float(ts))
            invalid_ratio = data.get("num_invalid_action_ratio")
            if isinstance(invalid_ratio, (int, float)) and not (isinstance(invalid_ratio, float) and math.isnan(invalid_ratio)):
                per_rep_invalid[base_name][rep_id].append(float(invalid_ratio))

    # Compute per-rep average invalid ratios then aggregate
    for base_name, rep_dict in per_rep_invalid.items():
        for rep_id, ratios in rep_dict.items():
            if ratios:
                grouped_invalid[base_name].append(sum(ratios)/len(ratios))

    # Prepare table rows
    headers = ["Experiment"] + all_dimensions_list + ["Avg Invalid Action Ratio"]
    rows: List[List[str]] = []

    for base_name in sorted(set(list(grouped.keys()) + list(grouped_invalid.keys()))):
        row: List[str] = [base_name]
        for dim in all_dimensions_list:
            values = grouped.get(base_name, {}).get(dim, [])
            cell = format_mean_std(compute_mean_std(values)) if values else "-"
            row.append(cell)
        # Invalid action ratio aggregation (not percentage) -> treat as raw [0,1], show *100? Original table shows raw ~0.5 so leave raw
        invalid_stats = compute_mean_std(grouped_invalid.get(base_name, []))
        row.append(format_mean_std(invalid_stats, percent=False) if invalid_stats else "-")
        rows.append(row)

    print(f"\n{'='*150}")
    print(f"Environment: {env_name.upper()} - AGGREGATED SUMMARY (Mean ± Std of Task Success)" )
    print(f"{'='*150}\n")
    print(tabulate(rows, headers=headers, tablefmt='pipe'))
    print()


def main():
    """Main function to process and display all results."""
    # Get the running directory
    script_dir = Path(__file__).parent
    running_dir = script_dir.parent / "running"

    if not running_dir.exists():
        print(f"Error: Running directory not found at {running_dir}")
        return

    print(f"Scanning results from: {running_dir}")

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

    # Print detailed tables for each environment
    print("\n" + "=" * 150)
    print("DETAILED TABLES BY DIMENSION")
    print("=" * 150)

    for env_name in sorted(results.keys()):
        print_environment_table(env_name, results[env_name])

    # Print compact tables
    print("\n" + "=" * 150)
    print("COMPACT TABLES - KEY METRICS")
    print("=" * 150)

    for env_name in sorted(results.keys()):
        print_compact_table(env_name, results[env_name])

    # Print summary tables (task_success with average invalid action ratio)
    print("\n" + "=" * 150)
    print("SUMMARY TABLES - TASK SUCCESS WITH AVG INVALID ACTION RATIO")
    print("=" * 150)

    # Collect all experiment names across all environments
    all_experiments = set()
    for env_data in results.values():
        all_experiments.update(env_data.keys())

    for env_name in sorted(results.keys()):
        print_summary_table(env_name, results[env_name], all_experiments)

    # Print aggregated mean ± std tables
    print("\n" + "=" * 150)
    print("AGGREGATED SUMMARY TABLES - MEAN ± STD OF TASK SUCCESS")
    print("=" * 150)
    for env_name in sorted(results.keys()):
        print_aggregated_summary_tables(env_name, results[env_name])


if __name__ == "__main__":
    main()
