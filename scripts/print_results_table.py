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
    results = defaultdict(lambda: defaultdict(dict))

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
action_num_per_plan
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


if __name__ == "__main__":
    main()
