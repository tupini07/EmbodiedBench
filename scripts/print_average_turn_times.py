from pathlib import Path
import typer
import json
import numpy as np
from math import sqrt


def compute_ci(mean: float, std: float, n: int, confidence: float = 0.95):
    """Return (lower, upper) confidence interval for the mean using a normal approximation.

    For n >= 30 this is typically fine. If n < 30 and high accuracy is desired,
    consider swapping to a t-distribution critical value or bootstrapping.
    """
    if n == 0:
        return (float('nan'), float('nan'))
    # Two-tailed z for common confidence levels; fallback to 1.96
    z_lookup = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_lookup.get(confidence, 1.96)
    margin = z * std / sqrt(n)
    return mean - margin, mean + margin


def main(
    experiments_path: Path = typer.Argument(
        "./running/eb_alfred/Qwen2.5-VL-7B-Instruct_cleaned_videos_unprocessed_split1_min_1.0_num_4.9K_videos_ActionCorrection_OneStepPlanLoose_3Retries_WithReasoning_temp0.6_maxTokens6192_rep1"
    ),
    episodes_per_task: int = typer.Option(50, help="Expected number of episodes per task."),
    confidence: float = typer.Option(0.95, help="Confidence level for intervals (e.g. 0.95)."),
):
    experiments_path = Path(experiments_path).resolve().absolute()

    if not experiments_path.exists():
        print("Input experiment path doesn't exist!")
        return

    averages_per_task = {}

    for task_folder in experiments_path.iterdir():
        # base, common_sense, etc
        task_results_dir = task_folder / "results"
        if not task_results_dir.exists():
            print(f"Task results dir doesn't exist, skipping: {task_results_dir}")
            continue

        task_step_averages = []
        episode_num_steps = []

        for ep_result in task_results_dir.glob("episode_*.json"):
            data = json.loads(ep_result.read_text())

            num_steps = data["num_steps"]
            episode_elapsed_seconds = data["episode_elapsed_seconds"]

            if float(num_steps) == 0:
                continue

            task_step_averages.append(float(episode_elapsed_seconds) / float(num_steps))
            episode_num_steps.append(float(num_steps))

        # Pairwise episode times (duration per episode) for more direct CI
        episode_times = [d * n for d, n in zip(task_step_averages, episode_num_steps)]

        averages_per_task[task_folder.name] = {
            "step_durations": task_step_averages,
            "num_steps": episode_num_steps,
            "episode_times": episode_times,
        }

    for task_name, data_dict in averages_per_task.items():
        step_durations = data_dict["step_durations"]
        num_steps_list = data_dict["num_steps"]
        episode_times = data_dict["episode_times"]

        n = len(step_durations)
        missing = episodes_per_task - n

        # Means & STDs
        step_mean = float(np.average(step_durations)) if n else float('nan')
        step_std = float(np.std(step_durations, ddof=1)) if n > 1 else float('nan')

        num_steps_mean = float(np.average(num_steps_list)) if n else float('nan')
        num_steps_std = float(np.std(num_steps_list, ddof=1)) if n > 1 else float('nan')

        episode_time_mean = float(np.average(episode_times)) if n else float('nan')
        episode_time_std = float(np.std(episode_times, ddof=1)) if n > 1 else float('nan')

        # CIs (normal approx)
        step_ci_low, step_ci_high = compute_ci(step_mean, step_std, n, confidence)
        num_steps_ci_low, num_steps_ci_high = compute_ci(num_steps_mean, num_steps_std, n, confidence)
        ep_time_ci_low, ep_time_ci_high = compute_ci(episode_time_mean, episode_time_std, n, confidence)

        # Aggregate expected total task time across episodes_per_task
        total_task_time_mean = episode_time_mean * episodes_per_task
        total_task_time_ci_low = ep_time_ci_low * episodes_per_task
        total_task_time_ci_high = ep_time_ci_high * episodes_per_task

        print(task_name)
        print(f"\tEpisodes observed: {n} / {episodes_per_task} (missing {missing if missing>0 else 0})")
        print(f"\tConfidence level: {confidence:.2f}")

        # Episode time (seconds)
        print("\tEpisode time (seconds):")
        print(f"\t\tMean: {episode_time_mean:.3f}")
        print(f"\t\tStd: {episode_time_std:.3f}")
        print(f"\t\tCI: [{ep_time_ci_low:.3f}, {ep_time_ci_high:.3f}]")

        # Total task time (seconds & hours)
        hours_mean = total_task_time_mean / 3600.0
        hours_ci_low = total_task_time_ci_low / 3600.0
        hours_ci_high = total_task_time_ci_high / 3600.0
        print("\tTotal expected task time (all episodes):")
        print(f"\t\tMean: {total_task_time_mean:.2f} s ({hours_mean:.3f} h)")
        print(f"\t\tCI: [{total_task_time_ci_low:.2f}, {total_task_time_ci_high:.2f}] s ([{hours_ci_low:.3f}, {hours_ci_high:.3f}] h)")

        # Step duration stats
        print("\tStep duration (seconds per step):")
        print(f"\t\tMean: {step_mean:.4f}")
        print(f"\t\tStd: {step_std:.4f}")
        print(f"\t\tCI: [{step_ci_low:.4f}, {step_ci_high:.4f}]")

        # Num steps stats
        print("\tNum steps per episode:")
        print(f"\t\tMean: {num_steps_mean:.3f}")
        print(f"\t\tStd: {num_steps_std:.3f}")
        print(f"\t\tCI: [{num_steps_ci_low:.3f}, {num_steps_ci_high:.3f}]")


if __name__ == "__main__":
    typer.run(main)
