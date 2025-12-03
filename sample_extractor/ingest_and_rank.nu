#!/usr/bin/env nu

# Ingest, rank, and prepare episode comparisons for specified run IDs
#
# Usage:
#   ./sample_extractor/ingest_and_rank.nu run1 run2 run3 --tasks "EB-ALFRED EB-Habitat"
#   ./sample_extractor/ingest_and_rank.nu "Qwen*" --tasks "EB-ALFRED" --model gpt-4o
#
# Parameters:
#   ...run_names: One or more run IDs or patterns (space-separated)
#   --tasks: Space-separated task names (default: "EB-ALFRED")
#   --model: Model for ranking (default: "gpt-4o")
#   --skip-outcomes: Skip episode outcome ingestion
#   --db: Database path (default: "outputs/steps_cache.sqlite")

def main [
    ...run_names: string,  # One or more run IDs
    --tasks: string = "EB-ALFRED",
    --model: string = "gpt-4o",
    --skip-outcomes,
    --db: string = "outputs/steps_cache.sqlite"
] {
    if ($run_names | is-empty) {
        print "Error: Please provide at least one run name"
        print "Usage: ./sample_extractor/ingest_and_rank.nu run1 run2 run3 --tasks 'EB-ALFRED EB-Habitat'"
        exit 1
    }

    let runs_str = ($run_names | str join " ")
    let runs_csv = ($run_names | str join ",")
    
    print $"==== Ingesting logs for runs: ($runs_str) ===="
    python -m sample_extractor.ingest --runs $runs_csv --tasks $tasks --db $db

    print $"\n==== Scoring missing steps ===="
    python -m sample_extractor.rank_missing --runs $runs_csv --batch-size 25 --trapi-model $model --db $db

    if not $skip_outcomes {
        print $"\n==== Ingesting episode outcomes ===="
        let task_dirs = ($tasks | split row " " | each { |t| 
            match $t {
                "EB-ALFRED" => "eb_alfred",
                "EB-Habitat" => "eb_habitat", 
                "EB-Navigation" => "eb_navigation",
                _ => ($t | str downcase)
            }
        })
        
        python -m sample_extractor.ingest_outcomes --base-dir running --tasks ...($task_dirs) --db $db
        
        print "\n==== Pipeline complete! ===="
        print "Next steps:"
        print "  1. Generate comparisons:"
        print "     python -m sample_extractor.generate_comparisons --target-runs 'YOUR_MODEL*' --baseline-runs 'Qwen*'"
        print "  2. View comparisons:"
        print "     streamlit run episode_comparison_viewer.py"
    } else {
        print "\n==== Pipeline complete (outcomes skipped) ===="
    }
}
