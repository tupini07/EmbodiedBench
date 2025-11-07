#!/usr/bin/env nu

# run_single_experiment.nu
# ---------------------------------------------------------------
# Launch a single experiment (identified by prefix) from experiment_specs.yaml
# in its own gnome-terminal tab. Handles:
#   * Duplicate / missing experiment detection
#   * Lock file creation & removal on completion or error
#   * SSH tunnel startup for its REMOTE_URL set
#   * Calls run_batch with merged settings
# Usage:
#   nu scripts/run_configs/run_single_experiment.nu <prefix>
#   (Optionally wrap in gnome-terminal externally, or let run_all_yaml.nu spawn it.)
# ---------------------------------------------------------------

use _common.nu *

def lock-file [prefix:string] { $"running/locks/($prefix).lock" }
def create-lock [prefix:string] { mkdir running/locks | ignore; $"timestamp=(date now); pid=($nu.pid)" | save -f (lock-file $prefix) }
def remove-lock [prefix:string] { let lf = (lock-file $prefix); if ($lf | path exists) { rm $lf } }

def main [prefix:string] {
    # Determine directory of this script (fallback to current dir if unavailable)
    const self_path = (path self)
    let dir = (dirname $self_path)
    let yaml_path = ($dir | path join "experiment_specs.yaml")
    let spec = (open $yaml_path)
    let base = ($spec.base_settings)
    let experiments = ($spec.experiments? | default [])
    if ($experiments | length) == 0 { print (ansi red_bold) + "No experiments defined." + (ansi reset); exit 1 }

    let matches = ($experiments | where {|e| ($e.prefix? | default '') == $prefix })
    if ($matches | length) == 0 { print (ansi red_bold) + $"Prefix not found: ($prefix)" + (ansi reset); exit 1 }
    if ($matches | length) > 1 { print (ansi red_bold) + $"Multiple experiments with same prefix: ($prefix)" + (ansi reset); exit 1 }
    let exp = ($matches | first)

    let lf = (lock-file $prefix)

    create-lock $prefix
    print $"[LOCK] Created lock for prefix: ($prefix)"

    let stop_seq: string = ($base.stop_seqs? | default ["</answer>"] | first)
    let temps = ($exp.temperature? | default [0.6])
    let max_tokens_list = ($exp.max_tokens? | default [6144])

    let base_env = ($base.extra_env? | default {})

    # extra env. Defaults from base, overridden by experiment-specific.
    let extra_env = ($base_env | merge ($exp.extra_env? | default {}))

    let remote_urls = ($extra_env.REMOTE_URL? | default "")
    let amlt_job_names = ($exp.amlt_job_names? | default [])
    let global_replicates = ($base.replicates? | default 3)
    let global_skip_if_done = ($base.skip_if_done? | default true)
    let global_no_pause = ($base.no_pause? | default false)

    if (($amlt_job_names | length) > 0) and ($remote_urls | str length) > 0 {
        print $"[SSH] Starting tunnels for prefix=($prefix) urls=($remote_urls)"
        start_ssh_tunnels $amlt_job_names $remote_urls
    }

    print $"[RUN] prefix=($prefix) temps=($temps) max_tokens=($max_tokens_list)"

    # Invoke run_batch (positionals first, then named flags). Avoid line continuations for Nu portability.
    let result = (run_batch $temps $max_tokens_list $extra_env 
                            --stop_seqs $stop_seq 
                            --prefix $prefix 
                            --replicates $global_replicates 
                            --skip_if_done $global_skip_if_done 
                            --no_pause $global_no_pause)

    let status = ($result.status? | default "unknown")
    print $"[DONE] prefix=($prefix) status=($status)"
    remove-lock $prefix
    print $"[LOCK] Removed lock for prefix: ($prefix)"
}
