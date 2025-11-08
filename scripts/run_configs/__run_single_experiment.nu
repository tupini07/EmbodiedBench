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
    
    # before doing anythin, kill any current jobs and wait. By this point we know we should not have any running jobs with this prefix
    kill-all-regex $prefix
    sleep 5sec

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
    let exp_extra_env = ($exp.extra_env? | default {})
    mut extra_env = ($base_env | merge $exp_extra_env)

    let amlt_job_names = ($exp.amlt_job_names? | default [])
    mut remote_urls = ($extra_env.REMOTE_URL? | default "")

    # Auto-generate REMOTE_URL if not provided but amlt_job_names exist
    if ($remote_urls | str length) == 0 and (($amlt_job_names | length) > 0) {
        print $"[AUTO-PORT] No REMOTE_URL provided, generating deterministic ports based on prefix hash..."
        
        # Generate a deterministic base port from prefix hash (range: 60000-90000)
        let prefix_sha256: string = ($prefix | hash sha256)
        let hash_sum: int = (python -c $"print\(sum\(bytearray\(b'($prefix_sha256)')))") | into int
        let initial_base_port = (60000 + (($hash_sum mod 30000) * 1000 / 1000)) | into int
        
        # Retry logic: try up to 10 different base ports (with offset of 100 each)
        mut ports_found = false
        mut local_ports = []
        mut final_base_port = $initial_base_port
        
        for attempt in 0..9 {
            let current_base_port = $initial_base_port + ($attempt * 100)
            
            # Generate ports for each amlt job (offset by 1 for each replica)
            let num_replicas = ($amlt_job_names | length)
            let candidate_ports = (1..$num_replicas | each {|i| $current_base_port + $i })
            
            # Check if any of the generated ports are already in use
            mut is_any_port_used = false
            for local_port in $candidate_ports {
                # Port availability check using lsof
                # We treat any existing LISTEN socket on the target port as an error to avoid
                # silently reusing an already-bound forward (or conflicting local service).
                let lsof_output = (bash -c $"lsof -iTCP:($local_port) -sTCP:LISTEN -Pn 2>/dev/null" | lines)
                let port_in_use = ($lsof_output | length) > 1
                if $port_in_use {
                    print $"[AUTO-PORT] Port ($local_port) already in use \(attempt ($attempt + 1)/10)"
                    $is_any_port_used = true
                    break
                }
            }
            
            if not $is_any_port_used {
                $ports_found = true
                $local_ports = $candidate_ports
                $final_base_port = $current_base_port
                print $"[AUTO-PORT] Found available ports starting at ($current_base_port) \(attempt ($attempt + 1)/10)"
                break
            }
        }
        
        if not $ports_found {
            print $"(ansi red_bold)Error: Could not find available ports after 10 attempts.\nLast attempted base port: ($final_base_port)\nYou can inspect port usage via: lsof -iTCP -sTCP:LISTEN -Pn(ansi reset)"
            remove-lock $prefix
            exit 1
        }

        $remote_urls = ($local_ports | each {|p| $"http://localhost:($p)/v1" } | str join ",")
        
        print $"[AUTO-PORT] Generated REMOTE_URL: ($remote_urls)"
        
        # Update extra_env with the generated REMOTE_URL
        $extra_env = ($extra_env | upsert REMOTE_URL $remote_urls)
    }

    let global_replicates = ($base.replicates? | default 3)
    let global_skip_if_done = ($base.skip_if_done? | default true)
    let global_no_pause = ($base.no_pause? | default false)

    if (($amlt_job_names | length) > 0) and ($remote_urls | str length) > 0 {
        job spawn {
            # reword all amulet jobs descriptions so they match the prefix. We don't really care about the result of this. It's mainly for bookkeeping.
            with-env {
                AZ_SUBSCRIPTION_ID: "2cd190bb-b42a-477c-b1bb-2f20932d8dc5"
                AZ_RESOURCE_GROUP: "atupinirg"
                AZ_WORKSPACE_NAME: "atupinirgws"
            } {
                let target_name = $"EmbodiedBench-vllm-($prefix)"
                for job_name in $amlt_job_names {
                    conda run -n base python scripts/amlt_set_display_name.py $"($target_name)" ...$amlt_job_names
                }
            }
        }

        print $"[SSH] Starting tunnels for prefix=($prefix) urls=($remote_urls)"
        start_ssh_tunnels $amlt_job_names $remote_urls
    }

    print $"[RUN] prefix=($prefix) temps=($temps) max_tokens=($max_tokens_list)"

    # Invoke run_batch (positionals first, then named flags). Avoid line continuations for Nu portability.
    let result = (run_batch $temps $max_tokens_list $extra_env 
                            --amlt_job_names $amlt_job_names
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
