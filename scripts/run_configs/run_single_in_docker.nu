#!/usr/bin/env nu

# ---------------------------------------------------------------
# run_single_in_docker.nu
# ---------------------------------------------------------------
# Docker-compatible single experiment launcher that reads from experiment_specs.yaml
# and runs a specific experiment by prefix without requiring gnome-terminal or SSH tunnels.
#
# This script is designed to run inside Docker containers where:
# - No X server or gnome-terminal is available
# - AMLT commands are not available
# - vLLM endpoints are directly accessible via specified ports
#
# Usage:
#   nu run_single_in_docker.nu <prefix> [--vllm-ports port1,port2,...]
#
# Example:
#   nu run_single_in_docker.nu my_experiment --vllm-ports 8000,8001
# ---------------------------------------------------------------

const this_script_path = (path self)
const yaml_path = ($this_script_path | path dirname | path join experiment_specs.yaml)

def main [
    prefix: string,                              # Experiment prefix to run
    --vllm-ports: string = "22001",              # Comma-separated list of vLLM server ports (e.g., "8000" or "8000,8001,8002")
    --vllm-host: string = "localhost"           # Hostname where vLLM servers are running
] {
    print $"[DOCKER] Running in Docker mode - prefix: ($prefix)"
    print $"[DOCKER] vLLM endpoints will be constructed from: ($vllm_host):($vllm_ports)"
    
    # Set environment variable to indicate Docker mode
    $env.EMBODIEDBENCH_DOCKER_MODE = "1"
    
    # Load experiment specs
    print $"[INFO] Loading experiment specs: ($yaml_path)"
    let spec = (open $yaml_path)
    let base = ($spec.base_settings)
    let experiments = ($spec.experiments? | default [])
    
    if ($experiments | length) == 0 {
        print (ansi red_bold) + "No experiments defined in YAML." + (ansi reset)
        exit 1
    }
    
    # Find the experiment with matching prefix
    let matches = ($experiments | where {|e| ($e.prefix? | default '') == $prefix })
    if ($matches | length) == 0 {
        print (ansi red_bold) + $"Prefix not found: ($prefix)" + (ansi reset)
        print "Available prefixes:"
        for exp in $experiments {
            let exp_prefix = ($exp.prefix? | default "")
            if ($exp_prefix | str length) > 0 {
                print $"  - ($exp_prefix)"
            }
        }
        exit 1
    }
    if ($matches | length) > 1 {
        print (ansi red_bold) + $"Multiple experiments with same prefix: ($prefix)" + (ansi reset)
        exit 1
    }
    
    let exp = ($matches | first)
    
    print $"[CONFIG] Found experiment configuration for prefix: ($prefix)"
    
    # Extract configuration
    let stop_seq: string = ($base.stop_seqs? | default ["</answer>"] | first)

    let temps = ($exp.temperature? | default [0.6])
    let max_tokens_list = ($exp.max_tokens? | default [6144])
    let global_replicates = ($base.replicates? | default 3)
    let global_skip_if_done = ($base.skip_if_done? | default true)
    let global_no_pause = ($base.no_pause? | default false)
    
    # Merge environment variables
    let base_env = ($base.extra_env? | default {})
    let exp_extra_env = ($exp.extra_env? | default {})
    mut extra_env = ($base_env | merge $exp_extra_env)
    
    # Construct REMOTE_URL from vllm-ports parameter
    let ports = ($vllm_ports | split row ',')
    let remote_urls = ($ports | each {|port| $"http://($vllm_host):($port)/v1" } | str join ",")
    
    print $"[DOCKER] Constructed REMOTE_URL: ($remote_urls)"
    $extra_env = ($extra_env | upsert REMOTE_URL $remote_urls)
    $extra_env = ($extra_env | upsert remote_url $remote_urls)

    
    # Add Docker mode flag to environment
    $extra_env = ($extra_env | upsert EMBODIEDBENCH_DOCKER_MODE "1")
    
    print ""
    print "================ Docker Experiment Configuration ================"
    print $"Prefix: ($prefix)"
    print $"Temperatures: ($temps)"
    print $"Max Tokens: ($max_tokens_list)"
    print $"Replicates: ($global_replicates)"
    print $"Skip if Done: ($global_skip_if_done)"
    print $"REMOTE_URL: ($remote_urls)"
    print $"Extra Environment:"
    for key in ($extra_env | columns) {
        let val = ($extra_env | get $key)
        print $"  ($key): ($val)"
    }
    print "================================================================="
    print ""
    
    # Now we need to inline the logic from __run_single_experiment.nu
    # but adapted for Docker (no AMLT, no SSH tunnels, no gnome-terminal)
    
    # Import run_batch from _common.nu
    use _common.nu run_batch
    
    print $"[RUN] Starting batch execution..."
    
	let xvfb_cmd = $"Xvfb :1 -screen 0 1024x768x24 +extension GLX +render -noreset -ac"
	print $"[XVFB] Starting: ($xvfb_cmd)"
	
	# Start Xvfb in background
	let xvfb_result = (bash -c $"($xvfb_cmd) > /tmp/xvfb_42.log 2>&1 & echo $!" | complete)

	$env.DISPLAY = ":1"

	# first, ensure that everything is set up properly (assets downloaded)
	^conda run --no-capture-output -n embench python -m embodiedbench.envs.eb_alfred.EBAlfEnv
	^conda run --no-capture-output -n embench python -m embodiedbench.envs.eb_habitat.EBHabEnv
	^conda run --no-capture-output -n embench_nav python -m embodiedbench.envs.eb_navigation.EBNavEnv
	^conda run --no-capture-output -n embench_man python -m embodiedbench.envs.eb_manipulation.EBManEnv

    let result = (run_batch $temps $max_tokens_list $extra_env 
								--amlt_job_names []
								--stop_seqs $stop_seq 
								--prefix $prefix 
								--replicates $global_replicates 
								--skip_if_done $global_skip_if_done 
								--no_pause true
								--auto_pause_amlt_jobs false)
    
    let status = ($result.status? | default "unknown")
    print ""
    print $"[DONE] Experiment completed with status: ($status)"
    
    if $status == "success" {
        exit 0
    } else {
        exit 1
    }
}