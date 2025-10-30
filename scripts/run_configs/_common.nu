#!/usr/bin/env nu

# Shared helper functions for evaluation run config scripts.
# Provides reusable logic for launching multiple replicate jobs with varying
# temperature and max_tokens parameters, handling cleanup on error, and polling
# for completion.
#
# Usage (inside another .nu script):
#   use _common.nu *
#   run_batch $temps $max_tokens_list $replicates {
#       # env block body executed per job spawn with variables available:
#       #   $temp $max_tokens $rep run_name
#       # Return a record of extra environment variables (optional)
#       { DEBUG_REMOTE_MODEL_INPUTS: "1", DEBUG_REMOTE_MODEL_OUTPUTS: "1" }
#   }
# You can also pass --no-pause true to disable interactive pauses.

export def run_batch [
    temps:list<number>,
    max_tokens_list:list<number>,
    extra_env:record= {    # record of additional environment variables
        DEBUG_REMOTE_MODEL_INPUTS: "0"          # print model inputs to the log
        DEBUG_REMOTE_MODEL_OUTPUTS: "0"         # print model outputs to the log
        ONLY_ONE_STEP_PLAN: "0"                 # model is asked to produce only one-step plans
        EMB_PLANNER_RETRY_TIMES: "1"            # how many times to retry planning if failed or empty plan
        REMOTE_URL: "http://localhost:43289/v1" # default remote model where vllm is reachable
    },  
    --replicates:int = 3,
    --no_pause = false,        # boolean switch default
    --reasoning_mode:string = "1",
    --stop_seqs:string = "</answer>",
    --prefix:string = ""       # optional prefix for run_name
] {
    mkdir logs | ignore
    try {
        for temp in $temps {
            for max_tokens in $max_tokens_list {
                for rep in (seq 1 $replicates) {
                    let base_name = $"temp($temp)_maxTokens($max_tokens)_rep($rep)"
                    let run_name = if $prefix == "" { $base_name } else { $"($prefix)($base_name)" }
                    let merged_env = {
                        REMOTE_MODEL_TEMPERATURE: $temp
                        REMOTE_MODEL_MAX_TOKENS: $max_tokens
                        EMB_REASONING_MODE: $reasoning_mode
                        REMOTE_MODEL_STOP_SEQS: $stop_seqs
                    } | merge $extra_env
                    let jobid = job spawn {
                        with-env $merged_env {
                            bash -c $"bash scripts/run_evals_basic.sh '($run_name)' > logs/($run_name).log 2>&1 & echo $!" | str trim
                            $"DONE ($run_name)" | job send 0
                        }
                    }
                    print $"Started job \(rep ($rep)/($replicates)\) for temperature ($temp) and max tokens ($max_tokens) with Job ID: ($jobid)"
                    if not $no_pause { input "Press Enter to launch the next replicate (or Ctrl+C to stop)..." }
                }
            }
        }
    } catch {  |err|
        print "Error occurred during job submission"
        print $"Error details: ($err)"
        clean_running_jobs
        return { status: "error" }
    }

    poll_until_finished
    { status: "ok" }
}

# Poll for completion of any running embodiedbench python jobs
export def poll_until_finished [] {
    loop {
        let active = ps --long | where command =~ "python -m embodiedbench" | select pid
        if ($active | length) == 0 { break }
        print $"Still running: ($active | get pid | str join ', ')"
        sleep 15sec
    }
    print "All jobs finished."
}

# Kill any currently running embodiedbench python jobs
export def clean_running_jobs [] {
    print "Running jobs to clean up"
    let ids = ps --long | where command =~ "python -m embodiedbench" | select pid
    print $"Found ($ids | length) embodiedbench jobs running."
    for jid in $ids { print $"Stopping job: ($jid.pid)"; kill -s 9 $jid.pid }
    print "All running jobs have been stopped."
}
