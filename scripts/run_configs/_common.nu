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
        EMB_REASONING_MODE: "0"                 # whether to add the reasoning prompt postfix
        REMOTE_URL: "http://localhost:43289/v1" # default remote model where vllm is reachable
    },  
    --replicates:int = 3,
    --no_pause = false,        # boolean switch default
    --stop_seqs:string = "</answer>",
    --prefix:string = "",      # optional prefix for run_name; also used for filtering jobs in polling/cleanup
    --skip_if_done = true       # by default, don't spawn job if dones file has ALL DONE
] {
    mkdir logs | ignore
    mkdir running/dones | ignore
    try {
        for temp in $temps {
            for max_tokens in $max_tokens_list {
                for rep in (seq 1 $replicates) {
                    let base_name = $"temp($temp)_maxTokens($max_tokens)_rep($rep)"
                    let run_name = if $prefix == "" { $base_name } else { $"($prefix)($base_name)" }
                    let merged_env = {
                        REMOTE_MODEL_TEMPERATURE: $temp
                        REMOTE_MODEL_MAX_TOKENS: $max_tokens
                        REMOTE_MODEL_STOP_SEQS: $stop_seqs
                        SKIP_IF_DONE: (if $skip_if_done { "1" } else { "0" })
                        FORCE_RERUN: "0"
                    } | merge $extra_env

                    if $skip_if_done {
                        let dones_path = $"running/dones/($run_name)_dones.txt"
                        let dones_exists = (ls running/dones | where name == $"($run_name)_dones.txt" | length) > 0
                        let contains_done = (if $dones_exists { (try { open $dones_path | str contains "ALL DONE" } catch { false }) } else { false })
                        let has_done = $dones_exists and $contains_done
                        if $has_done {
                            let msg = ("[SKIP] Completed run detected for run_name=" + $run_name)
                            print $msg
                            continue
                        }
                    }
                    let jobid = job spawn {
                        with-env $merged_env {
                            bash -c $"bash scripts/run_evals_basic.sh '($run_name)' > logs/($run_name).log 2>&1 & echo $!" | str trim
                            $"DONE ($run_name)" | job send 0
                        }
                    }
                    print $"Started job \(rep ($rep)/($replicates)\) for temperature ($temp) and max tokens ($max_tokens) with Job ID: ($jobid)"
                    if $no_pause {
                        sleep 10sec
                    } else { 
                        sleep 5sec
                        input "Press Enter to launch the next replicate (or Ctrl+C to stop)..." 
                    }
                }
            }
        }
    } catch {  |err|
        print "Error occurred during job submission"
        print $"Error details: ($err)"
        clean_running_jobs $prefix
        return { status: "error" }
    }

    print "All jobs submitted. Polling for completion..."
    sleep 10sec  # brief pause before polling

    try {
        input "Press Enter to automatically monitor job completion (or Ctrl+C to stop)..." 
    } catch { |err|
        print "Interrupted before polling (likely Ctrl-C). Cleaning up related jobs..."
        clean_running_jobs $prefix
        print $"Cleanup complete. \(Error: ($err)\)"
        return { status: "interrupted" }
    }

    poll_until_finished $prefix
    { status: "ok" }
}

###
# Helper: attempt to gather PID list matching a pattern.
# 1. Try Nushell's built-in `ps --long` (fast & structured) unless it errors.
# 2. On error or suspected file descriptor exhaustion, fallback to invoking bash `ps`.
# Returns a list of records with `pid` if successful, otherwise empty list.
def ps_pids_matching [pattern:string] {
    let nu_attempt = (try { ps --long | select pid command } catch { |err|
        # $err is a structured error record; convert to text before string operations.
        let err_text = ($err | to text)
        if ($err_text | str contains "file descriptors") {
            print "[WARN] Nushell ps likely hit FD exhaustion; falling back to bash ps." 
        } else {
            print $"[WARN] Nushell ps failed: ($err_text); falling back to bash ps."
        }
        null
    })
    if $nu_attempt != null {
        return ($nu_attempt | where command =~ $pattern | select pid)
    }

    # Fallback via bash. Using -eo pid,command to get all processes in two columns.
    let raw = (try { bash -c "ps -eo pid,command" } catch { |err|
        print $"[ERROR] bash ps failed: ($err)"
        ""
    })
    if $raw == "" { return [] }

    $raw | lines | skip 1 | each {|line|
        let trimmed = ($line | str trim)
        if ($trimmed | str length) == 0 { null } else {
            let cols = ($trimmed | split row ' ' | where {|x| $x != ""})
            if ($cols | length) > 1 { { pid: ($cols | first), command: ($cols | skip 1 | str join ' ') } } else { null }
        }
    } | where command =~ $pattern | select pid
}

# Determine if any jobs matching a filter/pattern are currently running.
# Improvements over previous version:
#   - BOUNDED retries (default 5 attempts) instead of infinite loop on transient ps failures.
#   - Exponential backoff to reduce hammering `ps` and leaking descriptors.
#   - Fallback to bash `ps` when Nushell `ps` errors or returns null.
#   - Returns false if unable to obtain process list after max retries (defensive instead of hanging).
def are_jobs_running_for_filter [filter:string, --max_retries:int=5] {
    let pattern = (if $filter != "" { $filter } else { "python -m embodiedbench" })
    mut attempt = 0
    loop {
        let pids = (ps_pids_matching $pattern)
        if $pids != null {
            if ($pids | length) == 0 { return false } else { return true }
        }
        if $attempt >= $max_retries {
            print "[WARN] Exceeded max retries obtaining process list; assuming no jobs running."
            return false
        }
        let backoff = ($attempt + 1) * 2
        print $"[INFO] Retry ($attempt + 1)/($max_retries) after transient process listing issue; sleeping ($backoff)s"
    # Dynamic sleep using duration multiplication to avoid string coercion issues
    sleep ($backoff * 1sec)
        $attempt = $attempt + 1
    }
}

# Poll for completion of any running embodiedbench python jobs, optionally filtered by prefix
export def poll_until_finished [filter:string=""] {
    let pattern = (if $filter != "" { $filter } else { "python -m embodiedbench" })
    try {
        loop {
            if !(are_jobs_running_for_filter $filter) {
                break
            }
            print --no-newline "."
            sleep 20sec
        }
        print ""
        print "All jobs finished."
    } catch { |err|
        print "Interrupted (likely Ctrl-C). Cleaning up related jobs..."
        clean_running_jobs $filter
        print $"Cleanup complete. \(Error: ($err)\)"
    }
}

# Kill any currently running embodiedbench python jobs, optionally filtered by prefix
export def clean_running_jobs [filter:string=""] {
    print "Running jobs to clean up"
    let pattern = (if $filter != "" { $filter } else { "python -m embodiedbench" })
    # Prefer structured helper first (avoids brittle grep parsing)
    let primary = (ps_pids_matching $pattern)
    mut pids = ($primary | select pid)
    if ($pids | length) == 0 {
        print "[INFO] No matching PIDs from structured ps; attempting bash text fallback..."
        # Safe fallback using ps -eo; avoids the classic 'grep -v grep' anti-pattern.
        let raw = (try { bash -c "PATTERN=\"$pattern\"; ps -eo pid,command | grep -F \"$PATTERN\"" } catch { |err|
            print $"[WARN] bash fallback ps failed: ($err)"
            ""
        })
        if $raw != "" {
            let lines = ($raw | lines | where {|l| ($l | str trim | str length) > 0 })
            let parsed = ($lines | each {|l|
                let parts = ($l | split row ' ' | where {|x| $x != ""})
                if ($parts | length) > 0 { { pid: ($parts | first) } } else { null }
            } | where {|r| $r != null })
            # Reassign mutable list of pids from parsed fallback
            $pids = ($parsed | select pid)
        }
    }

    print $"Found ($pids | length) embodiedbench jobs running."
    if ($pids | length) == 0 {
        print "Nothing to clean up. Still trying with bash" 
        bash -c $"ps a | grep -v 'grep' | grep '($filter)' | cut -f1 -d' ' | xargs -I {} kill -9 {}"
        return
    }

    for entry in $pids { 
        let pid = ($entry.pid | into int)
        print $"Stopping job: ($pid) \(SIGTERM then SIGKILL if needed\)"
        # Try graceful first
        try { kill $pid } catch { print $"[WARN] SIGTERM failed for ($pid); proceeding to SIGKILL." }
        sleep 300ms
        # Check if still alive
        let still = (ps_pids_matching $pattern | where pid == $pid | length) > 0
        if $still {
            try { kill -s 9 $pid } catch { print $"[WARN] SIGKILL failed or already exited: ($pid)" }
        }
    }
    print "Cleanup attempts complete."
}
