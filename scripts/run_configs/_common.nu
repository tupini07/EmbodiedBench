#!/usr/bin/env nu

export def start_ssh_tunnels  [
    amlt_job_names: list<string>,
    remote_urls: string
] {
    # amlt ssh "fair-bobcat" -o "StrictHostKeyChecking=no" -o "-4 -L 43003:localhost:43289"
    let local_ports = ($remote_urls | split row ',' | each { |url| 
        let parts = $url | split row ':'
        let port_part = ($parts | last)
        let port = ($port_part | split row '/' | first)
        $port
    })

    # there should be as many local_ports as amlt_job_names
    if ($local_ports | length) != ($amlt_job_names | length) {
        print $"(ansi red_bold)Error: Number of local ports (\($local_ports | length)) does not match number of AMLT job names (\($amlt_job_names | length)). Cannot start SSH tunnels.(ansi reset)"
        exit 1
    }

    # first, check all ports
    mut is_any_port_used = false
    for local_port in $local_ports {
        # --- Port availability check ---------------------------------------
        # We treat any existing LISTEN socket on the target port as an error to avoid
        # silently reusing an already-bound forward (or conflicting local service).
        # lsof output: header line + one or more rows if port is in use.
        let lsof_output = (bash -c $"lsof -iTCP:($local_port) -sTCP:LISTEN -Pn" | lines)
        let port_in_use = ($lsof_output | length) > 1
        if $port_in_use {
            print $"(ansi red_bold)Error: local port ($local_port) already in use.\nRefusing to start SSH tunnel to prevent conflicts.\nYou can inspect current owner via: lsof -iTCP:($local_port) -sTCP:LISTEN -Pn (ansi reset)"
            $is_any_port_used = true
        }
        # -------------------------------------------------------------------
    }

    if $is_any_port_used {
        print $"(ansi red_bold)One or more required local ports are already in use. Aborting SSH tunnel setup.(ansi reset)"
        exit 1
    } else {
        print $"(ansi green_bold)All required local ports are available. Proceeding to start SSH tunnels...(ansi reset)"
    }

    for bundle in ($local_ports | zip $amlt_job_names) {
        let local_port = $bundle | first
        let job_name = $bundle | last

        let cmd = $"amlt ssh \"($job_name)\" -o \"StrictHostKeyChecking=no\" -o \"-4 -L ($local_port):localhost:43289\""
        print $"Starting SSH tunnel for job '($job_name)' on local port ($local_port)..."

        # Echo the command first so users can easily copy-paste if needed
        let wrapper = $"echo 'Command: ($cmd)'; echo ''; ($cmd); exec bash"
        
        gnome-terminal --tab --title $"SSH Tunnel - ($job_name):($local_port)" -- bash -c $wrapper | ignore
        
        sleep 5sec
    }


    print $"(ansi green_bold)SSH tunnels started. Waiting for REMOTE_URLs to become reachable...(ansi reset)"
    sleep 10sec

    # verify that all URLs are reachable by pinging /models
    for url in ($remote_urls | split row ',') {
        mut reachable = false
        loop {
            let response = (curl -si $"($url)/models" | head -n 1)
            if ($response | str contains "200 OK") {
                $reachable = true
                break
            } else {
                print $"Waiting for REMOTE_URL ($url) to become reachable..."
                sleep 30sec
            }
        }
        if $reachable {
            print $"REMOTE_URL ($url) is reachable."
        }
    }
}

export def run_batch [
    temps:list<number>,
    max_tokens_list:list<number>,
    extra_env:record= {    # record of additional environment variables
        DEBUG_REMOTE_MODEL_INPUTS: "0"          # print model inputs to the log
        DEBUG_REMOTE_MODEL_OUTPUTS: "0"         # print model outputs to the log
        ONLY_ONE_STEP_PLAN: "0"                 # model is asked to produce only one-step plans
        EMB_PLANNER_RETRY_TIMES: "1"            # how many times to retry planning if failed or empty plan
        EMB_REASONING_MODE: "0"                 # whether to add the reasoning prompt postfix
        EB_SUPRESS_DURATION_LOGS: "0"           # suppress duration logs in embodiedbench
        REMOTE_URL: "http://localhost:43289/v1" # default remote model where vllm is reachable
        RUN_ALFRED: "1"                         # whether to run EB-ALFRED evaluations
        RUN_HABITAT: "1"                        # whether to run EB-Habitat
        RUN_MANIPULATION: "0"                   # whether to run EB-Manipulation evaluations
        RUN_NAVIGATION: "0"                     # whether to run EB-Navigation evaluations
    },  
    --amlt_job_names:list<string> = [],
    --replicates:int = 3,
    --no_pause = false,        # boolean switch default
    --stop_seqs:string = "</answer>",
    --prefix:string = "",      # optional prefix for run_name; also used for filtering jobs in polling/cleanup
    --skip_if_done = true       # by default, don't spawn job if dones file has ALL DONE
    --auto_pause_amlt_jobs = false   # whether to automatically pause amlt jobs upon completion
] {
    # Check if running in Docker mode
    let is_docker = ($env.EMBODIEDBENCH_DOCKER_MODE? | default "0") == "1"
    
    if $is_docker {
        print "[DOCKER] Running in Docker mode - auto-enabling no_pause and auto_pause_amlt_jobs=false"
    }
    
    # Force no_pause in Docker mode
    let effective_no_pause = if $is_docker { true } else { $no_pause }
    # Never pause AMLT jobs in Docker mode (since we don't have AMLT access)
    let effective_auto_pause = if $is_docker { false } else { $auto_pause_amlt_jobs }
    
    try {
        try { mkdir logs | ignore }
        try { mkdir running/dones | ignore }

        let main_job_id = job id
        
        let spawned_jobs_ids: list<int> = $temps | each { |temp|
            $max_tokens_list | each { |max_tokens|
                print $"Configured to run with temperature=($temp) and max_tokens=($max_tokens)"

                (seq 1 $replicates) | each { |rep|
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
                        # Only skip if prior marker indicates all sets succeeded (ALL DONE OK)
                        # let contains_done = (if $dones_exists { (try { open $dones_path | str contains "ALL DONE OK" } catch { false }) } else { false })
                        
                        # todo(atupini) forcing re-check for now. Adding new benchmaks breaks if all done is here even if that
                        # benchmark is not really done yet
                        let contains_done = false

                        let has_done = $dones_exists and $contains_done
                        if $has_done {
                            let msg = ("[SKIP] Completed run detected for run_name=" + $run_name)
                            print $msg
                            return null
                        }
                    }

                    let jobid = job spawn {
                        with-env $merged_env {
                            run_basic_evals $run_name $main_job_id
                        }
                    }

                    print $"Started job \(rep ($rep)/($replicates)\) for temperature ($temp) and max tokens ($max_tokens) with Job ID: ($jobid)"

                    if $effective_no_pause {
                        sleep 20sec
                    } else { 
                        try {
                            sleep 10sec

                            print ""
                            input $"(ansi purple_italic)Press Enter to launch the next replicate \(or Ctrl+C to stop)...(ansi reset)" 
                            print ""
                        } catch { |err|
                            print "Interrupted by user during pause. Cleaning up spawned jobs..."
                            print $err

                            print "Currently running jobs:"
                            job list
                            job list | each { |j| 
                                print $"Killing job ID: ($j.id)"
                                job kill $j.id
                            }

                            kill-all-regex $prefix
                            exit 1
                        }
                    }

                    $jobid
                } | flatten
            } | flatten
        } | flatten | where $it != null

        try {
            print "All jobs submitted. Polling for completion..."
            sleep 10sec  # brief pause before polling

            print "Spawned jobs after initial wait"
            job list

            # Structured monitoring of completion messages from run_basic_evals
            print "Starting structured job monitoring (waiting for completion messages)"
            mut completed_runs: list<string> = []
            mut statuses: list<record> = []
            let expected_jobs = ($spawned_jobs_ids | length)

            loop {
                if (($completed_runs | length) >= $expected_jobs) { break }

                let msg = (try { job recv } catch { null })
                if $msg != null {
                    match $msg {
                        {kind: 'exp', exp_name: $ename, status: $st} => {
                            let succ: int = ($msg.successes? | default 0)
                            let fail: int = ($msg.failures? | default 0)

                            if not ($completed_runs | any {|x| $x == $ename}) {
                                $completed_runs = ($completed_runs ++ [$ename])
                            }
                            $statuses = ($statuses ++ [$msg])
                            print $"[MONITOR] exp_name=($ename) status=($st) successes=($succ) failures=($fail) progress=($completed_runs | length)/($expected_jobs)"
                        },
                        _ => { print $"[MONITOR][WARN] Unexpected mailbox message: ($msg)" }
                    }
                }
            }

            print "================ Batch Summary ================"
            for s in $statuses {
                let succ = ($s.successes? | default null)
                let fail = ($s.failures? | default null)
                print $"Run: ($s.exp_name) Status: ($s.status) Successes: ($succ) Failures: ($fail)"
            }
            let success_count = ($statuses | where status == 'success' | length)
            let failure_count = ($statuses | where status == 'failure' | length)
            let skipped_count = ($statuses | where status == 'skipped' | length)
            let aborted_count = ($statuses | where status == 'aborted' | length)
            print $"Totals -> success: ($success_count), failure: ($failure_count), skipped: ($skipped_count), aborted: ($aborted_count)"
            print "=========================================================="

            # if no failures then offer to stop amulet jobs (skip in Docker mode)
            if (not $is_docker) and ($failure_count == 0 and ($amlt_job_names | length) > 0) {
                if $effective_auto_pause {
                    print $"(ansi green_bold)All jobs completed successfully with zero failures. Pausing Amulet jobs automatically.(ansi reset)"
                    for job_name in $amlt_job_names {
                        print $"Pausing Amulet job: ($job_name)"
                        bash -c $'amlt pause ($job_name)'
                    }
                    print $"(ansi green_bold)Amulet jobs paused.(ansi reset)"

                } else {
                    let answer = input $"(ansi green_bold)All jobs completed successfully with zero failures. Do you want me to stop the Amulet jobs? \(y/[n])(ansi reset)"
                    if $answer == "y" or $answer == "Y" {
                        for job_name in $amlt_job_names {
                            print $"Pausing Amulet job: ($job_name)"
                            bash -c $'amlt pause ($job_name)'
                        }
                        print $"(ansi green_bold)Amulet jobs paused.(ansi reset)"
                    } else {
                        print $"(ansi yellow_bold)Amulet jobs left running as per user choice.(ansi reset)"
                    }

                }
            }


        } catch { |err|
            print "Error occurred during job submission"
            print $err
            
            $spawned_jobs_ids | each { |jid| 
                print $"Cleaning up job ID: ($jid)"
                job kill $jid
            }

            kill-all-regex $prefix
            return {status: "error"}
        }
        
        # Return success status
        return {status: "success"}
    } catch { |err|
        # Fallback error handler: attempt regex-based termination of stray processes using the provided prefix
        print "Error during run_batch execution. Attempting regex-based cleanup with prefix pattern."
        print $err

        kill-all-regex $prefix
        return {status: "error"}
    }
}

# Regex-based killer that matches the FULL command line (using ps -eo pid,command) against a user-provided regex.
# Safer than killall with partial names; ignores empty pattern and self process.
export def kill-all-regex [pattern: string] {
    if ($pattern | str length) == 0 {
        print "[kill-all-regex] Empty pattern provided; nothing to kill."
        return
    }
    print $"[kill-all-regex] Searching for processes matching: /($pattern)/"
    
    let script = $"timeout 5s ps -eo pid,command 2>/dev/null | grep -E '($pattern)' | grep 'python -m embodiedbench.main' | grep -v grep | awk '{print $1}' || true"
    let pids = (bash -c $script | lines | where $it != "" | where $it != null)
    
    if ($pids | length) == 0 {
        print "[kill-all-regex] No matching processes found."
        return
    }
    
    print $"[kill-all-regex] Found ($pids | length) processes to kill"
    for p in $pids {
        let current_pid = $nu.pid
        if $p == $current_pid { continue }
        print $"[kill-all-regex] Killing PID ($p)"
        try { bash -c $"kill ($p) 2>/dev/null || true" } catch { print $"[kill-all-regex][WARN] Failed to kill pid ($p)" }
    }
}

def run_basic_evals [
    exp_name: string, 
    main_job_id: int
] {
    # rather than call into bash, implement everything in nu
    let remote_url = $env.REMOTE_URL? | default 'http://localhost:43289/v1'
    with-env {
        REMOTE_URL: $remote_url
        remote_url: $remote_url
        OPENAI_API_KEY: 'empty'
        EXTRA_ARGS: ($env.EXTRA_ARGS? | default '')
        EXTRA_ARGS_EB_ALFRED: ($env.EXTRA_ARGS_EB_ALFRED? | default '')
        EXTRA_ARGS_EB_HAB: ($env.EXTRA_ARGS_EB_HAB? | default '')
        EXTRA_ARGS_EB_MAN: ($env.EXTRA_ARGS_EB_MAN? | default '')
        EXTRA_ARGS_EB_NAV: ($env.EXTRA_ARGS_EB_NAV? | default '')
        MODEL_NAME: ($env.MODEL_NAME? | default 'Qwen2.5-VL-7B-Instruct')
        SKIP_IF_DONE: ($env.SKIP_IF_DONE? | default '1')
        FORCE_RERUN: ($env.FORCE_RERUN? | default '0')
    } { 
        mkdir "running/dones/"
        let prior_done_file = $"running/dones/($exp_name)_dones.txt"

        let prior_done_exists = ($prior_done_file | path exists)
       
        # Prior run considered complete only if success-only marker present
        # let prior_has_all_done = (if $prior_done_exists { (try { open $prior_done_file | str contains "ALL DONE OK" } catch { false }) } else { false })

        # todo(atupini) forcing re-check for now. Adding new benchmarks breaks if all done is here even if that
        let prior_has_all_done = false

        if ($env.FORCE_RERUN == "1") {
            print $"[FORCE_RERUN] Forcing rerun for exp_name '($exp_name)' \(ignoring any existing results)."
        } else {
            if (
                ($env.SKIP_IF_DONE == "1") and 
                $prior_done_exists and
                $prior_has_all_done
            ) {
                print $"[SKIP] Completed run detected via ($prior_done_file) \(contains ALL DONE OK). Skipping execution."
                {kind: 'exp', exp_name: $exp_name, status: 'skipped'} | job send $main_job_id
                return
            }
        }


        # # ensure that remoteURL is reachable
        # if ! curl -si "$REMOTE_URL/models" | head -n 1 | grep "200 OK" > /dev/null; then
        #     echo "Error: REMOTE_URL $REMOTE_URL is not reachable."
        #     echo ""
        #     echo "Please ensure SSH tunnel is running in another terminal:"
        #     echo ""
        #     echo "    amlt ssh \"job_name\" -o \"StrictHostKeyChecking=no\" -o \"-4 -L 43289:localhost:43289\""
        #     echo ""
        #     echo ""
        #     exit 1
        # fi

        # print evaluation configuration
        print "================ Evaluation Configuration ================"
        print $"Experiment Name: ($exp_name)"
        print $"Model Name: ($env.MODEL_NAME)"
        print $"REMOTE_URL: ($env.REMOTE_URL)"
        print $"EXTRA_ARGS: ($env.EXTRA_ARGS)"
        print $"EXTRA_ARGS_EB_ALFRED: ($env.EXTRA_ARGS_EB_ALFRED)"
        print $"EXTRA_ARGS_EB_HAB: ($env.EXTRA_ARGS_EB_HAB)"
        print $"EXTRA_ARGS_EB_MAN: ($env.EXTRA_ARGS_EB_MAN)"
        print $"EXTRA_ARGS_EB_NAV: ($env.EXTRA_ARGS_EB_NAV)"
        print "=========================================================="

        print $"Running evaluation with exp_name: ($exp_name)"

        mkdir "./running"

        # ---------------------------------------------------------------
        # Remove previous results directory if rerun conditions met.
        # Mirrors bash logic:
        #   if FORCE_RERUN==1 OR SKIP_IF_DONE!=1 OR NOT(all done marker present) -> purge.
        
        if (($env.FORCE_RERUN == "1") or ($env.SKIP_IF_DONE != "1") or (not ($prior_done_exists and $prior_has_all_done))) {
            let target_name = $"($env.MODEL_NAME)_($exp_name)"
            let targets = (try { ls running/** | where type == dir and name == $target_name } catch { [] })
            
            for t in $targets { 
                print $"[CLEAN] Removing prior results dir: ($t.name)"
                rm -r $t.name 
            }
        }

        # ---------------------------------------------------------------
        # Headless toggle: if HEADLESS=1 enable software rendering + unset DISPLAY.
        # Otherwise, use the DISPLAY variable set by the parent (from shared Xvfb server).
        if (($env.HEADLESS? | default "") == "1") {
            print "[HEADLESS] Enabling software rendering (EGL surfaceless)."
            load-env {
                CUDA_VISIBLE_DEVICES: ""
                MAGNUM_DEFAULT_GL_CONTEXT_VERSION: "330"
                GALLIUM_DRIVER: "llvmpipe"
                EGL_PLATFORM: "surfaceless"
            }
            hide-env DISPLAY
        } else {
            print $"[DISPLAY] Using shared Xvfb server \(DISPLAY=($env.DISPLAY? | default 'not set'))"
        }

        # ---------------------------------------------------------------
        # Initialize (truncate) dones file for this experiment.
        mkdir running/dones | ignore
        "" | save -f $prior_done_file

        # Per-experiment log directory
        let log_dir = $"logs/($exp_name)"

        mkdir $log_dir | ignore

        let parent_job_id = job id

        let model_basename = ($env.MODEL_NAME | split row '/' | last)

        # EB-ALFRED (parallel over all eval_sets) ---------------------------
        # Always run all supported evaluation sets; user no longer configures subset.
        mut alfred_job_ids: list<int> = []
        mut spawned_alfred_sets: list<string> = []
        
        if ($env.RUN_ALFRED? | default "1") != "1" {
            print "[EB-ALFRED] Skipping EB-ALFRED evaluations as per RUN_ALFRED!=1."
        } else {
            let alfred_all_sets = ["base" "common_sense" "complex_instruction" "spatial" "visual_appearance" "long_horizon"]
            print "[EB-ALFRED] Running all evaluation sets." 

            let alfred_log_dir = ($log_dir | path join "EB-ALFRED")
            mkdir $alfred_log_dir | ignore

            for eval_set in $alfred_all_sets {
                # Skip if summary.json already exists and SKIP_IF_DONE enabled
                let summary_path = $"running/eb_alfred/($model_basename)_($exp_name)/($eval_set)/results/summary.json"
                let already_done = ($summary_path | path exists)
                if $already_done and ($env.SKIP_IF_DONE == '1') {
                    print $"[EB-ALFRED][SKIP] summary.json detected for eval_set='($eval_set)' -> skipping"
                    {env: 'alfred', set: $eval_set, status: 'done'} | job send $parent_job_id
                    $spawned_alfred_sets = ($spawned_alfred_sets ++ [$eval_set])
                    continue
                }

                let log_file = ($alfred_log_dir | path join $"($eval_set).log")
                print $"[EB-ALFRED] Spawning eval_set='($eval_set)' ..."
                let jobid = job spawn {
                    let cmd = $"conda run --no-capture-output -n embench python -m embodiedbench.main env=eb-alf model_name='($env.MODEL_NAME)' exp_name='($exp_name)' eval_sets='[($eval_set)]' ($env.EXTRA_ARGS) ($env.EXTRA_ARGS_EB_ALFRED) > '($log_file)' 2>&1"
                    let results = (bash -c $cmd | complete)
                    if $results.exit_code != 0 {
                        echo $"[EB-ALFRED] Eval set ($eval_set) failed exit_code=($results.exit_code)"
                        {env: 'alfred', set: $eval_set, status: 'error'} | job send $parent_job_id
                    } else {
                        echo $"[EB-ALFRED] Eval set ($eval_set) done"
                        {env: 'alfred', set: $eval_set, status: 'done'} | job send $parent_job_id
                    }
                }
                $alfred_job_ids = ($alfred_job_ids ++ [$jobid])
                $spawned_alfred_sets = ($spawned_alfred_sets ++ [$eval_set])

                sleep 5sec
            }

        }

        let alfred_jobs_count = ($spawned_alfred_sets | length)
        if $alfred_jobs_count == 0 {
            print "[EB-ALFRED] No eval_set jobs spawned (all skipped or none specified)."
        }


        # EB-Habitat (parallel per eval_set) --------------------------------
        # EB-Habitat always run all sets
        mut habitat_job_ids: list<int> = []
        mut spawned_habitat_sets: list<string> = []

        if ($env.RUN_HABITAT? | default "1") != "1" {
            print "[EB-Habitat] Skipping EB-Habitat evaluations as per RUN_HABITAT!=1."
        } else {
            let habitat_all_sets = ["base" "common_sense" "complex_instruction" "spatial_relationship" "visual_appearance" "long_horizon"]
            print "[EB-Habitat] Running all evaluation sets." 

            let habitat_log_dir = ($log_dir | path join "EB-Habitat")
            mkdir $habitat_log_dir | ignore

            for eval_set in $habitat_all_sets {
                let summary_path = $"running/eb_habitat/($model_basename)_($exp_name)/($eval_set)/results/summary.json"
                let already_done = ($summary_path | path exists)
                if $already_done and ($env.SKIP_IF_DONE == '1') {
                    print $"[EB-Habitat][SKIP] summary.json detected for eval_set='($eval_set)' -> skipping"
                    {env: 'habitat', set: $eval_set, status: 'done'} | job send $parent_job_id
                    $spawned_habitat_sets = ($spawned_habitat_sets ++ [$eval_set])
                    continue
                }
                let log_file = ($habitat_log_dir | path join $"($eval_set).log")
                print $"[EB-Habitat] Spawning eval_set='($eval_set)' ..."
                let jobid = job spawn {
                    let cmd = $"conda run --no-capture-output -n embench python -m embodiedbench.main env=eb-hab model_name='($env.MODEL_NAME)' exp_name='($exp_name)' eval_sets='[($eval_set)]' ($env.EXTRA_ARGS) ($env.EXTRA_ARGS_EB_HAB) > '($log_file)' 2>&1"
                    
                    # habitat only works on gpu, and will fail with a segmentation fault if the GPU is full
                    # so we do a `do while` loop here to retry on segfaults
                    mut results = {}
                    while true {
                        $results = (bash -c $cmd | complete)

                        let has_seg_fault: bool = (try { open $log_file | str contains "Segmentation fault" } catch { false })
                        if (not $has_seg_fault) {
                            break
                        } 

                        # wait 30 min with some jitter of 15 min
                        let sleep_duration = (30 + (random int 0..15))

                        print $"[retry_habitat:($eval_set)] segfault detected; sleeping ($sleep_duration) minutes then retrying..."

                        sleep ($"($sleep_duration)min" | into duration)
                    }

                    if $results.exit_code != 0 {
                        echo $"[EB-Habitat] Eval set ($eval_set) failed exit_code=($results.exit_code)"
                        {env: 'habitat', set: $eval_set, status: 'error'} | job send $parent_job_id
                    } else {
                        echo $"[EB-Habitat] Eval set ($eval_set) done"
                        {env: 'habitat', set: $eval_set, status: 'done'} | job send $parent_job_id
                    }
                }
                $habitat_job_ids = ($habitat_job_ids ++ [$jobid])
                $spawned_habitat_sets = ($spawned_habitat_sets ++ [$eval_set])

                sleep 10sec
            }

        }
        
        let habitat_jobs_count = ($spawned_habitat_sets | length)
        if $habitat_jobs_count == 0 { 
            print "[EB-Habitat] No eval_set jobs spawned (all skipped or none specified)." 
        }

        # EB-Manipulation (parallel per eval_set) ---------------------------
        # Spawns one job per manipulation eval set if RUN_MANIPULATION==1.
        # Includes basic CoppeliaSim/PyRep environment setup inside each job.
        mut manipulation_job_ids: list<int> = []
        mut spawned_manipulation_sets: list<string> = []

        if ($env.RUN_MANIPULATION? | default "0") != "1" {
            print "[EB-Manipulation] Skipping EB-Manipulation evaluations as per RUN_MANIPULATION!=1."
        } else {
            let manipulation_all_sets = ["base" "common_sense" "complex" "spatial" "visual"]
            print "[EB-Manipulation] Running all evaluation sets." 

            let manip_log_dir = ($log_dir | path join "EB-Manipulation")
            mkdir $manip_log_dir | ignore

            for eval_set in $manipulation_all_sets {
                # Skip if summary.json already exists and SKIP_IF_DONE enabled
                let summary_path = $"running/eb_manipulation/($model_basename)/($exp_name)/($eval_set)/results/summary.json"
                let already_done = ($summary_path | path exists)
                if $already_done and ($env.SKIP_IF_DONE == '1') {
                    print $"[EB-Manipulation][SKIP] summary.json detected for eval_set='($eval_set)' -> skipping"
                    {env: 'manip', set: $eval_set, status: 'done'} | job send $parent_job_id
                    $spawned_manipulation_sets = ($spawned_manipulation_sets ++ [$eval_set])
                    continue
                }

                let log_file = ($manip_log_dir | path join $"($eval_set).log")
                print $"[EB-Manipulation] Spawning eval_set='($eval_set)' ..."
                let jobid = job spawn {
                    let parent_job_id = job id
                    # Attempt lightweight environment setup (CoppeliaSim / PyRep)
                    let root_coppelia = ($env.PWD | path join "CoppeliaSim_Pro_V4_1_0_Ubuntu20_04")
                    let alt_coppelia = ($env.PWD | path join "embodiedbench" "envs" "eb_manipulation" "CoppeliaSim_Pro_V4_1_0_Ubuntu20_04")
                    let chosen = if ($root_coppelia | path exists) { $root_coppelia } else if ($alt_coppelia | path exists) { $alt_coppelia } else { null }
                    if $chosen == null {
                        print "[EB-Manipulation] CoppeliaSim directory not found; aborting eval set.";
                        {env: 'manip', set: $eval_set, status: 'error'} | job send $parent_job_id
                    } else {
                        let sys_dir = ($chosen | path join "system")
                        let usrset = ($sys_dir | path join "usrset.txt")
                        if not ($usrset | path exists) { mkdir $sys_dir | ignore; touch $usrset }
                        let ld_old = ($env.LD_LIBRARY_PATH? | default "")
                        let ld_new = (if ($ld_old | is-empty) { $chosen } else { $"($chosen):($ld_old)" })
                        load-env {
                            COPPELIASIM_ROOT: $chosen
                            LD_LIBRARY_PATH: $ld_new
                            QT_QPA_PLATFORM_PLUGIN_PATH: $chosen
                        }
                        # Quick PyRep import test (non-fatal)
                        let pyrep_test = (bash -c "conda run -n embench_man python -c 'import pyrep'" | complete)
                        if $pyrep_test.exit_code != 0 {
                            print "[EB-Manipulation] PyRep import failed pre-run test; continuing but evaluation may fail." 
                        }
                        let cmd = $"conda run --no-capture-output -n embench_man python -m embodiedbench.main env=eb-man model_name='($env.MODEL_NAME)' exp_name='($exp_name)' eval_sets='[($eval_set)]' ($env.EXTRA_ARGS) ($env.EXTRA_ARGS_EB_MAN) > '($log_file)' 2>&1"
                        
                        # EB-Manipulation can fail with CUDA out of memory errors when GPU is full
                        # so we do a `do while` loop here to retry on CUDA OOM errors
                        mut results = {}
                        while true {
                            $results = (bash -c $cmd | complete)

                            let has_cuda_oom: bool = (try { open $log_file | str contains "CUDA error: out of memory" } catch { false })
                            if (not $has_cuda_oom) {
                                break
                            } 

                            # wait 30 min with some jitter of 15 min
                            let sleep_duration = (30 + (random int 0..15))

                            print $"[retry_manipulation:($eval_set)] CUDA OOM detected; sleeping ($sleep_duration) minutes then retrying..."

                            sleep ($"($sleep_duration)min" | into duration)
                        }

                        if $results.exit_code != 0 {
                            echo $"[EB-Manipulation] Eval set ($eval_set) failed exit_code=($results.exit_code)"
                            {env: 'manip', set: $eval_set, status: 'error'} | job send $parent_job_id
                        } else {
                            echo $"[EB-Manipulation] Eval set ($eval_set) done"
                            {env: 'manip', set: $eval_set, status: 'done'} | job send $parent_job_id
                        }
                    }
                }
                $manipulation_job_ids = ($manipulation_job_ids ++ [$jobid])
                $spawned_manipulation_sets = ($spawned_manipulation_sets ++ [$eval_set])
                sleep 10sec
            }
        }

        let manipulation_jobs_count = ($spawned_manipulation_sets | length)
        if $manipulation_jobs_count == 0 {
            print "[EB-Manipulation] No eval_set jobs spawned (all skipped or none specified)."
        }

        # EB-Navigation (parallel per eval_set) ------------------------------
        mut navigation_job_ids: list<int> = []
        mut spawned_navigation_sets: list<string> = []

        if ($env.RUN_NAVIGATION? | default "0") != "1" {
            print "[EB-Navigation] Skipping EB-Navigation evaluations as per RUN_NAVIGATION!=1."
        } else {
            let navigation_all_sets = ["base" "common_sense" "complex_instruction" "visual_appearance" "long_horizon"]
            print "[EB-Navigation] Running all evaluation sets." 

            let nav_log_dir = ($log_dir | path join "EB-Navigation")
            mkdir $nav_log_dir | ignore

            for eval_set in $navigation_all_sets {
                let summary_path = $"running/eb_nav/($model_basename)_($exp_name)/($eval_set)/results/summary.json"
                let already_done = ($summary_path | path exists)
                if $already_done and ($env.SKIP_IF_DONE == '1') {
                    print $"[EB-Navigation][SKIP] summary.json detected for eval_set='($eval_set)' -> skipping"
                    {env: 'nav', set: $eval_set, status: 'done'} | job send $parent_job_id
                    $spawned_navigation_sets = ($spawned_navigation_sets ++ [$eval_set])
                    continue
                }
                let log_file = ($nav_log_dir | path join $"($eval_set).log")
                print $"[EB-Navigation] Spawning eval_set='($eval_set)' ..."
                let jobid = job spawn {
                    let parent_job_id = job id
                    # AI2-THOR Navigation needs GLX and proper rendering extensions (already set in shared Xvfb)
                    let cmd = $"conda run --no-capture-output -n embench_nav python -m embodiedbench.main env=eb-nav model_name='($env.MODEL_NAME)' exp_name='($exp_name)' eval_sets='[($eval_set)]' ($env.EXTRA_ARGS) ($env.EXTRA_ARGS_EB_NAV) > '($log_file)' 2>&1"
                    let results = (bash -c $cmd | complete)
                    if $results.exit_code != 0 {
                        echo $"[EB-Navigation] Eval set ($eval_set) failed exit_code=($results.exit_code)"
                        {env: 'nav', set: $eval_set, status: 'error'} | job send $parent_job_id
                    } else {
                        echo $"[EB-Navigation] Eval set ($eval_set) done"
                        {env: 'nav', set: $eval_set, status: 'done'} | job send $parent_job_id
                    }
                }
                $navigation_job_ids = ($navigation_job_ids ++ [$jobid])
                $spawned_navigation_sets = ($spawned_navigation_sets ++ [$eval_set])
                sleep 5sec
            }
        }

        let navigation_jobs_count = ($spawned_navigation_sets | length)
        if $navigation_jobs_count == 0 {
            print "[EB-Navigation] No eval_set jobs spawned (all skipped or none specified)."
        }

        # Immutable snapshot of job ids for later cleanup
        let alfred_job_ids_snapshot = $alfred_job_ids
        let habitat_job_ids_snapshot = $habitat_job_ids
        let manipulation_job_ids_snapshot = $manipulation_job_ids
        let navigation_job_ids_snapshot = $navigation_job_ids

        # use jobs mailbox to wait for all done/error signals
        try {
            # total expected: parallel Alfred + Habitat + Manipulation + Navigation jobs
            let total_envs = ($alfred_jobs_count + $habitat_jobs_count + $manipulation_jobs_count + $navigation_jobs_count)
            print $"Waiting for ($total_envs) evaluation sets to complete..."

            mut processed = 0
            mut successes = 0
            mut failures = 0

            loop {
                if $processed >= $total_envs { break }

                let msg = job recv
                match $msg {
                    {env: $e, status: $st, set: $s} => {
                        let human_env = match $e {
                            'alfred' => 'EB-ALFRED'
                            'habitat' => 'EB-Habitat'
                            'manip' => 'EB-Manipulation'
                            'nav' => 'EB-Navigation'
                            _ => $e
                        }
                        if $st == 'done' {
                            $processed = $processed + 1
                            $successes = $successes + 1
                            print $"[MAILBOX] ($human_env) set=($s) (ansi green_bold)success(ansi reset). Progress: ($processed)/($total_envs)"
                            $"($human_env):($s)" | save --append $prior_done_file
                        } else if $st == 'error' {
                            $processed = $processed + 1
                            $failures = $failures + 1
                            print $"[MAILBOX] ($human_env) set=($s) (ansi red_bold)failure(ansi reset). Progress: ($processed)/($total_envs)"
                        } else {
                            print $"[MAILBOX][WARN] Unknown status ($st) for env ($human_env) set=($s)"
                        }
                    },
                    _ => { print $"[WARN] Received unknown message in mailbox: ($msg)" }
                }
            }

            if ($failures == 0) and ($successes == $total_envs) {
                $"ALL DONE OK successes=($successes) failures=($failures)" | save --append $prior_done_file
                print $"All evaluations completed SUCCESSFULLY. Successes=($successes) Failures=($failures) \(markers written to: ($prior_done_file))"
                # send success completion to supervising batch job
                {kind: 'exp', exp_name: $exp_name, status: 'success', successes: $successes, failures: $failures} | job send $main_job_id
            } else {
                $"ALL DONE WITH FAILURES successes=($successes) failures=($failures)" | save --append $prior_done_file
                print $"Evaluations finished with FAILURES. Successes=($successes) Failures=($failures). Run will NOT be skipped next time. \(markers written to: ($prior_done_file))"
                # send failure completion to supervising batch job
                {kind: 'exp', exp_name: $exp_name, status: 'failure', successes: $successes, failures: $failures} | job send $main_job_id
            }
        } catch { |err|
            print "Interrupted while waiting for evaluations to complete (likely Ctrl-C). Cleaning up related jobs..."
            print $err

            print "Currently running jobs:"
            job list

            for jid in $alfred_job_ids_snapshot { try { job kill $jid } }
            for jid in $habitat_job_ids_snapshot { try { job kill $jid } }
            for jid in $manipulation_job_ids_snapshot { try { job kill $jid } }
            for jid in $navigation_job_ids_snapshot { try { job kill $jid } }
            {kind: 'exp', exp_name: $exp_name, status: 'aborted'} | job send $main_job_id
        }
    }
}