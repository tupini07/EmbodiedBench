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

        gnome-terminal --tab --title $"SSH Tunnel - ($job_name):($local_port)" -- bash -c $"($cmd); exec bash" | ignore
        
        sleep 5sec
    }


    print $"(ansi green_bold)SSH tunnels started. Waiting for REMOTE_URLs to become reachable...(ansi reset)"

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
    },  
    --replicates:int = 3,
    --no_pause = false,        # boolean switch default
    --stop_seqs:string = "</answer>",
    --prefix:string = "",      # optional prefix for run_name; also used for filtering jobs in polling/cleanup
    --skip_if_done = true       # by default, don't spawn job if dones file has ALL DONE
] {
    mkdir logs | ignore
    mkdir running/dones | ignore
    
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
                    let contains_done = (if $dones_exists { (try { open $dones_path | str contains "ALL DONE OK" } catch { false }) } else { false })

                    let has_done = $dones_exists and $contains_done
                    if $has_done {
                        let msg = ("[SKIP] Completed run detected for run_name=" + $run_name)
                        print $msg
                        return null
                    }
                }

                let jobid = job spawn {
                    with-env $merged_env {
                        run_basic_evals $run_name

                        $"DONE ($run_name)" | job send 0
                    }
                }

                print $"Started job \(rep ($rep)/($replicates)\) for temperature ($temp) and max tokens ($max_tokens) with Job ID: ($jobid)"

                if $no_pause {
                    sleep 20sec
                } else { 
                    try {
                        sleep 10sec

                        print ""
                        input $"(ansi purple_italic)Press Enter to launch the next replicate \(or Ctrl+C to stop)...(ansi reset)" 
                        print ""
                    } catch {
                        print "Interrupted by user during pause. Cleaning up spawned jobs..."
                        print "Currently running jobs:"
                        job list
                        job list | each { |j| 
                            print $"Killing job ID: ($j.id)"
                            job kill $j.id
                        }

                        kill-all $prefix
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

        input $"(ansi yellow_bold)[Manual-Gate] Press Enter to start to automatically monitor job completion \(or Ctrl+C to stop)...(ansi reset)" 

        # monitor ids
        print --no-newline "Starting job monitoring"
        loop {
            if (job list | length) == 0 {
                break
            }

            print --no-newline "."
            sleep 20sec
        }

    } catch { |err|
        print "Error occurred during job submission"
        print $"Error details: ($err)"
        
        $spawned_jobs_ids | each { |jid| 
            print $"Cleaning up job ID: ($jid)"
            job kill $jid
        }

        kill-all $prefix
        exit 1
    }
}

def kill-all [filter: string] {
    print $"(ansi red_bold)Killing all jobs matching filter: '($filter)' (ansi reset)"
    bash -c $'ps a | grep -v "grep" | grep "($filter)" | cut -f1 -d" " | xargs -I {} kill -9 {}'
}

def run_basic_evals [exp_name: string] {
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
        let prior_has_all_done = (if $prior_done_exists { (try { open $prior_done_file | str contains "ALL DONE OK" } catch { false }) } else { false })

        if ($env.FORCE_RERUN == "1") {
            print $"[FORCE_RERUN] Forcing rerun for exp_name '($exp_name)' \(ignoring any existing results)."
        } else {
            if (
                ($env.SKIP_IF_DONE == "1") and 
                $prior_done_exists and
                $prior_has_all_done
            ) {
                print $"[SKIP] Completed run detected via ($prior_done_file) \(contains ALL DONE OK). Skipping execution."
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
            load-env { DISPLAY: ":1" }
        }

        # ---------------------------------------------------------------
        # Initialize (truncate) dones file for this experiment.
        mkdir running/dones | ignore
        "" | save -f $prior_done_file

        # Per-experiment log directory
        let log_dir = $"logs/($exp_name)"

        # delete if exists
        try { rm -r $log_dir }

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
            return
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

        # Immutable snapshot of job ids for later cleanup
        let alfred_job_ids_snapshot = $alfred_job_ids
        let habitat_job_ids_snapshot = $habitat_job_ids

        # # EB-Manipulation ----------------------------------------------------
        # let manipulation_job_id = job spawn {
        #     let log_file = ($log_dir | path join "EB-Manipulation.log")
        #     print $"[EB-Man] Starting manipulation setup..."

        #     let root_coppelia = ($env.PWD | path join "CoppeliaSim_Pro_V4_1_0_Ubuntu20_04")
        #     let alt_coppelia = ($env.PWD | path join "embodiedbench" "envs" "eb_manipulation" "CoppeliaSim_Pro_V4_1_0_Ubuntu20_04")
        #     let chosen = if ($root_coppelia | path exists) { $root_coppelia } else if ($alt_coppelia | path exists) { $alt_coppelia } else { null }
        #     if $chosen == null {
        #         print "[EB-Man] CoppeliaSim directory not found in either expected path.";
        #         print $"[EB-Man] Expected one of: ($root_coppelia) or ($alt_coppelia)";
        #         "ERROR_MANIPULATION" | job send $parent_job_id
        #     } else {
        #         # Ensure usrset.txt exists
        #         let sys_dir = ($chosen | path join "system")
        #         let usrset = ($sys_dir | path join "usrset.txt")
        #         if not ($usrset | path exists) { mkdir $sys_dir | ignore; touch $usrset }
        #         # Prepare environment vars
        #         let ld_old = ($env.LD_LIBRARY_PATH? | default "")
        #         let ld_new = (if ($ld_old | is-empty) { $chosen } else { $"($chosen):($ld_old)" })
        #         load-env {
        #             COPPELIASIM_ROOT: $chosen
        #             LD_LIBRARY_PATH: $ld_new
        #             QT_QPA_PLATFORM_PLUGIN_PATH: $chosen
        #         }
        #         # Attempt PyRep import; install if missing
        #         let pyrep_test = (bash -c "conda run -n embench_man python -c 'import pyrep'" | complete)
        #         if $pyrep_test.exit_code != 0 {
        #             print "[EB-Man] PyRep not found. Attempting local install..."
        #             let pyrep_src = ($env.PWD | path join "embodiedbench" "envs" "eb_manipulation" "PyRep")
        #             if ($pyrep_src | path exists) {
        #                 bash -c $"conda run -n embench_man python -m pip install -e ($pyrep_src)" | ignore
        #                 let pyrep_retest = (bash -c "conda run -n embench_man python -c 'import pyrep'" | complete)
        #                 if $pyrep_retest.exit_code != 0 { print "[EB-Man] PyRep import still failing after install." }
        #             } else {
        #                 print "[EB-Man] PyRep source directory missing. Please clone stepjam/PyRep.";
        #             }
        #         }
        #         let pyrep_final = (bash -c "conda run -n embench_man python -c 'import pyrep'" | complete)
        #         if $pyrep_final.exit_code != 0 {
        #             print "[EB-Man] Aborting EB-Manipulation evaluation due to PyRep import failure.";
        #             "ERROR_MANIPULATION" | job send $parent_job_id
        #         } else {
        #             print $"Running EB-Manipulation evaluation..."

        #             let results = (bash -c $"conda run --no-capture-output -n embench_man python -m embodiedbench.main env=eb-man model_name='($env.MODEL_NAME)' exp_name='($exp_name)' ($env.EXTRA_ARGS) ($env.EXTRA_ARGS_EB_MAN) > '($log_file)' 2>&1" | complete)

        #             if $results.exit_code != 0 {
        #                 print "[EB-Manipulation] Evaluation encountered errors. Please check the log for details."
        #                 "ERROR_MANIPULATION" | job send $parent_job_id
        #             } else {
        #                 print "[EB-Manipulation] Evaluation completed successfully."
        #                 "DONE_MANIPULATION" | job send $parent_job_id
        #             }
        #         }
        #     }
        # }

        # # EB-Navigation ------------------------------------------------------
        # let navigation_job_id = job spawn {
        #     let log_file = ($log_dir | path join "EB-Navigation.log")
        #     print $"Running EB-Navigation evaluation..."

        #     let results = (bash -c $"conda run --no-capture-output -n embench_nav python -m embodiedbench.main env=eb-nav model_name='($env.MODEL_NAME)' exp_name='($exp_name)' ($env.EXTRA_ARGS) ($env.EXTRA_ARGS_EB_NAV) > '($log_file)' 2>&1" | complete)

        #     if $results.exit_code != 0 {
        #         print "[EB-Navigation] Evaluation encountered errors. Please check the log for details."
        #         "ERROR_NAVIGATION" | job send $parent_job_id
        #     } else {
        #         print "[EB-Navigation] Evaluation completed successfully."
        #         "DONE_NAVIGATION" | job send $parent_job_id
        #     }
        # }

        
        # use jobs mailbox to wait for all done/error signals
        try {
            # total expected: parallel Alfred + parallel Habitat jobs
            let total_envs = ($alfred_jobs_count + $habitat_jobs_count)

            mut processed = 0
            mut successes = 0
            mut failures = 0

            loop {
                if $processed >= $total_envs { break }

                let msg = job recv
                match $msg {
                    {env: 'alfred', status: 'done', set: $s} => {
                        $processed = $processed + 1
                        $successes = $successes + 1
                        print $"[MAILBOX] EB-ALFRED set=($s) success. Progress: ($processed)/($total_envs)"
                        $"EB-ALFRED:($s)" | save --append $prior_done_file
                    },
                    {env: 'alfred', status: 'error', set: $s} => {
                        $processed = $processed + 1
                        $failures = $failures + 1
                        print $"[MAILBOX] EB-ALFRED set=($s) failure. Progress: ($processed)/($total_envs)"
                    },
                    {env: 'habitat', status: 'done', set: $s} => {
                        $processed = $processed + 1
                        $successes = $successes + 1
                        print $"[MAILBOX] EB-Habitat set=($s) success. Progress: ($processed)/($total_envs)"
                        $"EB-Habitat:($s)" | save --append $prior_done_file
                    },
                    {env: 'habitat', status: 'error', set: $s} => {
                        $processed = $processed + 1
                        $failures = $failures + 1
                        print $"[MAILBOX] EB-Habitat set=($s) failure. Progress: ($processed)/($total_envs)"
                    },
                    _ => { print $"[WARN] Received unknown message in mailbox: ($msg)" }
                }
            }

            if ($failures == 0) and ($successes == $total_envs) {
                $"ALL DONE OK successes=($successes) failures=($failures)" | save --append $prior_done_file
                print $"All evaluations completed SUCCESSFULLY. Successes=($successes) Failures=($failures) (markers written to: ($prior_done_file))"
            } else {
                $"ALL DONE WITH FAILURES successes=($successes) failures=($failures)" | save --append $prior_done_file
                print $"Evaluations finished with FAILURES. Successes=($successes) Failures=($failures). Run will NOT be skipped next time. (markers written to: ($prior_done_file))"
            }
        } catch { |err|
            print "Interrupted while waiting for evaluations to complete (likely Ctrl-C). Cleaning up related jobs..."

            print "Currently running jobs:"
            job list

            for jid in $alfred_job_ids_snapshot { try { job kill $jid } }
            for jid in $habitat_job_ids_snapshot { try { job kill $jid } }
            # try { job kill $manipulation_job_id }
            # try { job kill $navigation_job_id }
        }
    }
}