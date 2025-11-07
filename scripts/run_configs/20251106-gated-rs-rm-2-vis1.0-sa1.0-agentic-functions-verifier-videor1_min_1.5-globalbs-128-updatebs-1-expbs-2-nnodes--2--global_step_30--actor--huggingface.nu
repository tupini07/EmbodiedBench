#!/usr/bin/env nu

use _common.nu *

let temps = [0.6]
let max_tokens_list = [6144, 8192]

let amlt_job_names = [
	prepared-crab
	talented-serval
	robust-kodiak
]

let remote_urls = "http://localhost:50001/v1,http://localhost:50002/v1,http://localhost:50003/v1"

start_ssh_tunnels $amlt_job_names $remote_urls

let extra_env = { 
    DEBUG_REMOTE_MODEL_INPUTS: "0"
    DEBUG_REMOTE_MODEL_OUTPUTS: "1" 
    EMB_PLANNER_RETRY_TIMES: "3"
    ONLY_ONE_STEP_PLAN: "1"    
    EMB_REASONING_MODE: "1"
	EB_SUPRESS_DURATION_LOGS: "0"
    REMOTE_URL: $remote_urls

	WRAP_IN_CONTEXT_EXAMPLES_WITH_BOX_TAGS: "1"

	RUN_ALFRED: "1"
	RUN_HABITAT: "1"
}

## To disable interactive pauses, add --no_pause to the run_batch call line.
let result = (
	run_batch
	$temps              # temperatures
	$max_tokens_list    # max token sizes
	$extra_env          # extra environment variables
	--stop_seqs "</answer>"
	--prefix "20251106-gated-rs-rm-2-vis1.0-sa1.0-agentic-functions-verifier-videor1_min_1.5-globalbs-128-updatebs-1-expbs-2-nnodes--2--global_step_30--actor--huggingface"
	--no_pause true       # uncomment to skip interactive prompts
)

print $"Status: ($result.status).";
