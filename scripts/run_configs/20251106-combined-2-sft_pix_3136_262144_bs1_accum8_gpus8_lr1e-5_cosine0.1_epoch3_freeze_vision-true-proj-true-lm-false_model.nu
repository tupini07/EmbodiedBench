#!/usr/bin/env nu

use _common.nu *

let temps = [0.6, 0.7]
let max_tokens_list = [6144]

let amlt_job_names = [
	sharp-sculpin
	driving-seal
	fond-parakeet
]

let remote_urls = "http://localhost:46001/v1,http://localhost:46002/v1,http://localhost:46003/v1"

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
}

## To disable interactive pauses, add --no_pause to the run_batch call line.
let result = (
	run_batch
	$temps              # temperatures
	$max_tokens_list    # max token sizes
	$extra_env          # extra environment variables
	--stop_seqs "</answer>"
	--prefix "20251106-combined-2-sft_pix_3136_262144_bs1_accum8_gpus8_lr1e-5_cosine0.1_epoch3_freeze_vision-true-proj-true-lm-false_model_ActionCorrection_OneStepPlanLoose_3Retries_WithReasoning_"
	--no_pause true       # uncomment to skip interactive prompts
)

print $"Status: ($result.status).";
