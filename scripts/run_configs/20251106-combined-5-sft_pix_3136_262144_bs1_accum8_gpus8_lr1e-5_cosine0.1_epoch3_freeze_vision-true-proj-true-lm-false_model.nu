#!/usr/bin/env nu

use _common.nu *

let temps = [0.6]
let max_tokens_list = [6144]

let extra_env = { 
    DEBUG_REMOTE_MODEL_INPUTS: "0"
    DEBUG_REMOTE_MODEL_OUTPUTS: "1" 
    EMB_PLANNER_RETRY_TIMES: "3"
    ONLY_ONE_STEP_PLAN: "1"    
    EMB_REASONING_MODE: "1"
	EB_SUPRESS_DURATION_LOGS: "0"
    REMOTE_URL: "http://localhost:48001/v1,http://localhost:48002/v1,http://localhost:48003/v1"

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
	--prefix "20251106-combined-5-sft_pix_3136_262144_bs1_accum8_gpus8_lr1e-5_cosine0.1_epoch3_freeze_vision-true-proj-true-lm-false_model"
	--no_pause true       # uncomment to skip interactive prompts
)

print $"Status: ($result.status).";
