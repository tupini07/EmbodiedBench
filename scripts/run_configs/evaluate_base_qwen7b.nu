#!/usr/bin/env nu

# Evaluation script using shared `run_batch` helper.
# Configuration: temps [0.0,0.6]; max_tokens [2048,4096]; replicates 3.
# Debug env: inputs=1 outputs=1.

use _common.nu *

let temps = [0.0, 0.6]
let max_tokens_list = [2048, 4096]

let extra_env = { 
    DEBUG_REMOTE_MODEL_INPUTS: "1"
    DEBUG_REMOTE_MODEL_OUTPUTS: "1" 
    ONLY_ONE_STEP_PLAN: "1"    
    EMB_PLANNER_RETRY_TIMES: "3"
}

## To disable interactive pauses, add --no_pause to the run_batch call line.
let result = (
	run_batch
	$temps              # temperatures
	$max_tokens_list    # max token sizes
	$extra_env          # extra environment variables
	--reasoning_mode "1"
	--stop_seqs "</answer>"
	--prefix "reproduce_qwen2.5-7b_NewPrompt_NewParse_"
	# --no_pause        # uncomment to skip interactive prompts
)

print $"Status: ($result.status).";
