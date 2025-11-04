#!/usr/bin/env nu

# Evaluation script using shared `run_batch` helper.
# Configuration: temps [0.6]; max_tokens [4000,6000]; replicates 3.
# Debug env: inputs=0 outputs=1.

use _common.nu *

let temps = [0.6]
let max_tokens_list = [6192]
let extra_env = { 
    DEBUG_REMOTE_MODEL_INPUTS: "0"
    DEBUG_REMOTE_MODEL_OUTPUTS: "1" 
    ONLY_ONE_STEP_PLAN: "1"    
    EMB_PLANNER_RETRY_TIMES: "3"
    EMB_REASONING_MODE: "1"
    REMOTE_URL: "http://localhost:43001/v1,http://localhost:43002/v1,http://localhost:43003/v1"
}

## To disable interactive pauses, add --no_pause to the run_batch call line.
let result = (
	run_batch
	$temps              # temperatures
	$max_tokens_list    # max token sizes
	$extra_env          # extra environment variables
	--stop_seqs "</answer>"
	--prefix "cleaned_videos_unprocessed_split1_min_1.0_num_4.9K_videos_ActionCorrection_OneStepPlanLoose_3Retries_WithReasoning_"
	# --no_pause        # uncomment to skip interactive prompts
)

print $"Status: ($result.status).";
