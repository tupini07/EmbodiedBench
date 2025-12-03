#!/usr/bin/env nu

# Analyze EmbodiedBench run artifacts and surface problems:
#   * Missing or partial runs vs expected runs derived from experiment_specs.yaml
#   * Runs with failures ("ALL DONE WITH FAILURES" marker)
#   * Incomplete eval set coverage (missing EB-ALFRED / EB-Habitat set markers)
#   * Extraneous log or running artifacts not belonging to any enabled experiment
#   * Segfault indicators in habitat logs
#   * Lines containing "File name too long" in any log
#
# Optional cleanup (interactive or flag) for extraneous artifacts.


const this_script_path = (path self)

const run_configs_dir = ($this_script_path | path dirname)
const yaml_path = ($run_configs_dir | path join run_configs experiment_specs.yaml)
const logs_dir = ($run_configs_dir | path dirname | path dirname | path join logs)  # ../../logs relative to this file
const running_dir = ($run_configs_dir | path dirname | path dirname | path join running)

# Expected evaluation sets (mirrors logic in run_basic_evals)
const EB_ALFRED_SETS = ["base" "common_sense" "complex_instruction" "spatial" "visual_appearance" "long_horizon"]
const EB_HABITAT_SETS = ["base" "common_sense" "complex_instruction" "spatial_relationship" "visual_appearance" "long_horizon"]

#--------------------------------------------------------------------------------------------------
# Utility: Load and normalize experiment specs -> list<record>
#--------------------------------------------------------------------------------------------------
def load-specs [] {
	if not ($yaml_path | path exists) { error make {msg: $"Spec file not found: ($yaml_path)"} }
	# 'open' already parses YAML into a record; don't pipe through 'from yaml' again.
	let raw = (open $yaml_path)
	let base_rep = ($raw.base_settings.replicates? | default 3)
	let exps = ($raw.experiments? | default [])
	$exps | each {|e|
		let prefix = ($e.prefix? | default ($e.name? | default "NO_PREFIX"))
		let temps = ($e.temperature? | default [])
		let max_tokens = ($e.max_tokens? | default [])
		let reps = ($e.replicates? | default $base_rep)
		let draft = ($e.draft? | default false | into bool)
		let enabled = (if ($e.enabled? | is-empty) { true } else { ($e.enabled | into bool) })
		$e | merge {
			prefix: $prefix
			temps: $temps
			max_tokens: $max_tokens
			replicates: $reps
			draft: $draft
			enabled: $enabled
		}
	}
}

# Normalize float formatting to match run_batch interpolation (strip trailing .0)
def normalize-temp [n:number] {
	let s = ($n | into string)
	if ($s | str ends-with '.0') { $s | str replace '.0' '' } else { $s }
}

#--------------------------------------------------------------------------------------------------
# Compute expected run_names for all enabled, non-draft experiments
#--------------------------------------------------------------------------------------------------
def expected-runs [specs:list<record>] {
	# Return simple records without embedding full spec (simplifies downstream access)
	$specs | where {|s| (not $s.draft) and $s.enabled } | each {|e|
		$e.temps | each {|t|
			let tstr = (normalize-temp $t)
			$e.max_tokens | each {|mt|
				(seq 1 $e.replicates) | each {|rep|
					{run_name: $"($e.prefix)temp($tstr)_maxTokens($mt)_rep($rep)", prefix: $e.prefix, temp: $tstr, max_tokens: $mt, rep: $rep}
				} | flatten
			} | flatten
		} | flatten
	} | flatten
}

# Helper: find spec by prefix (first match)
def spec-for-prefix [specs:list<record>, pfx:string] { $specs | where {|s| $s.prefix == $pfx } | first }

#--------------------------------------------------------------------------------------------------
# Gather artifacts for a single run_name
#--------------------------------------------------------------------------------------------------
def inspect-run [run_name:string, spec:record] {
	let dones_file = ($running_dir | path join dones | path join $"($run_name)_dones.txt")
	let has_dones = ($dones_file | path exists)
	let dones_content = (if $has_dones { open $dones_file | str trim } else { "" })
	let status = if (not $has_dones) {
		"missing"
	} else if ($dones_content | str contains "ALL DONE OK") {
		"success"
	} else if ($dones_content | str contains "ALL DONE WITH FAILURES") {
		"failure"
	} else {
		"partial"
	}

	# Extract completed sets markers
	let completed_lines = ($dones_content | lines | where ($it | str starts-with "EB-"))
	let done_alfred = ($completed_lines | where ($it | str starts-with "EB-ALFRED:") | each {|l| $l | str replace 'EB-ALFRED:' '' } )
	let done_habitat = ($completed_lines | where ($it | str starts-with "EB-Habitat:") | each {|l| $l | str replace 'EB-Habitat:' '' } )

	let extra_env = ($spec.extra_env? | default {})
	let expect_alfred = ($extra_env.RUN_ALFRED? | default "1") == "1"
	let expect_habitat = ($extra_env.RUN_HABITAT? | default "1") == "1"

	let missing_alfred = if $expect_alfred { ($EB_ALFRED_SETS | where {|s| not ($done_alfred | any {|x| $x == $s}) }) } else { [] }
	let missing_habitat = if $expect_habitat { ($EB_HABITAT_SETS | where {|s| not ($done_habitat | any {|x| $x == $s}) }) } else { [] }

	# Scan logs directory for this run (may have env subdirs); gather issues.
	let run_logs_dir = ($logs_dir | path join $run_name)
	let log_exists = ($run_logs_dir | path exists)
	mut segfault_sets = []
	mut long_name_hits = []
	if $log_exists {
		let all_logs = (ls $run_logs_dir ** | where type == file)
		for lf in $all_logs { 
			let content = (try { open $lf.name } catch { "" })
			if ($content | str contains "Segmentation fault") { $segfault_sets = ($segfault_sets ++ [$lf.name]) }
			if ($content | str contains "File name too long") { $long_name_hits = ($long_name_hits ++ [$lf.name]) }
		}
	}

	{
		run_name: $run_name
		status: $status
		dones_file: (if $has_dones { $dones_file } else { null })
		done_alfred: $done_alfred
		done_habitat: $done_habitat
		missing_alfred: $missing_alfred
		missing_habitat: $missing_habitat
		segfault_logs: $segfault_sets
		long_name_logs: $long_name_hits
		has_logs: $log_exists
	}
}

#--------------------------------------------------------------------------------------------------
# Detect extraneous artifacts not matching expected run_names
#--------------------------------------------------------------------------------------------------
def find-extras [expected:list<string>] {
	let expected_set = $expected
	let log_dirs = (if ($logs_dir | path exists) { ls $logs_dir | where type == dir | each {|d| $d.name | path basename } } else { [] })
	let dones_files = (if ($running_dir | path exists) { ls ($running_dir | path join dones) | where type == file | each {|f| $f.name | path basename | str replace '_dones.txt' '' } } else { [] })
	let all_found = ($log_dirs ++ $dones_files) | uniq
	$all_found | where {|r| not ($expected_set | any {|x| $x == $r}) } | sort
}

#--------------------------------------------------------------------------------------------------
# Main entry
#--------------------------------------------------------------------------------------------------
def main [
	--prefix_filter:string = "",         # restrict to run_names whose prefix contains this substring
	--show_extras = true,                 # whether to list unexpected artifacts
	--remove_extras = false,              # delete extraneous logs + dones (DANGEROUS)
	--json = false                        # output JSON instead of table
] {
	let specs = (load-specs)
	let exp_runs = (expected-runs $specs)
	let filtered = if ($prefix_filter | str length) > 0 { $exp_runs | where {|r| $r.prefix | str contains $prefix_filter } } else { $exp_runs }

	let inspected = ($filtered | each {|r| 
		let spec = (spec-for-prefix $specs $r.prefix)
		inspect-run $r.run_name $spec
	})

	let extras = if $show_extras { find-extras ($exp_runs | each {|r| $r.run_name }) } else { [] }

	if $remove_extras and ($extras | length) > 0 {
		print $"(ansi red_bold)[CLEANUP] Removing extraneous artifacts... (ansi reset)"
		for ex in $extras {
			let log_dir = ($logs_dir | path join $ex)
			if ($log_dir | path exists) { try { rm -r $log_dir } catch { print $"Failed to rm log_dir ($log_dir)" } }
			let dones_file = ($running_dir | path join dones | path join $"($ex)_dones.txt")
			if ($dones_file | path exists) { try { rm $dones_file } catch { print $"Failed to rm dones_file ($dones_file)" } }
		}
	}

	let summary = $inspected | each {|r|
		mut problems = []
		if $r.status == 'missing' { $problems = ($problems ++ ['NO_DONES']) }
		if $r.status == 'partial' { $problems = ($problems ++ ['PARTIAL']) }
		if ($r.missing_alfred | length) > 0 { $problems = ($problems ++ ['MISSING_ALFRED_SETS']) }
		if ($r.missing_habitat | length) > 0 { $problems = ($problems ++ ['MISSING_HABITAT_SETS']) }
		if ($r.segfault_logs | length) > 0 { $problems = ($problems ++ ['SEGFAULT_LOGS']) }
		if ($r.long_name_logs | length) > 0 { $problems = ($problems ++ ['LONG_NAME_ERR']) }
		if $r.status == 'failure' { $problems = ($problems ++ ['RUN_FAILURE']) }
		$r | merge {problems: $problems}
	}

	let problems_only = ($summary | where {|s| ($s.problems | length) > 0 })

	if $json {
		{summary: $summary, problems: $problems_only, extras: $extras} | to json | print
	} else {
		print "==================== Problem Runs ===================="
		if ($problems_only | length) == 0 { print "(none)" } else { $problems_only | select run_name status problems missing_alfred missing_habitat segfault_logs long_name_logs | sort-by run_name | table -e }
		print "\n==================== All Expected Runs ===================="
		$summary | select run_name status missing_alfred missing_habitat problems | sort-by run_name | table -e
		if $show_extras {
			print "\n==================== Extraneous Artifacts ===================="
			if ($extras | length) == 0 { print "(none)" } else { $extras | each {|e| {artifact: $e}} | table -e }
		}
		print "\n(Use --json for machine-readable output; add --remove_extras to delete extraneous artifacts.)"
	}
}

main
