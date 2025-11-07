#!/usr/bin/env nu

use _common.nu *

# ---------------------------------------------------------------
# run_all_yaml.nu
# ---------------------------------------------------------------
# Central launcher that reads experiment_specs.yaml and spawns all
# experiments not already running. It uses simple lock files to avoid
# re-launching the same prefix while jobs are still active.
#
# LOCK STRATEGY
# -------------
# A lock file per experiment prefix is stored in: running/locks/<prefix>.lock
# Contents: timestamp + script pid. A lock persists while processes whose
# command line contains the prefix regex are alive. On each invocation we:
#   1. Ensure lock directory exists.
#   2. Garbage collect stale locks (no live processes AND dones marker present
#      OR no live processes & no partial activity).
#   3. Create new lock before spawning run_batch.
#   4. If lock exists and live processes are found, skip launching.
#
# DONE MARKER HEURISTIC
# ---------------------
# We treat an experiment "fully completed" if a dones file containing
# "ALL DONE OK" appears under running/dones with prefix substring.
# (Exact format depends on run_basic_evals; we approximate here.)
#
# YAML SCHEMA (current simplified version):
# base_settings:
#   s: ["</answer>"]
#   replicates: 3
#   skip_if_done: true
#   no_pause: true
#   extra_env: { ... }
# experiments: list of maps each containing:
#   prefix: string (unique)
#   max_tokens: list<int>
#   temperature: list<float>
#   amlt_job_names: list<string> (optional)
#   extra_env: record overrides
#
# ---------------------------------------------------------------

const this_script_path = (path self)
const yaml_path = ($this_script_path | path dirname | path join experiment_specs.yaml)

def ensure-lock-dir [] { mkdir running/locks | ignore }

def lock-file [prefix:string] { $"running/locks/($prefix).lock" }

def is-prefix-running [prefix:string] {
	let script = $"ps -eo pid,command | grep -E '($prefix)' | grep -v grep | awk '{print $1}'"
	let pids = (bash -c $script | lines | where $it != "")
	($pids | length) > 0
}

def gc-stale-locks [] {
	ensure-lock-dir
	let lock_files = (ls running/locks | where type == "File" | each {|f| $f.name })
	if ($lock_files | length) == 0 { return }
	for lf in $lock_files {
		let prefix = ($lf | str replace '.lock' '')
		let running = (is-prefix-running $prefix)
		if not $running {
			# If not running anymore, remove lock regardless; optional check for done marker.
			print $"[GC] Removing stale lock for prefix: ($prefix)"
			try { rm $"running/locks/($lf)" } catch { }
		}
	}
}

def skip-launch? [prefix:string] {
	let lf = (lock-file $prefix)
	let lock_exists = ($lf | path exists)
	if (not $lock_exists) { return false }
	let running = (is-prefix-running $prefix)
	if $running { return true } else { false }
}

print $"[INFO] Loading experiment specs: ($yaml_path)"
let spec = (open $yaml_path)

gc-stale-locks

let base = ($spec.base_settings)
let experiments = ($spec.experiments? | default [])

if ($experiments | length) == 0 {
	print (ansi red_bold) + "No experiments defined in YAML." + (ansi reset)
	exit 1
}

# ---------------------------------------------------------------
# Duplicate validation (prefix & remote URL sets)
# ---------------------------------------------------------------
do {
	mut prefix_counts = {}               # map prefix -> count
	mut remote_counts = {}               # map remote_urls (sorted canonical string) -> list of prefixes
	for exp in $experiments {
		let prefix = ($exp.prefix? | default null)
		if $prefix != null {
			let cur = ($prefix_counts | get -o $prefix | default 0)
			$prefix_counts = ($prefix_counts | upsert $prefix ($cur + 1))
		}
		let ru = ($exp.extra_env?.REMOTE_URL? | default ($exp.extra_env? | default {} | get -o REMOTE_URL | default ""))
		if ($ru | str length) > 0 {
			# canonicalize by splitting and sorting (avoid order-based false dup or ensure consistent grouping)
			let canonical = ($ru | split row ',' | sort | str join ',')
			let existing = ($remote_counts | get -o $canonical | default [])
			let new_list = ($existing | append $prefix)
			$remote_counts = ($remote_counts | upsert $canonical $new_list)
		}
	}

	# Build duplicate lists (avoid items since version differences can cause issues)
	mut dup_prefixes: list<string> = []
	for pfx in ($prefix_counts | columns) {
		let count = ($prefix_counts | get -o $pfx | default 0)
		if $count > 1 { $dup_prefixes = ($dup_prefixes | append $pfx) }
	}

	mut dup_remote: list<record<urls: string, prefixes: list<string>>> = []
	for ru in ($remote_counts | columns) {
		let owners = ($remote_counts | get -o $ru | default [])
		if ($owners | length) > 1 {
			$dup_remote = ($dup_remote | append { urls: $ru, prefixes: $owners })
		}
	}

	if (($dup_prefixes | length) > 0) or (($dup_remote | length) > 0) {
		print (ansi red_bold) + "Duplicate configuration detected. Aborting launches." + (ansi reset)
		if ($dup_prefixes | length) > 0 {
			print (ansi red_bold) + "-- Duplicate prefixes --" + (ansi reset)
			for p in $dup_prefixes { print $"   prefix: ($p)" }
		}
		if ($dup_remote | length) > 0 {
			print (ansi red_bold) + "-- Duplicate REMOTE_URL sets (canonicalized) --" + (ansi reset)
			for entry in $dup_remote {
				let urls = $entry.urls
				let owners = $entry.prefixes
				print $"   urls: ($urls) used by prefixes: ($owners | str join ',')"
			}
		}
		exit 1
	} else {
		print "[CHECK] No duplicate prefixes or REMOTE_URL sets detected."
	}
}

for exp in $experiments {
    if $exp.draft? == true {
        print $"(ansi yellow)[SKIP] Skipping draft experiment: ($exp.prefix)(ansi reset)"
        continue
    }

	let prefix = ($exp.prefix? | default null)
	if $prefix == null {
		print $"(ansi yellow)[WARN] Skipping experiment missing 'prefix' key.(ansi reset)"
		continue
	}

	if (skip-launch? $prefix) {
		print $"(ansi yellow)[SKIP] Already running or locked: (ansi gb)($prefix)(ansi reset)"
		continue
	}

	# Spawn a new gnome-terminal tab that runs the single experiment launcher.
	let script_dir = (dirname $this_script_path)
	let single_path = ($script_dir | path join "__run_single_experiment.nu")
	if not ($single_path | path exists) {
		print (ansi red_bold) + $"Missing __run_single_experiment.nu at: ($single_path)" + (ansi reset)
		continue
	}

	gnome-terminal --window --title $"EXP: ($prefix)" -- bash -c $"nu '($single_path)' '($prefix)'; exec bash" | ignore
}

print "[DONE] run_all_yaml completed launch cycle."