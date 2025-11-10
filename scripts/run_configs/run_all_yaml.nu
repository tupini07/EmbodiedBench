#!/usr/bin/env nu

use _common.nu *

# ---------------------------------------------------------------
# run_all_yaml.nu
# ---------------------------------------------------------------
# Central launcher that reads experiment_specs.yaml and spawns up to
# MAX_CONCURRENT_JOBS experiments at a time. It continuously monitors
# running jobs and launches new ones as slots become available.
#
# MAX_CONCURRENT_JOBS: Maximum number of experiments running simultaneously
#
# LOCK STRATEGY
# -------------
# A lock file per experiment prefix is stored in: running/locks/<prefix>.lock
# Lock files are used to prevent duplicate launches of the same experiment.
# The script uses process detection (via `ps a | grep 'embodiedbench.main'`) 
# to determine if an experiment is truly running, rather than relying on PIDs
# alone, as PIDs can be reused by the OS for different processes.
#
# Lock lifecycle:
#   1. Lock directory is ensured to exist on startup
#   2. Stale locks (where no matching process exists) are garbage collected
#   3. Before launching, check if experiment is already running via process grep
#   4. Create lock file when spawning new experiment
#   5. Locks are removed during garbage collection when no process is found
#
# PROCESS DETECTION
# ----------------
# Uses: `ps a | grep 'embodiedbench.main' | grep <prefix>`
# This ensures we detect actual running experiments by checking:
#   - The process command line contains 'embodiedbench.main'
#   - The process command line contains the experiment prefix
# This is more reliable than PID-based detection and matches the logic
# used in experiment_monitor.py for consistency.
#
# COMPLETION SIGNALING
# -------------------
# Each experiment creates a signal file in running/signals/<prefix>.signal
# when it completes. Background monitor jobs wait for these signals and
# notify the main loop when an experiment finishes, allowing immediate
# slot reallocation rather than waiting for the next polling interval.
#
# YAML SCHEMA
# -----------
# base_settings:
#   s: ["</answer>"]
#   replicates: 3
#   skip_if_done: true
#   no_pause: true
#   extra_env: { ... }
# experiments: list of maps each containing:
#   prefix: string (unique identifier for experiment)
#   max_tokens: list<int>
#   temperature: list<float>
#   amlt_job_names: list<string> (optional)
#   extra_env: record (overrides for environment variables)
#   draft: bool (optional, if true experiment is skipped)
#
# VALIDATION
# ----------
# On startup, the script validates that:
#   - No duplicate prefixes exist across experiments
#   - No duplicate REMOTE_URL sets exist (canonicalized by sorting)
# The script will abort if duplicates are detected.
#
# ---------------------------------------------------------------

const this_script_path = (path self)
const yaml_path = ($this_script_path | path dirname | path join experiment_specs.yaml)
const MAX_CONCURRENT_JOBS = 20

def ensure-lock-dir [] { mkdir running/locks | ignore }

def lock-file [prefix:string] { $"running/locks/($prefix).lock" }

def get-terminal-windows [] {
	# Get all gnome-terminal window titles that match our "EXP: " prefix pattern
	# This only works when running in a graphical session with X11
	# Returns list of records with prefix and window_id
	let result = (bash -c "xprop -root _NET_CLIENT_LIST 2>/dev/null | grep -o '0x[0-9a-f]*' | while read wid; do title=$(xprop -id $wid WM_NAME 2>/dev/null | grep -i 'EXP:'); if [ -n \"$title\" ]; then echo \"$wid|||$title\"; fi; done" | complete)
	
	if $result.exit_code != 0 {
		return []
	}
	
	# Parse window IDs and titles: window_id|||WM_NAME(STRING) = "EXP: prefix-here"
	$result.stdout 
		| lines 
		| parse '{window_id}|||WM_NAME(STRING) = "EXP: {prefix}"'
		| select window_id prefix
}

def get-terminal-window-prefixes [] {
	# Get just the prefixes (for backward compatibility)
	get-terminal-windows | get prefix
}

def raise-window [window_id:string] {
	# Bring a window to the front using xdotool
	# First check if xdotool is available
	let has_xdotool = (try { which xdotool | is-not-empty } catch { false })
	
	if $has_xdotool {
		# Convert hex window ID to decimal for xdotool
		let decimal_id = (bash -c $"printf '%d' ($window_id)" | complete | get stdout | str trim)
		bash -c $"xdotool windowactivate ($decimal_id) 2>/dev/null" | complete | ignore
	} else {
		# Fallback: use wmctrl if available
		let has_wmctrl = (try { which wmctrl | is-not-empty } catch { false })
		if $has_wmctrl {
			bash -c $"wmctrl -ia ($window_id) 2>/dev/null" | complete | ignore
		}
	}
}

def has-terminal-window [prefix:string] {
	# Check if there's a gnome-terminal window with this prefix
	let windows = (get-terminal-window-prefixes)
	($windows | where $it == $prefix | length) > 0
}

def is-prefix-running [prefix:string] {
	# Check if there's a __run_single_experiment.nu process running for this prefix
	# This is more reliable than checking for embodiedbench.main since that may not be
	# running continuously (e.g., waiting for remote servers, retrying habitat, etc.)
	let ps_output = (bash -c "ps a | grep '__run_single_experiment.nu' | grep -v grep" | complete)
	
	if $ps_output.exit_code != 0 {
		return false
	}
	
	# Check if any line contains the prefix as a complete space-delimited argument
	# Use word boundaries to avoid substring matches (e.g., "step20" shouldn't match "step20--actor")
	let pattern = $"__run_single_experiment.nu[[:space:]]+($prefix)[[:space:]]"
	let matching_lines = ($ps_output.stdout | lines | where { |line| 
		($line | bash -c $"grep -E '($pattern)'" | complete | get exit_code) == 0
	})
	
	if ($matching_lines | length) == 0 {
		return false
	}
	
	# Additional paranoid check: verify the process is associated with a pts (pseudo-terminal)
	# This helps catch zombie processes that aren't actually in a gnome-terminal
	# ps output format has TTY as the second column (e.g., "pts/5", "pts/6")
	let has_terminal = ($matching_lines | any { |line|
		# Check if line contains pts/N pattern (indicates a terminal session)
		($line | str contains "pts/")
	})

	if not $has_terminal {
		print $"(ansi yellow)[WARNING] Found process for prefix '($prefix)' but no associated terminal session. This may indicate a zombie process.(ansi reset)"
	}
	
	$has_terminal
}

def count-running-jobs [] {
	ensure-lock-dir
	let lock_files = (try { ls running/locks/ | where name =~ '\.lock$' } catch { [] })
	mut running_count = 0
	for lf in $lock_files {
		let basename = ($lf.name | path basename | str replace '.lock' '')
		if (is-prefix-running $basename) {
			$running_count = $running_count + 1
		}
	}
	$running_count
}

def get-all-prefixes [] {
	let spec = (open $yaml_path)
	let experiments = ($spec.experiments? | default [])
	$experiments | where { |exp| ($exp.draft? | default false) != true } | each { |exp| $exp.prefix? | default null } | where $it != null
}

def get-available-experiments [launched_prefixes: list<string>] {
	let all_prefixes = (get-all-prefixes)
	mut available = []
	for prefix in $all_prefixes {
		let is_running = (is-prefix-running $prefix)
		let already_launched = ($prefix in $launched_prefixes)
		let has_window = (has-terminal-window $prefix)
		
		# If there's a terminal window but no running process, warn the user
		if $has_window and (not $is_running) and (not $already_launched) {
			print $"(ansi yellow)[WARNING] Found terminal window for '($prefix)' but no running process. This may indicate a failed experiment.(ansi reset)"
		}
		
		# Skip if already running or already launched this session
		# Also skip if there's a terminal window (even if process died) to avoid double-launching
		if (not $is_running) and (not $already_launched) and (not $has_window) {
			$available = ($available | append $prefix)
		}
	}
	$available
}

def gc-stale-locks [] {
	ensure-lock-dir
	let lock_files = (try { ls running/locks/ | where name =~ '\.lock$' } catch { [] })
	if ($lock_files | length) == 0 { return }
	for lf in $lock_files {
		let basename = ($lf.name | path basename | str replace '.lock' '')
		let running = (is-prefix-running $basename)
		if not $running {
			# If not running anymore, remove lock regardless; optional check for done marker.
			print $"[GC] Removing stale lock for prefix: (ansi green)($basename)(ansi reset)"
			try { rm $lf.name } catch { }
		}
	}
}

def launch-experiment [prefix:string, signal_dir:string, parent_job_id: int] {
	# Spawn a new gnome-terminal tab that runs the single experiment launcher.
	let script_dir = (dirname $this_script_path)
	let single_path = ($script_dir | path join "__run_single_experiment.nu")
	if not ($single_path | path exists) {
		print (ansi red_bold) + $"Missing __run_single_experiment.nu at: ($single_path)" + (ansi reset)
		return
	}

	# Create signal file path for this experiment
	let signal_file = ($signal_dir | path join $"($prefix).signal" | path expand)

	# before starting the jobs we need to clean up any existing signal file
	if ($signal_file | path exists) {
		try { rm $signal_file }
	}
	
	gnome-terminal --window --title $"EXP: ($prefix)" -- bash -c $"nu '($single_path)' '($prefix)' '($signal_file)'; exec bash" 
	print $"(ansi green)[LAUNCHED] Started experiment: ($prefix)(ansi reset)"
	
	job spawn {
		# Wait for signal file to appear with completion status
		loop {
			if ($signal_file | path exists) {
				let status = (try { open $signal_file } catch { "unknown" })
				print $"[COMPLETION] Experiment ($prefix) finished with status: ($status)"
				break
			}
			sleep 30sec
		}

		$prefix | job send $parent_job_id
	}
}

print $"[INFO] Loading experiment specs: ($yaml_path)"
print $"[INFO] Maximum concurrent jobs: ($MAX_CONCURRENT_JOBS)"

# Initial validation
let spec = (open $yaml_path)

gc-stale-locks

# Check for existing terminal windows
let existing_windows = (get-terminal-windows)
if ($existing_windows | length) > 0 {
	print $"(ansi cyan)[INFO] Found ($existing_windows | length) existing gnome-terminal windows with experiment prefixes:(ansi reset)"
	
	# Check for duplicate windows and failed windows
	let prefixes = ($existing_windows | get prefix)
	let window_counts = ($prefixes | group-by {|x| $x} | items {|key, val| {prefix: $key, count: ($val | length)}})
	let duplicates = ($window_counts | where count > 1)
	
	mut windows_to_raise = []
	
	if ($duplicates | length) > 0 {
		print $"(ansi red_bold)[WARNING] Found duplicate terminal windows!(ansi reset)"
		for dup in $duplicates {
			print $"  (ansi red_bold)⚠ ($dup.prefix) has ($dup.count) windows(ansi reset)"
			# Add all windows with this prefix to raise list
			let dup_windows = ($existing_windows | where prefix == $dup.prefix)
			$windows_to_raise = ($windows_to_raise | append $dup_windows)
		}
		print ""
	}
	
	# Check for failed windows (no running process)
	for win in $existing_windows {
		let is_running = (is-prefix-running $win.prefix)
		if $is_running {
			print $"  ✓ ($win.prefix) (ansi green)\(running)(ansi reset)"
		} else {
			print $"  ✗ ($win.prefix) (ansi red)\(no process - may have failed)(ansi reset)"
			# Add to raise list if not already there
			if not ($win in $windows_to_raise) {
				$windows_to_raise = ($windows_to_raise | append $win)
			}
		}
	}
	
	# Raise problematic windows
	if ($windows_to_raise | length) > 0 {
		print $"(ansi yellow)[ACTION] Bringing ($windows_to_raise | length) problematic window\(s) to the front...(ansi reset)"
		for win in $windows_to_raise {
			raise-window $win.window_id
			sleep 200ms  # Brief delay between window activations
		}

		# wait for user acknowledgement
		print $"(ansi yellow)[ACTION] Please acknowledge to continue...(ansi reset)"
		input "Press Enter to continue..."
	}
	
	print ""
}

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

# ---------------------------------------------------------------
# Main monitoring loop
# ---------------------------------------------------------------
print "[START] Beginning continuous job monitoring and launching..."
print ""

# Create signal directory for experiment completion notifications
let current_job_id = (job id)

let signal_dir = "running/signals"
mkdir $signal_dir | ignore

mut launched_prefixes: list<string> = []
mut available_exps = (get-available-experiments $launched_prefixes)

loop {
	# Garbage collect stale locks
	gc-stale-locks
	
	# Count currently running jobs
	let running = (count-running-jobs)
	print $"[MONITOR] Currently running: ($running)/($MAX_CONCURRENT_JOBS) jobs"
	print $"[MONITOR] Total launched this session: ($launched_prefixes | length)"
	
	# Check if we can launch more jobs
	if $running < $MAX_CONCURRENT_JOBS {
		let available_slots = $MAX_CONCURRENT_JOBS - $running
		print $"[MONITOR] ($available_slots) slots available for new jobs"
		
		# Get available experiments (not running, not already launched)
		if ($available_exps | length) > 0 {
			print $"[MONITOR] Found ($available_exps | length) experiments waiting to run"
			
			# Launch experiments up to available slots
			let to_launch = if ($available_exps | length) < $available_slots { 
				$available_exps | length 
			} else { 
				$available_slots 
			}
			
			for i in 1..$to_launch {
				# Pick first experiment from available ones (they're already in YAML order)
				let chosen_prefix = ($available_exps | get 0)
				
				print $"[SELECT] Launching experiment: ($chosen_prefix)"

				launch-experiment $chosen_prefix $signal_dir $current_job_id
				
				# Add to launched list
				$launched_prefixes = ($launched_prefixes | append $chosen_prefix)
				
				# Brief pause between launches to avoid race conditions
				sleep 5sec
				
				# Remove from available list for this iteration
				$available_exps = ($available_exps | where $it != $chosen_prefix)
			}
		} else {
			print "[MONITOR] No more experiments available to launch"
			
			# If nothing is running and nothing is available, we're done
			if $running == 0 {
				print (ansi green_bold) + "[COMPLETE] All experiments have been completed!" + (ansi reset)
				break
			}
		}
	} else {
		print "[MONITOR] All slots occupied, waiting..."
	}

	sleep 30sec

	$available_exps = (get-available-experiments $launched_prefixes)

	# Check completion conditions
	let running = (count-running-jobs)
	if ($available_exps | length) == 0 and $running == 0 {
		print (ansi green_bold) + "[COMPLETE] All experiments have been completed!" + (ansi reset)
		break
	}
	
	# Wait for a job completion notification if there are running jobs
	if $running > 0 {
		print "Waiting for a job completion notification..."
		job recv
		print "A job has completed, re-evaluating available slots..."
	}
}

print "[DONE] run_all_yaml completed all experiments."