#!/usr/bin/env nu

# Test script to drop into a Docker container with EmbodiedBench
# This script starts an interactive bash shell inside the Docker container for manual testing
# 
# Usage: 
#   nu test_docker_run.nu [--vllm-ports ports] [--registry registry] [--tag tag]
#
# Example:
#   nu test_docker_run.nu --vllm-ports 8000
#   nu test_docker_run.nu --vllm-ports 8000,8001 --tag latest

def main [
    --vllm-ports: string = "64215",                      # Comma-separated vLLM ports to expose
    --vllm-host: string = "localhost",       # vLLM host (use host.docker.internal to access host machine)
    --registry: string = "atupinirgacr",                # Azure Container Registry name
    --tag: string = "latest",                           # Docker image tag
    --gpus: string = "all",                             # GPU devices to use
    --skip-pull = false                                 # Skip pulling the image from registry
] {
    let image_name = "embodiedbenchrunner"
    let registry_url = $"($registry).azurecr.io"
    let full_image_name = $"($registry_url)/($image_name):($tag)"
    
    print "=========================================="
    print "EmbodiedBench Docker Interactive Shell"
    print "=========================================="
    print $"Image: ($full_image_name)"
    print $"vLLM endpoints will be accessible at: ($vllm_host):($vllm_ports)"
    print $"GPU access: ($gpus)"
    print ""
    
    # Pull image from registry unless skip-pull is set
    if not $skip_pull {
        print "[PULL] Pulling Docker image from registry..."
        print $"If this fails, make sure you're logged in: az acr login -n ($registry)"
        print ""
        
        ^docker pull $full_image_name
        
        if $env.LAST_EXIT_CODE != 0 {
            print $"(ansi red_bold)Error: Failed to pull image from registry.(ansi reset)"
            print "You can:"
            print $"  1. Login to ACR: az acr login -n ($registry)"
            print "  2. Use --skip-pull if the image already exists locally"
            print "  3. Build the image locally first"
            exit 1
        }
        
        print ""
        print "[PULL] Image pulled successfully"
        print ""
    } else {
        print "[SKIP] Skipping image pull (using local image)"
        print ""
    }
    
    # Construct port mappings
    # let port_mappings = ($vllm_ports | split row ',' | each {|port| $"-p ($port):($port)"} | str join ' ')
    
    print "[RUN] Starting interactive Docker container..."
    print ""
    print "Container configuration:"
    print "  - GPU access: enabled"
    print "  - Shared memory: 256GB"
    print "  - IPC mode: host"
    print "  - Working directory: /workspace"
    print $"  - Port mappings: ($vllm_ports)"
    print ""
    print "Once inside, you can run experiments with:"
    print $"  nu scripts/run_configs/run_single_in_docker.nu <prefix> --vllm-ports ($vllm_ports) --vllm-host ($vllm_host)"
    print ""
    print "Or explore the environment:"
    print "  conda env list                    # See available conda environments"
    print "  ls scripts/run_configs/           # See available scripts"
    print "  cat scripts/run_configs/experiment_specs.yaml  # View experiments"
    print ""
    print "=========================================="
    print ""
    
    # Run interactive bash shell in Docker
    # Mount current directory to /workspace so results are persisted
    let workspace_path = (pwd)
    
    # Build the docker run command with port mappings
    let docker_cmd = $"docker run -it --rm --gpus ($gpus) --shm-size 256g --ipc host --network host -v ($workspace_path):/workspace -w /workspace -e DEBIAN_FRONTEND=noninteractive ($full_image_name) /bin/bash -l"
    
    print $"[DEBUG] Running: ($docker_cmd)"
    print ""
    
    bash -c $docker_cmd
}
