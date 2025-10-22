#!/usr/bin/env bash

# Complete EmbodiedBench evaluation workflow:
# 1. Wait for SSH tunnel to vLLM server to be established
# 2. Run all evaluations in parallel using Python threading  

job_name=$1
exp_name=$2
override_port=$3

if [ -z "$job_name" ] || [ -z "$exp_name" ]; then
    echo "Usage: $0 <job_name> <exp_name> [port]"
    echo ""
    echo "Arguments:"
    echo "  job_name: Name identifier (used for logging; use 'local' for local runs)"
    echo "  exp_name: Experiment name for the evaluation"
    echo "  port:     (Optional) Port number where vLLM server is running"
    echo ""
    echo "Modes of Operation:"
    echo ""
    echo "  1. Local vLLM server (port provided):"
    echo "     - Directly connects to vLLM server on specified port"
    echo "     - No amlt/SSH tunnel needed"
    echo "     Example: $0 local my-experiment 8000"
    echo ""
    echo "  2. Remote vLLM server via amlt (no port):"
    echo "     - Script finds unused local port and shows SSH tunnel command"
    echo "     - You must run the tunnel command in another terminal"
    echo "     Example: $0 my-cluster-job my-experiment"
    echo ""
    echo "Prerequisites (for remote mode):"
    echo "  1. vLLM server running on remote cluster on port 43289"
    echo "  2. SSH tunnel in another terminal (command will be shown)"
    echo ""
    echo "The script will:"
    echo "  - Verify vLLM server is accessible via /v1/models endpoint"
    echo "  - Run all EmbodiedBench evaluations in parallel"
    echo "  - Save results with the specified experiment name"
    exit 1
fi

# Function to find an unused port starting from a base port
find_unused_port() {
    local base_port=${1:-43289}
    local port=$base_port
    
    while true; do
        # Check if port is in use using netstat or ss
        if command -v ss >/dev/null 2>&1; then
            if ! ss -tuln | grep -q ":$port "; then
                echo $port
                return 0
            fi
        elif command -v netstat >/dev/null 2>&1; then
            if ! netstat -tuln 2>/dev/null | grep -q ":$port "; then
                echo $port
                return 0
            fi
        else
            # Fallback: try to bind to the port briefly
            if timeout 1 bash -c "</dev/tcp/localhost/$port" 2>/dev/null; then
                # Port is in use, try next
                :
            else
                # Port is free
                echo $port
                return 0
            fi
        fi
        
        port=$((port + 1))
        
        # Safety check to avoid infinite loop
        if [ $port -gt $((base_port + 100)) ]; then
            echo "Error: Could not find unused port in range $base_port to $((base_port + 100))" >&2
            return 1
        fi
    done
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    echo "Please manually close the SSH tunnel in the other terminal when done."
    echo "Cleanup completed. (vLLM server left running on remote cluster)"
}

# Set up cleanup trap
trap cleanup EXIT INT TERM

# Determine which port to use
if [ -n "$override_port" ]; then
    # Use the provided port directly
    LOCAL_PORT=$override_port
    echo "Using provided port: $LOCAL_PORT"
    echo "Job name: $job_name"
    echo "Experiment name: $exp_name"
    echo ""
    echo "⏳ Checking if vLLM server is accessible on port ${LOCAL_PORT}..."
    echo ""
else
    # Find unused port
    echo "Finding unused local port..."
    LOCAL_PORT=$(find_unused_port 43289)
    if [ $? -ne 0 ]; then
        echo "Failed to find unused port"
        exit 1
    fi

    echo "Using local port: $LOCAL_PORT"
    echo "Job name: $job_name"
    echo "Experiment name: $exp_name"

    # amlt ssh "prepared-gelding" -o "StrictHostKeyChecking=no" -o "-4 -L 43289:localhost:43289"

    # Step 1: Wait for vLLM server to be ready
    echo ""
    echo "Please ensure SSH tunnel is running in another terminal:"
    echo ""
    echo "    amlt ssh $job_name -o \"StrictHostKeyChecking=no\" -o \"-4 -L ${LOCAL_PORT}:localhost:43289\""
    echo ""
    echo ""
    echo "⏳ Waiting for tunnel to establish and server to respond..."
    echo ""
    echo ""
fi

attempt=1
while true; do
    if curl -s "http://localhost:${LOCAL_PORT}/v1/models" >/dev/null 2>&1; then
        echo "✅ Connection successful! SSH tunnel and vLLM server are ready."
        break
    fi
    
    # Show progress every minute (every 6 attempts)
    if [ $((attempt % 6)) -eq 0 ]; then
        minutes=$((attempt / 6))
        echo "⏳ Still waiting..."
    fi
    
    sleep 5
    attempt=$((attempt + 1))
done

# Step 3: Run all EmbodiedBench evaluations in parallel
echo ""
echo "📜 Starting EmbodiedBench evaluations in parallel"
echo "📊 Experiment name: $exp_name"
echo "🌐 vLLM server URL: http://localhost:$LOCAL_PORT/v1"
echo ""
echo "Press Ctrl+C to stop evaluations and clean up everything"
echo ""

# Set environment variables for the evaluation script
export REMOTE_URL="http://localhost:${LOCAL_PORT}/v1"
export remote_url="http://localhost:${LOCAL_PORT}/v1"
export OPENAI_API_KEY='empty'

# Run evaluations in parallel using the Python script
python scripts/run_evals_parallel.py "$exp_name" --remote-url "http://localhost:${LOCAL_PORT}/v1"

eval_exit_code=$?

echo ""
if [ $eval_exit_code -eq 0 ]; then
    echo "🎉 All evaluations completed successfully!"
else
    echo "❌ Some evaluations failed (exit code: $eval_exit_code)"
fi

# Cleanup will be handled by the trap
exit $eval_exit_code