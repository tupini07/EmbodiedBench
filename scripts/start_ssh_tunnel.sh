#!/usr/bin/env bash

job_name=$1

if [ -z "$job_name" ]; then
  echo "Usage: $0 <job_name>"
  exit 1
fi

amlt ssh $job_name -o "-L 43289:localhost:43289"
