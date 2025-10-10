#!/usr/bin/env bash

for env_name in embench; do
   echo "Applying patches to environment: $env_name"
   cp "patches/ai2thor/controller.py" \
      "/home/aiscuser/.conda/envs/${env_name}/lib/python3.9/site-packages/ai2thor/controller.py"
done

echo "Done."