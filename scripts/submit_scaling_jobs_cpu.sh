#!/bin/bash
set -e

echo "Submitting scaling jobs..."

for p in 1 2 4 8 16; do
    echo "  -> submitting scaling_cpu/scaling_${p}.pbs"
    jobid="$(qsub "scaling_cpu/scaling_${p}.pbs")"
    echo "     jobid: ${jobid}"
done

echo "All jobs submitted."
