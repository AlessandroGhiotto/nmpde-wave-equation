#!/bin/bash
set -e

echo "Submitting scaling jobs..."

for p in 1 2 4 8 16; do
    echo "  -> submitting scaling_${p}.pbs"
    qsub "scaling_${p}.pbs"
done

echo "All jobs submitted."
