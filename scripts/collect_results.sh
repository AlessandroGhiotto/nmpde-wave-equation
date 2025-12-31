#!/bin/bash
set -e

OUT="scalability-results.csv"

# Grab header from first file
head -n 1 scalability-results-1.csv > "$OUT"

for p in 1 2 4 8 16; do
    tail -n +2 "scalability-results-${p}.csv" >> "$OUT"
done

echo "Merged results into $OUT"
