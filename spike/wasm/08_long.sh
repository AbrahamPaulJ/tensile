#!/bin/bash
# The slow checks after the half-burn change, in order; each logs to build/.
root=/mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge
cd "$root"; mkdir -p build
s=$(date +%s); bash tools/m3_check.sh > build/m3_check.log 2>&1; echo "m3_check exit $? $(( $(date +%s) - s )) s"
s=$(date +%s); python3 tools/m3_mutants.py --fresh > build/mutants.log 2>&1; echo "mutants exit $? $(( $(date +%s) - s )) s"
s=$(date +%s); bash tools/m4_numbers.sh > build/m4_numbers.log 2>&1; echo "m4_numbers exit $? $(( $(date +%s) - s )) s"
