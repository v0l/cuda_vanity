#!/bin/bash

# Benchmark script for CUDA vanity generator

if [ $# -ne 1 ]; then
    echo "Usage: $0 <pattern>"
    echo "Example: $0 test"
    exit 1
fi

PATTERN=$1
ITERATIONS=5

# Get GPU info
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_COMPUTE=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)

echo "========================================"
echo "CUDA Vanity Address Benchmark"
echo "========================================"
if [ -n "$GPU_NAME" ]; then
    echo "GPU: $GPU_NAME"
    if [ -n "$GPU_COMPUTE" ]; then
        echo "Compute Capability: $GPU_COMPUTE"
    fi
fi
echo "Pattern: npub1$PATTERN"
echo "Iterations: $ITERATIONS"
echo "========================================"
echo ""

# Benchmark CUDA version
echo "Running benchmark..."
HASH_RATES=()
for i in $(seq 1 $ITERATIONS); do
    echo -n "  Run $i/$ITERATIONS... "
    timeout 60 ./vanity_npub $PATTERN > /tmp/cuda_output.txt 2>&1
    
    if grep -q "Found matching npub" /tmp/cuda_output.txt; then
        # Extract hash rate from output (looks for "avg XXXXXX.XX keys/s")
        RATE=$(grep "avg" /tmp/cuda_output.txt | grep -oP 'avg \K[0-9]+\.[0-9]+' | head -1)
        if [ -n "$RATE" ]; then
            HASH_RATES+=($RATE)
            echo "${RATE} keys/s"
        else
            echo "Could not parse hash rate"
        fi
    else
        echo "TIMEOUT or FAILED"
    fi
done

echo ""
echo "========================================"
echo "Results Summary"
echo "========================================"

if [ ${#HASH_RATES[@]} -gt 0 ]; then
    # Calculate min, max, avg
    MIN=${HASH_RATES[0]}
    MAX=${HASH_RATES[0]}
    SUM=0
    
    for rate in "${HASH_RATES[@]}"; do
        SUM=$(echo "$SUM + $rate" | bc)
        if (( $(echo "$rate < $MIN" | bc -l) )); then
            MIN=$rate
        fi
        if (( $(echo "$rate > $MAX" | bc -l) )); then
            MAX=$rate
        fi
    done
    
    AVG=$(echo "scale=2; $SUM / ${#HASH_RATES[@]}" | bc)
    
    echo "Successful runs: ${#HASH_RATES[@]}/$ITERATIONS"
    echo ""
    echo "Hash Rate Statistics:"
    echo "  Min: $(printf "%.2f" $MIN) keys/s"
    echo "  Max: $(printf "%.2f" $MAX) keys/s"
    echo "  Avg: $(printf "%.2f" $AVG) keys/s"
else
    echo "No successful runs"
fi

echo "========================================"
