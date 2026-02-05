#!/bin/bash

# Benchmark script to compare CUDA vanity generator vs rana

if [ $# -ne 1 ]; then
    echo "Usage: $0 <pattern>"
    echo "Example: $0 test"
    exit 1
fi

PATTERN=$1
ITERATIONS=3

echo "========================================"
echo "Vanity Address Benchmark"
echo "Pattern: npub1$PATTERN"
echo "Iterations: $ITERATIONS"
echo "========================================"
echo ""

# Benchmark CUDA version
echo "Testing CUDA implementation..."
CUDA_TIMES=()
for i in $(seq 1 $ITERATIONS); do
    echo -n "  Run $i/$ITERATIONS... "
    START=$(date +%s.%N)
    timeout 30 ./vanity_npub $PATTERN > /tmp/cuda_output.txt 2>&1
    END=$(date +%s.%N)
    
    if grep -q "Found matching npub" /tmp/cuda_output.txt; then
        TIME=$(echo "$END - $START" | bc)
        CUDA_TIMES+=($TIME)
        echo "${TIME}s"
    else
        echo "TIMEOUT or FAILED"
    fi
done

echo ""

# Benchmark rana
echo "Testing rana (CPU)..."
RANA_TIMES=()
for i in $(seq 1 $ITERATIONS); do
    echo -n "  Run $i/$ITERATIONS... "
    START=$(date +%s.%N)
    
    # Start rana in background and capture its PID
    rana --vanity-n-prefix $PATTERN --no-scaling > /tmp/rana_output.txt 2>&1 &
    RANA_PID=$!
    
    # Poll the output file until we find a match or timeout
    COUNTER=0
    FOUND=0
    while [ $COUNTER -lt 300 ]; do  # 300 * 0.1s = 30s
        if grep -q "npub1$PATTERN" /tmp/rana_output.txt 2>/dev/null; then
            FOUND=1
            break
        fi
        sleep 0.1
        COUNTER=$((COUNTER + 1))
    done
    
    # Kill rana process
    kill -9 $RANA_PID 2>/dev/null
    wait $RANA_PID 2>/dev/null
    
    END=$(date +%s.%N)
    
    if [ $FOUND -eq 1 ]; then
        TIME=$(echo "$END - $START" | bc)
        RANA_TIMES+=($TIME)
        echo "${TIME}s"
    else
        echo "TIMEOUT"
    fi
    
    # Clean up for next iteration
    rm -f /tmp/rana_output.txt
done

echo ""
echo "========================================"
echo "Results Summary"
echo "========================================"

# Calculate averages
if [ ${#CUDA_TIMES[@]} -gt 0 ]; then
    CUDA_AVG=$(echo "${CUDA_TIMES[@]}" | tr ' ' '+' | bc)
    CUDA_AVG=$(echo "scale=3; $CUDA_AVG / ${#CUDA_TIMES[@]}" | bc)
    echo "CUDA Average: ${CUDA_AVG}s (${#CUDA_TIMES[@]} successful runs)"
else
    echo "CUDA: No successful runs"
fi

if [ ${#RANA_TIMES[@]} -gt 0 ]; then
    RANA_AVG=$(echo "${RANA_TIMES[@]}" | tr ' ' '+' | bc)
    RANA_AVG=$(echo "scale=3; $RANA_AVG / ${#RANA_TIMES[@]}" | bc)
    echo "Rana Average: ${RANA_AVG}s (${#RANA_TIMES[@]} successful runs)"
else
    echo "Rana: No successful runs"
fi

# Calculate speedup
if [ ${#CUDA_TIMES[@]} -gt 0 ] && [ ${#RANA_TIMES[@]} -gt 0 ]; then
    SPEEDUP=$(echo "scale=2; $RANA_AVG / $CUDA_AVG" | bc)
    echo ""
    echo "CUDA is ${SPEEDUP}x faster than rana"
fi

echo "========================================"
