#!/usr/bin/env bash
# scripts/benchmark.sh — Latency benchmark for the BGE-M3 embedding server
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:10631}"
API_KEY="${2:-${EMBEDDING_API_KEY:-local}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=== Benchmark: BGE-M3 Embedding Server ==="
echo "Target: $BASE_URL"
echo ""

# Generate N copies of a sample sentence
sample_text="The quick brown fox jumps over the lazy dog near the riverbank."

for BATCH_SIZE in 1 4 8 16; do
    # Build JSON array of $BATCH_SIZE copies
    INPUT_JSON=$("$PYTHON_BIN" -c "
import json
texts = ['$sample_text'] * $BATCH_SIZE
print(json.dumps({'input': texts, 'model': 'BAAI/bge-m3'}))
")

    echo "--- Batch size: $BATCH_SIZE ---"
    # Run 3 iterations, report each
    for i in 1 2 3; do
        TIME_SEC=$(curl -sf -o /dev/null -w "%{time_total}" \
            "$BASE_URL/v1/embeddings" \
            -H "Authorization: Bearer $API_KEY" \
            -H "Content-Type: application/json" \
            -d "$INPUT_JSON")
        TIME_MS=$("$PYTHON_BIN" -c "print(f'{float(\"$TIME_SEC\") * 1000:.0f}')")
        echo "  Run $i: ${TIME_MS}ms"
    done
    echo ""
done

echo "=== Benchmark complete ==="
