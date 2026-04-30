#!/usr/bin/env bash
# scripts/smoke_test.sh — Smoke test for the BGE-M3 embedding server
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:10631}"
API_KEY="${2:-local}"

echo "=== Smoke Test: BGE-M3 Embedding Server ==="
echo "Target: $BASE_URL"
echo ""

# Test 1: Health check (no auth required)
echo "--- Test 1: GET /health ---"
HEALTH=$(curl -sf "$BASE_URL/health")
echo "$HEALTH" | python3 -m json.tool
DIM=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin)['embedding_dimension'])")
if [ "$DIM" != "1024" ]; then
    echo "FAIL: expected dimension 1024, got $DIM"
    exit 1
fi
echo "PASS: dimension is 1024"
echo ""

# Test 2: Models list
echo "--- Test 2: GET /v1/models ---"
curl -sf "$BASE_URL/v1/models" \
    -H "Authorization: Bearer $API_KEY" | python3 -m json.tool
echo "PASS"
echo ""

# Test 3: Single string embedding
echo "--- Test 3: POST /v1/embeddings (single string) ---"
RESP=$(curl -sf "$BASE_URL/v1/embeddings" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"input": "Hello, world!", "model": "BAAI/bge-m3"}')
EMB_DIM=$(echo "$RESP" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['data'][0]['embedding']))")
if [ "$EMB_DIM" != "1024" ]; then
    echo "FAIL: expected 1024-d embedding, got $EMB_DIM"
    exit 1
fi
echo "PASS: embedding is 1024-d"
echo ""

# Test 4: List input embedding
echo "--- Test 4: POST /v1/embeddings (list input) ---"
RESP=$(curl -sf "$BASE_URL/v1/embeddings" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"input": ["first text", "second text"], "model": "BAAI/bge-m3"}')
COUNT=$(echo "$RESP" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['data']))")
if [ "$COUNT" != "2" ]; then
    echo "FAIL: expected 2 embeddings, got $COUNT"
    exit 1
fi
echo "PASS: got 2 embeddings"
echo ""

# Test 5: Auth rejection
echo "--- Test 5: Auth rejection (no key) ---"
HTTP_CODE=$(curl -so /dev/null -w "%{http_code}" "$BASE_URL/v1/embeddings" \
    -H "Content-Type: application/json" \
    -d '{"input": "test", "model": "BAAI/bge-m3"}')
if [ "$HTTP_CODE" != "401" ]; then
    echo "FAIL: expected 401, got $HTTP_CODE"
    exit 1
fi
echo "PASS: got 401"
echo ""

echo "=== All smoke tests passed ==="
