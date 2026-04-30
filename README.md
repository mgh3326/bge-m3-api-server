# openai-compatible-bge-m3-server

Local OpenAI-compatible embedding server for [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3). Designed to replace quota-limited cloud embedding APIs (like Gemini Embedding's 1K RPD free tier) with a zero-quota local alternative.

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-username/openai-compatible-bge-m3-server.git
cd openai-compatible-bge-m3-server
python3.12 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### 2. Run the server

```bash
# Set Intel CPU threading (adjust for your core count)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Start the server
python -m bge_m3_server
```

The server starts on `http://127.0.0.1:10631` by default. First startup downloads the model (~600MB) and takes 1-2 minutes.

### 3. Verify

```bash
# Health check
curl http://127.0.0.1:10631/health

# Generate embeddings
curl http://127.0.0.1:10631/v1/embeddings \
  -H "Authorization: Bearer local" \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!", "model": "BAAI/bge-m3"}'
```

## Configuration

All settings are controlled via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_HOST` | `127.0.0.1` | Bind address (keep loopback for security) |
| `EMBEDDING_PORT` | `10631` | Bind port |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | HuggingFace model ID |
| `EMBEDDING_API_KEY` | `local` | API key for Bearer auth |
| `EMBEDDING_BATCH_SIZE` | `8` | Max batch size |
| `OMP_NUM_THREADS` | (system) | OpenMP threads (set to core count) |
| `MKL_NUM_THREADS` | (system) | MKL threads (set to core count) |

## Using with Honcho

Add these to your Honcho `.env`:

```env
EMBEDDING_MODEL_CONFIG__TRANSPORT=openai
EMBEDDING_MODEL_CONFIG__MODEL=BAAI/bge-m3
EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL=http://127.0.0.1:10631/v1
EMBEDDING_MODEL_CONFIG__OVERRIDES__API_KEY=local
EMBEDDING_VECTOR_DIMENSIONS=1024
VECTOR_STORE_DIMENSIONS=1024
```

### IMPORTANT: Database Recreation Required

BGE-M3 produces **1024-dimensional** vectors. If your Honcho instance previously used Gemini Embedding (1536-d) or another model with different dimensions, you **must** recreate the vector store:

1. **Back up your existing database** before proceeding
2. Stop Honcho
3. Drop and recreate the database (or the vector extension tables)
4. Restart Honcho with the new env vars above
5. Run a canary test: write a memory and read it back
6. Only after the canary passes, remove your old database backup

**Do NOT drop your existing database until you have confirmed the new configuration works end-to-end.**

## Using with the OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:10631/v1",
    api_key="local",
)

response = client.embeddings.create(
    input=["Your text here"],
    model="BAAI/bge-m3",
)

# Each embedding is 1024 dimensions
print(len(response.data[0].embedding))  # 1024
```

## Running as a macOS Service (launchd)

```bash
cp launchd/com.local.bge-m3-server.plist ~/Library/LaunchAgents/
# Edit the plist to verify paths match your installation
launchctl load ~/Library/LaunchAgents/com.local.bge-m3-server.plist
```

To stop:
```bash
launchctl unload ~/Library/LaunchAgents/com.local.bge-m3-server.plist
```

Logs: `/tmp/bge-m3-server.out.log` and `/tmp/bge-m3-server.err.log`

## Testing

```bash
# Unit tests
pytest tests/ -v

# Smoke test (server must be running)
./scripts/smoke_test.sh

# Benchmark
./scripts/benchmark.sh
```

## Intel Mac CPU Performance Notes

- BGE-M3 on Intel i9 CPU with `OMP_NUM_THREADS=8`: expect ~200-500ms per single embedding, ~1-2s for batch of 16
- First request after startup is slower (model warmup); the server runs a warmup call automatically
- `use_fp16=False` is required for CPU mode (fp16 requires GPU)
- Setting `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to your physical core count (not logical) typically gives best throughput

### Future Optimization Options

When CPU performance becomes a bottleneck, consider these options (not included in v0.1):

- **ONNX Runtime:** Convert the model to ONNX format for faster CPU inference. See `myeolinmalchi/bge-m3-fastapi` for a reference implementation.
- **OpenVINO:** Intel's inference engine, optimized for Intel CPUs. Can provide 2-4x speedup on Intel hardware.
- **Quantization:** INT8 quantization can reduce memory and improve throughput with minimal quality loss.

Benchmark with `./scripts/benchmark.sh` before and after any optimization to verify improvement.

## API Reference

### GET /health
No auth required. Returns model info.
```json
{"status": "ok", "model": "BAAI/bge-m3", "embedding_dimension": 1024}
```

### GET /v1/models
Requires Bearer auth. Returns OpenAI-compatible model list.

### POST /v1/embeddings
Requires Bearer auth. Accepts `input` as string or list of strings.
```json
{
  "input": "text to embed",
  "model": "BAAI/bge-m3"
}
```
Response:
```json
{
  "object": "list",
  "data": [{"object": "embedding", "embedding": [0.1, ...], "index": 0}],
  "model": "BAAI/bge-m3",
  "usage": {"prompt_tokens": 4, "total_tokens": 4}
}
```

## License

MIT
