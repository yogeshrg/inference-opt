# Inference Optimization

A comprehensive **inference optimization toolkit** for LLMs featuring:

- ⚡ **Quantization** — AWQ, GPTQ, GGUF support with calibration
- 🚀 **Speculative Decoding** — Draft + target model for faster generation
- 💾 **KV Cache Management** — LRU eviction, disk offloading, hit rate tracking
- 📊 **Benchmarking** — Throughput, latency (p50/p99), TTFT measurement
- 🔄 **Continuous Batching** — Dynamic batching for maximum throughput

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended)
- Docker (optional)

### Local Development

```bash
cd inference-opt

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev,gpu]"

# Copy environment config
cp .env.example .env

# Run API server
python -m app.main

# Or use CLI
python -m app.cli quantize meta-llama/Meta-Llama-3.1-8B --method awq --bits 4
python -m app.cli generate "The future of AI is" --max-tokens 100
python -m app.cli benchmark meta-llama/Meta-Llama-3.1-8B --preset standard
```

### Docker Deployment (GPU)

```bash
cd docker
docker-compose up -d

# API available at http://localhost:8003/docs
```

## Features

### Quantization (AWQ, GPTQ, GGUF)

```python
import httpx

async with httpx.AsyncClient() as client:
    # Quantize with AWQ
    response = await client.post("http://localhost:8003/api/v1/quantize", json={
        "model_name": "meta-llama/Meta-Llama-3.1-8B",
        "method": "awq",
        "bits": 4,
        "group_size": 128,
        "calibration_samples": 512
    })
    result = response.json()
    print(f"Compression: {result['compression_ratio']:.2f}x")
```

### Speculative Decoding

```python
async with httpx.AsyncClient() as client:
    # Generate with speculative decoding
    response = await client.post("http://localhost:8003/api/v1/generate", json={
        "prompt": "Explain quantum computing",
        "max_tokens": 256,
        "temperature": 0.7,
        "speculative": {
            "enabled": True,
            "acceptance_threshold": 0.9,
            "max_draft_tokens": 5
        }
    })
    result = response.json()
    print(f"Acceptance rate: {result['speculative_stats']['acceptance_rate']:.1%}")
```

### KV Cache Management

```python
async with httpx.AsyncClient() as client:
    # Get cache stats
    stats = await client.get("http://localhost:8003/api/v1/kv-cache/stats")
    print(f"Hit rate: {stats.json()['hit_rate']:.1%}")
    print(f"Cache size: {stats.json()['cache_size_mb']:.1f} MB")

    # Clear cache
    await client.post("http://localhost:8003/api/v1/kv-cache/clear")
```

### Benchmarking

```python
async with httpx.AsyncClient(timeout=300) as client:
    # Run comprehensive benchmark
    response = await client.post("http://localhost:8003/api/v1/benchmark", json={
        "model_name": "meta-llama/Meta-Llama-3.1-8B",
        "prompt_lengths": [128, 512, 1024],
        "gen_lengths": [128, 256, 512],
        "batch_sizes": [1, 4, 8, 16],
        "warmup_runs": 3,
        "benchmark_runs": 10
    })
    result = response.json()
    print(f"Max throughput: {result['summary']['max_throughput_tps']:.1f} tok/s")
```

## CLI Usage

```bash
# Quantize a model
python -m app.cli quantize meta-llama/Meta-Llama-3.1-8B \
    --method awq --bits 4 --group-size 128

# Generate text with speculative decoding
python -m app.cli generate "The future of AI is" \
    --max-tokens 200 --speculative

# Run benchmark
python -m app.cli benchmark meta-llama/Meta-Llama-3.1-8B \
    --preset standard

# Check health
python -m app.cli health

# KV Cache stats
python -m app.cli kv-cache
```

## Configuration

Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `QUANT_METHOD` | `awq` | Quantization method: awq, gptq, gguf |
| `QUANT_BITS` | `4` | Quantization bits (2-8) |
| `SPECULATIVE_ENABLED` | `true` | Enable speculative decoding |
| `ACCEPTANCE_THRESHOLD` | `0.9` | Speculative acceptance threshold |
| `MAX_DRAFT_TOKENS` | `5` | Max tokens per draft step |
| `MAX_KV_CACHE_SIZE` | `1GB` | Maximum KV cache size |
| `KV_CACHE_OFFLOAD` | `false` | Enable disk offloading |
| `CONTINUOUS_BATCHING` | `true` | Enable continuous batching |
| `MAX_BATCH_SIZE` | `32` | Maximum batch size |

## Project Structure

```
inference-opt/
├── app/
│   ├── api/routes.py              # FastAPI routes
│   ├── cli.py                     # Typer CLI
│   ├── core/config.py             # Pydantic settings
│   ├── main.py                    # FastAPI app entry
│   ├── models/schemas.py          # Pydantic models
│   └── services/
│       ├── quantization_service.py # AWQ/GPTQ/GGUF quantization
│       ├── speculative_service.py  # Speculative decoding
│       ├── kv_cache_service.py     # KV cache management
│       ├── benchmark_service.py    # Performance benchmarking
│       └── inference_service.py    # Unified service
├── docker/
│   ├── Dockerfile                 # GPU-enabled build
│   └── docker-compose.yml         # Full stack
├── tests/                         # Pytest suite
├── pyproject.toml                 # Project config
└── .env.example                   # Environment template
```

## Supported Models

- Llama 3 / 3.1 (8B, 70B)
- Mistral / Mixtral
- Qwen 2 / 2.5
- Phi-3
- Gemma 2
- Any Hugging Face causal LM

## Quantization Methods

| Method | Speed | Quality | GPU Memory | Use Case |
|--------|-------|---------|------------|----------|
| AWQ | ⚡⚡⚡ | ★★★★☆ | Low | Production serving |
| GPTQ | ⚡⚡ | ★★★★☆ | Low | Quality-critical |
| GGUF | ⚡ | ★★★☆☆ | Very Low | CPU/Edge |
| FP16/BF16 | ⚡⚡ | ★★★★★ | Medium | Baseline |

## Benchmarking Metrics

- **Throughput** — Tokens per second (total and per-request)
- **Latency** — p50, p95, p99 in milliseconds
- **TTFT** — Time to first token
- **Memory** — GPU memory allocated/reserved
- **Speculative Stats** — Acceptance rate, speedup factor

## License

MIT License

## Author

Yogesh Gaikwad — Yogesh.r.gaikwad@gmail.com