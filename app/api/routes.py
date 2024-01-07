"""API routes for Inference Optimization."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Any

from app.models.schemas import (
    QuantizationRequest,
    QuantizationResult,
    GenerationRequest,
    GenerationResponse,
    BenchmarkRequest,
    BenchmarkResult,
    HealthResponse,
    ModelInfo,
    KVCacheStats,
)
from app.services.inference_service import inference_service

router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return await inference_service.get_health()


@router.get("/model/info", response_model=ModelInfo | None)
async def get_model_info() -> ModelInfo | None:
    """Get current model information."""
    return inference_service.get_model_info()


# ========== Quantization ==========

@router.post("/quantize", response_model=QuantizationResult)
async def quantize_model(request: QuantizationRequest) -> QuantizationResult:
    """Quantize a model."""
    if not inference_service.is_ready():
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        return inference_service.quantize_model(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quantize/load", response_model=ModelInfo)
async def load_model_for_quantization(model_name: str) -> ModelInfo:
    """Load a model for quantization."""
    try:
        return inference_service.load_model_for_quantization(model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== Generation ==========

@router.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest) -> GenerationResponse:
    """Generate text with optional speculative decoding."""
    if not inference_service.is_ready():
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        return inference_service.generate(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== KV Cache ==========

@router.get("/kv-cache/stats", response_model=KVCacheStats)
async def get_kv_cache_stats() -> KVCacheStats:
    """Get KV cache statistics."""
    return inference_service.get_kv_cache_stats()


@router.post("/kv-cache/clear")
async def clear_kv_cache() -> dict[str, str]:
    """Clear KV cache."""
    return inference_service.clear_kv_cache()


# ========== Benchmarking ==========

@router.post("/benchmark", response_model=BenchmarkResult)
async def run_benchmark(request: BenchmarkRequest) -> BenchmarkResult:
    """Run inference benchmark."""
    if not inference_service.is_ready():
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        return inference_service.run_benchmark(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/benchmark/presets")
async def get_benchmark_presets() -> dict[str, Any]:
    """Get predefined benchmark configurations."""
    return {
        "quick": {
            "prompt_lengths": [128],
            "gen_lengths": [128],
            "batch_sizes": [1, 4],
            "warmup_runs": 2,
            "benchmark_runs": 5,
        },
        "standard": {
            "prompt_lengths": [128, 512, 1024],
            "gen_lengths": [128, 256, 512],
            "batch_sizes": [1, 4, 8],
            "warmup_runs": 3,
            "benchmark_runs": 10,
        },
        "comprehensive": {
            "prompt_lengths": [128, 512, 1024, 2048],
            "gen_lengths": [128, 256, 512, 1024],
            "batch_sizes": [1, 2, 4, 8, 16, 32],
            "warmup_runs": 5,
            "benchmark_runs": 20,
        },
    }