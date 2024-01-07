"""Main inference service coordinating all optimization features."""

import logging
from typing import Any

from app.core.config import settings
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
    SpeculativeConfig,
)
from app.services.quantization_service import quantization_service
from app.services.speculative_service import speculative_service
from app.services.kv_cache_service import kv_cache_manager
from app.services.benchmark_service import benchmark_service

logger = logging.getLogger(__name__)


class InferenceService:
    """Unified inference service with optimization features."""

    def __init__(self) -> None:
        self._model_loaded = False
        self._current_model: str | None = None

    async def initialize(self, model_name: str | None = None) -> None:
        """Initialize the inference service."""
        model = model_name or settings.model_name
        logger.info(f"Initializing inference service with model: {model}")

        # Load models for speculative decoding
        speculative_service.load_models(
            target_model=model,
            draft_model=settings.draft_model_name if settings.speculative_enabled else None,
        )

        self._model_loaded = True
        self._current_model = model
        logger.info("Inference service initialized")

    def is_ready(self) -> bool:
        """Check if service is ready."""
        return self._model_loaded

    # ========== Quantization ==========

    def quantize_model(self, request: QuantizationRequest) -> QuantizationResult:
        """Quantize a model."""
        return quantization_service.quantize(request)

    def load_model_for_quantization(self, model_name: str) -> ModelInfo:
        """Load a model for quantization."""
        return quantization_service.load_model(model_name)

    # ========== Generation ==========

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text with optional speculative decoding."""
        if not self._model_loaded:
            raise RuntimeError("Service not initialized. Call initialize() first.")

        # Override speculative config if provided
        if request.speculative:
            # The speculative service uses the config from the request
            pass

        return speculative_service.generate(request)

    # ========== KV Cache ==========

    def get_kv_cache_stats(self) -> KVCacheStats:
        """Get KV cache statistics."""
        stats = kv_cache_manager.get_stats()
        return KVCacheStats(
            total_tokens=stats.get("total_entries", 0) * 2,  # Approximate
            cache_size_mb=stats.get("cache_size_mb", 0.0),
            hit_rate=stats.get("hit_rate", 0.0),
            eviction_count=stats.get("evictions", 0),
            offloaded_blocks=stats.get("offloaded", 0),
        )

    def clear_kv_cache(self) -> dict[str, str]:
        """Clear KV cache."""
        kv_cache_manager.clear()
        return {"status": "cleared"}

    # ========== Benchmarking ==========

    def run_benchmark(self, request: BenchmarkRequest) -> BenchmarkResult:
        """Run inference benchmark."""
        return benchmark_service.run_benchmark(request)

    # ========== Model Info ==========

    def get_model_info(self) -> ModelInfo | None:
        """Get current model information."""
        info = quantization_service.get_model_info()
        if info and self._current_model:
            info.name = self._current_model
        return info

    # ========== Health ==========

    async def get_health(self) -> HealthResponse:
        """Get health status."""
        import torch

        gpu_allocated = None
        gpu_reserved = None
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)

        return HealthResponse(
            status="healthy" if self._model_loaded else "initializing",
            version=settings.app_version,
            model_loaded=self._model_loaded,
            model_name=self._current_model,
            device=settings.device,
            gpu_memory_allocated_mb=gpu_allocated,
            gpu_memory_reserved_mb=gpu_reserved,
            kv_cache_stats=self.get_kv_cache_stats(),
        )


# Global instance
inference_service = InferenceService()