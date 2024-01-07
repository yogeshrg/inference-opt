"""Tests for Inference Optimization."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from app.core.config import settings
from app.models.schemas import (
    QuantizationMethod,
    QuantizationRequest,
    QuantizationResult,
    SpeculativeConfig,
    GenerationRequest,
    GenerationResponse,
    BenchmarkRequest,
    BenchmarkResult,
    HealthResponse,
    ModelInfo,
    KVCacheStats,
)
from app.services.quantization_service import QuantizationService
from app.services.speculative_service import SpeculativeDecodingService
from app.services.kv_cache_service import KVCacheManager
from app.services.benchmark_service import BenchmarkService


class TestConfig:
    """Test configuration loading."""

    def test_settings_defaults(self):
        """Test default settings values."""
        assert settings.app_name == "Inference Optimization"
        assert settings.quant_method == "awq"
        assert settings.quant_bits == 4
        assert settings.speculative_enabled is True
        assert settings.continuous_batching is True


class TestSchemas:
    """Test Pydantic schemas."""

    def test_quantization_request(self):
        """Test quantization request model."""
        req = QuantizationRequest(
            model_name="test-model",
            method=QuantizationMethod.AWQ,
            bits=4,
        )
        assert req.model_name == "test-model"
        assert req.method == QuantizationMethod.AWQ
        assert req.bits == 4

    def test_quantization_result(self):
        """Test quantization result model."""
        result = QuantizationResult(
            model_name="test-model",
            method=QuantizationMethod.AWQ,
            bits=4,
            original_size_mb=1000.0,
            quantized_size_mb=250.0,
            compression_ratio=4.0,
            calibration_time_seconds=10.0,
            quantization_time_seconds=60.0,
            output_path="/path/to/output",
        )
        assert result.compression_ratio == 4.0
        assert result.original_size_mb == 1000.0

    def test_speculative_config(self):
        """Test speculative config model."""
        config = SpeculativeConfig(enabled=True, acceptance_threshold=0.9)
        assert config.enabled is True
        assert config.acceptance_threshold == 0.9

    def test_generation_request(self):
        """Test generation request model."""
        req = GenerationRequest(
            prompt="Hello world",
            max_tokens=128,
            temperature=0.7,
        )
        assert req.prompt == "Hello world"
        assert req.max_tokens == 128

    def test_generation_response(self):
        """Test generation response model."""
        response = GenerationResponse(
            text="Generated text",
            tokens_generated=50,
            prompt_tokens=10,
            generation_time_seconds=2.0,
            tokens_per_second=25.0,
        )
        assert response.tokens_per_second == 25.0

    def test_benchmark_result(self):
        """Test benchmark result model."""
        result = BenchmarkResult(
            model_name="test-model",
            device="cuda",
            dtype="float16",
            results=[],
            summary={"avg_throughput_tps": 50.0},
        )
        assert result.model_name == "test-model"

    def test_health_response(self):
        """Test health response model."""
        health = HealthResponse(
            status="healthy",
            version="0.1.0",
            model_loaded=True,
            model_name="test-model",
            device="cuda",
        )
        assert health.status == "healthy"
        assert health.model_loaded is True

    def test_model_info(self):
        """Test model info model."""
        info = ModelInfo(
            name="test-model",
            path="/path/to/model",
            size_mb=1000.0,
            parameters=7000000000,
            quantized=False,
        )
        assert info.parameters == 7000000000


class TestQuantizationService:
    """Test quantization service."""

    @pytest.fixture
    def quant_service(self):
        """Create quantization service."""
        return QuantizationService()

    def test_init(self, quant_service):
        """Test initialization."""
        assert quant_service._model is None
        assert quant_service._tokenizer is None
        assert quant_service._current_model_info is None

    def test_get_model_info(self, quant_service):
        """Test getting model info."""
        assert quant_service.get_model_info() is None


class TestSpeculativeService:
    """Test speculative decoding service."""

    @pytest.fixture
    def spec_service(self):
        """Create speculative service."""
        return SpeculativeDecodingService()

    def test_init(self, spec_service):
        """Test initialization."""
        assert spec_service._target_model is None
        assert spec_service._draft_model is None
        assert spec_service._loaded is False


class TestKVCacheManager:
    """Test KV cache manager."""

    @pytest.fixture
    def kv_cache(self):
        """Create KV cache manager."""
        return KVCacheManager()

    def test_init(self, kv_cache):
        """Test initialization."""
        assert kv_cache._cache is not None
        assert kv_cache._max_size == settings.max_kv_cache_size

    def test_put_and_get(self, kv_cache):
        """Test putting and getting cache entries."""
        import torch

        key_tensor = torch.randn(1, 10, 64, dtype=torch.float16)
        value_tensor = torch.randn(1, 10, 64, dtype=torch.float16)

        kv_cache.put("test_key", key_tensor, value_tensor)
        result = kv_cache.get("test_key")

        assert result is not None
        assert result[0].shape == key_tensor.shape
        assert result[1].shape == value_tensor.shape

    def test_get_miss(self, kv_cache):
        """Test cache miss."""
        result = kv_cache.get("nonexistent")
        assert result is None

    def test_eviction(self, kv_cache):
        """Test LRU eviction."""
        import torch

        # Set small max size
        kv_cache._max_size = 1000  # Very small

        key_tensor = torch.randn(1, 100, 64, dtype=torch.float16)
        value_tensor = torch.randn(1, 100, 64, dtype=torch.float16)

        # Add first entry
        kv_cache.put("key1", key_tensor, value_tensor)
        assert len(kv_cache._cache) == 1

        # Add second entry (should evict first)
        kv_cache.put("key2", key_tensor, value_tensor)
        assert len(kv_cache._cache) == 1
        assert "key2" in kv_cache._cache

    def test_get_stats(self, kv_cache):
        """Test getting statistics."""
        import torch

        kv_cache.put("test", torch.randn(1, 10, 64), torch.randn(1, 10, 64))
        kv_cache.get("test")  # Hit
        kv_cache.get("missing")  # Miss

        stats = kv_cache.get_stats()

        assert stats["total_entries"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_clear(self, kv_cache):
        """Test clearing cache."""
        import torch

        kv_cache.put("test", torch.randn(1, 10, 64), torch.randn(1, 10, 64))
        kv_cache.clear()

        assert len(kv_cache._cache) == 0
        assert kv_cache.get("test") is None


class TestBenchmarkService:
    """Test benchmark service."""

    @pytest.fixture
    def bench_service(self):
        """Create benchmark service."""
        return BenchmarkService()

    def test_init(self, bench_service):
        """Test initialization."""
        assert bench_service._model is None
        assert bench_service._tokenizer is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])