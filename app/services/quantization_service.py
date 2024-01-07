"""Quantization service for model compression."""

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.core.config import settings
from app.models.schemas import (
    QuantizationMethod,
    QuantizationRequest,
    QuantizationResult,
    ModelInfo,
)

logger = logging.getLogger(__name__)


class QuantizationService:
    """Service for model quantization."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._current_model_info: ModelInfo | None = None

    def load_model(self, model_name: str, model_path: Path | None = None) -> ModelInfo:
        """Load a model for quantization."""
        logger.info(f"Loading model: {model_name}")

        path = model_path or settings.model_cache_dir / model_name.replace("/", "_")
        path.mkdir(parents=True, exist_ok=True)

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=settings.model_cache_dir,
            trust_remote_code=True,
        )

        # Load model
        device_map = "auto" if settings.device == "auto" else settings.device
        torch_dtype = getattr(torch, settings.dtype)

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=settings.model_cache_dir,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # Get model info
        num_params = sum(p.numel() for p in self._model.parameters())
        size_mb = sum(p.numel() * p.element_size() for p in self._model.parameters()) / (1024 * 1024)

        self._current_model_info = ModelInfo(
            name=model_name,
            path=str(path),
            size_mb=size_mb,
            parameters=num_params,
            quantized=False,
        )

        logger.info(f"Model loaded: {num_params:,} parameters, {size_mb:.1f} MB")
        return self._current_model_info

    def quantize(self, request: QuantizationRequest) -> QuantizationResult:
        """Quantize the loaded model."""
        if self._model is None:
            self.load_model(request.model_name)

        if request.method == QuantizationMethod.NONE:
            return self._save_original(request)

        logger.info(f"Starting {request.method.value} quantization...")

        if request.method == QuantizationMethod.AWQ:
            return self._quantize_awq(request)
        elif request.method == QuantizationMethod.GPTQ:
            return self._quantize_gptq(request)
        elif request.method == QuantizationMethod.GGUF:
            return self._quantize_gguf(request)
        else:
            raise ValueError(f"Unsupported quantization method: {request.method}")

    def _quantize_awq(self, request: QuantizationRequest) -> QuantizationResult:
        """Quantize using AWQ (Activation-aware Weight Quantization)."""
        try:
            from awq import AutoAWQForCausalLM
        except ImportError:
            raise RuntimeError("autoawq not installed. Install with: pip install autoawq")

        start_time = time.time()
        output_dir = request.output_dir or str(settings.quant_output_dir / f"{request.model_name.replace('/', '_')}-awq-{request.bits}bit")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Get calibration data
        calibration_data = self._get_calibration_data(request.calibration_samples)

        # Quantize
        quant_start = time.time()
        model = AutoAWQForCausalLM.from_pretrained(
            request.model_name,
            device_map="auto" if settings.device == "auto" else settings.device,
        )
        model.quantize(
            self._tokenizer,
            quant_config={
                "zero_point": True,
                "q_group_size": request.group_size,
                "w_bit": request.bits,
                "version": "GEMM",
            },
            calib_data=calibration_data,
        )

        # Save
        model.save_quantized(output_dir)
        self._tokenizer.save_pretrained(output_dir)

        quant_time = time.time() - quant_start
        total_time = time.time() - start_time

        # Calculate sizes
        original_size = self._current_model_info.size_mb if self._current_model_info else 0
        quantized_size = self._get_dir_size(output_dir)

        return QuantizationResult(
            model_name=request.model_name,
            method=QuantizationMethod.AWQ,
            bits=request.bits,
            original_size_mb=original_size,
            quantized_size_mb=quantized_size,
            compression_ratio=original_size / quantized_size if quantized_size > 0 else 0,
            calibration_time_seconds=quant_time,
            quantization_time_seconds=total_time,
            output_path=output_dir,
        )

    def _quantize_gptq(self, request: QuantizationRequest) -> QuantizationResult:
        """Quantize using GPTQ."""
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        except ImportError:
            raise RuntimeError("auto-gptq not installed. Install with: pip install auto-gptq")

        start_time = time.time()
        output_dir = request.output_dir or str(settings.quant_output_dir / f"{request.model_name.replace('/', '_')}-gptq-{request.bits}bit")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        calibration_data = self._get_calibration_data(request.calibration_samples)

        quantize_config = BaseQuantizeConfig(
            bits=request.bits,
            group_size=request.group_size,
            desc_act=False,
        )

        quant_start = time.time()
        model = AutoGPTQForCausalLM.from_pretrained(
            request.model_name,
            quantize_config=quantize_config,
            device_map="auto",
        )

        model.quantize(calibration_data)
        model.save_quantized(output_dir)
        self._tokenizer.save_pretrained(output_dir)

        quant_time = time.time() - quant_start
        total_time = time.time() - start_time

        original_size = self._current_model_info.size_mb if self._current_model_info else 0
        quantized_size = self._get_dir_size(output_dir)

        return QuantizationResult(
            model_name=request.model_name,
            method=QuantizationMethod.GPTQ,
            bits=request.bits,
            original_size_mb=original_size,
            quantized_size_mb=quantized_size,
            compression_ratio=original_size / quantized_size if quantized_size > 0 else 0,
            calibration_time_seconds=quant_time,
            quantization_time_seconds=total_time,
            output_path=output_dir,
        )

    def _quantize_gguf(self, request: QuantizationRequest) -> QuantizationResult:
        """Quantize to GGUF format."""
        try:
            import llama_cpp
        except ImportError:
            raise RuntimeError("llama-cpp-python not installed. Install with: pip install llama-cpp-python")

        start_time = time.time()
        output_dir = request.output_dir or str(settings.quant_output_dir / f"{request.model_name.replace('/', '_')}-gguf-{request.bits}bit")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # llama.cpp quantization would go here
        # For now, return placeholder
        total_time = time.time() - start_time

        return QuantizationResult(
            model_name=request.model_name,
            method=QuantizationMethod.GGUF,
            bits=request.bits,
            original_size_mb=self._current_model_info.size_mb if self._current_model_info else 0,
            quantized_size_mb=0,
            compression_ratio=0,
            calibration_time_seconds=0,
            quantization_time_seconds=total_time,
            output_path=output_dir,
        )

    def _save_original(self, request: QuantizationRequest) -> QuantizationResult:
        """Save model without quantization."""
        output_dir = request.output_dir or str(settings.model_cache_dir / request.model_name.replace("/", "_"))
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(output_dir)
        self._tokenizer.save_pretrained(output_dir)

        return QuantizationResult(
            model_name=request.model_name,
            method=QuantizationMethod.NONE,
            bits=16,
            original_size_mb=self._current_model_info.size_mb if self._current_model_info else 0,
            quantized_size_mb=self._current_model_info.size_mb if self._current_model_info else 0,
            compression_ratio=1.0,
            calibration_time_seconds=0,
            quantization_time_seconds=0,
            output_path=output_dir,
        )

    def _get_calibration_data(self, num_samples: int) -> list[str]:
        """Get calibration data for quantization."""
        # In production, use a proper calibration dataset like WikiText or C4
        return [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "Machine learning models require large datasets.",
            "Natural language processing enables computers to understand text.",
            "Deep learning uses neural networks with many layers.",
        ] * (num_samples // 5 + 1)

    def _get_dir_size(self, path: str) -> float:
        """Get directory size in MB."""
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
        return total / (1024 * 1024)

    def get_model_info(self) -> ModelInfo | None:
        """Get current model info."""
        return self._current_model_info


# Global instance
quantization_service = QuantizationService()