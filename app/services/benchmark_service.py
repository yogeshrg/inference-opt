"""Benchmark service for inference performance measurement."""

import logging
import statistics
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.core.config import settings
from app.models.schemas import BenchmarkRequest, BenchmarkResult

logger = logging.getLogger(__name__)


class BenchmarkService:
    """Service for running inference benchmarks."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    def load_model(self, model_name: str) -> None:
        """Load model for benchmarking."""
        logger.info(f"Loading model for benchmark: {model_name}")

        device_map = "auto" if settings.device == "auto" else settings.device
        torch_dtype = getattr(torch, settings.dtype)

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=settings.model_cache_dir,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=settings.model_cache_dir,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        self._model.eval()
        logger.info("Model loaded for benchmarking")

    def run_benchmark(self, request: BenchmarkRequest) -> BenchmarkResult:
        """Run comprehensive benchmark."""
        if self._model is None:
            self.load_model(request.model_name)

        device = str(self._model.device)
        dtype = str(next(self._model.parameters()).dtype).replace("torch.", "")

        all_results = []

        for prompt_len in request.prompt_lengths:
            for gen_len in request.gen_lengths:
                for batch_size in request.batch_sizes:
                    result = self._benchmark_config(
                        prompt_len=prompt_len,
                        gen_len=gen_len,
                        batch_size=batch_size,
                        warmup_runs=request.warmup_runs,
                        benchmark_runs=request.benchmark_runs,
                    )
                    all_results.append(result)

        # Calculate summary statistics
        summary = self._calculate_summary(all_results)

        return BenchmarkResult(
            model_name=request.model_name,
            device=device,
            dtype=dtype,
            results=all_results,
            summary=summary,
        )

    def _benchmark_config(
        self,
        prompt_len: int,
        gen_len: int,
        batch_size: int,
        warmup_runs: int,
        benchmark_runs: int,
    ) -> dict[str, Any]:
        """Run benchmark for a specific configuration."""
        logger.info(f"Benchmarking: prompt={prompt_len}, gen={gen_len}, batch={batch_size}")

        # Generate test prompts
        prompts = self._generate_prompts(prompt_len, batch_size)

        # Warmup
        for _ in range(warmup_runs):
            self._run_generation(prompts, gen_len)

        # Benchmark runs
        latencies = []
        token_counts = []
        first_token_times = []

        for _ in range(benchmark_runs):
            # Generate with timing
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()

            # Time to first token
            first_token_start = time.perf_counter()
            _ = self._model.generate(
                **prompts,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            first_token_time = time.perf_counter() - first_token_start
            first_token_times.append(first_token_time)

            # Full generation
            gen_start = time.perf_counter()
            outputs = self._model.generate(
                **prompts,
                max_new_tokens=gen_len,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            gen_time = time.perf_counter() - gen_start
            latencies.append(gen_time)

            # Count generated tokens
            tokens_generated = outputs.shape[1] - prompts["input_ids"].shape[1]
            token_counts.append(tokens_generated)

        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
        avg_first_token = statistics.mean(first_token_times)
        avg_tokens = statistics.mean(token_counts)
        throughput = avg_tokens / avg_latency if avg_latency > 0 else 0

        return {
            "prompt_length": prompt_len,
            "generation_length": gen_len,
            "batch_size": batch_size,
            "avg_latency_ms": avg_latency * 1000,
            "std_latency_ms": std_latency * 1000,
            "avg_first_token_ms": avg_first_token * 1000,
            "avg_tokens_per_request": avg_tokens,
            "tokens_per_second": throughput,
            "latency_p50_ms": sorted(latencies)[len(latencies) // 2] * 1000,
            "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] * 1000,
            "latency_p99_ms": sorted(latencies)[int(len(latencies) * 0.99)] * 1000,
        }

    def _generate_prompts(self, prompt_len: int, batch_size: int) -> dict[str, torch.Tensor]:
        """Generate test prompts of specified length."""
        # Create a prompt that tokenizes to approximately prompt_len
        base_text = "The quick brown fox jumps over the lazy dog. " * 10
        tokens = self._tokenizer.encode(base_text, add_special_tokens=False)

        # Truncate or repeat to get desired length
        if len(tokens) < prompt_len:
            repeat = (prompt_len // len(tokens)) + 1
            tokens = tokens * repeat
        tokens = tokens[:prompt_len]

        # Create batch
        input_ids = torch.tensor([tokens] * batch_size, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids.to(self._model.device),
            "attention_mask": attention_mask.to(self._model.device),
        }

    def _run_generation(self, prompts: dict, gen_len: int) -> torch.Tensor:
        """Run a single generation."""
        with torch.no_grad():
            return self._model.generate(
                **prompts,
                max_new_tokens=gen_len,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

    def _calculate_summary(self, results: list[dict]) -> dict[str, float]:
        """Calculate summary statistics across all configurations."""
        if not results:
            return {}

        throughputs = [r["tokens_per_second"] for r in results]
        latencies = [r["avg_latency_ms"] for r in results]
        first_tokens = [r["avg_first_token_ms"] for r in results]

        return {
            "max_throughput_tps": max(throughputs),
            "min_throughput_tps": min(throughputs),
            "avg_throughput_tps": statistics.mean(throughputs),
            "max_latency_ms": max(latencies),
            "min_latency_ms": min(latencies),
            "avg_latency_ms": statistics.mean(latencies),
            "avg_first_token_ms": statistics.mean(first_tokens),
        }


# Global instance
benchmark_service = BenchmarkService()