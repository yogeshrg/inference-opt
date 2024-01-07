"""Inference Optimization configuration."""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "Inference Optimization"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # Server
    host: str = "0.0.0.0"
    port: int = 8003

    # Model
    model_name: str = "meta-llama/Meta-Llama-3.1-8B"
    model_path: Path | None = None
    device: str = "auto"  # auto, cuda, cpu, mps
    dtype: str = "float16"  # float16, bfloat16, float32

    # Quantization
    quant_method: str = "awq"  # awq, gptq, gguf, none
    quant_bits: int = 4
    quant_group_size: int = 128
    quant_calibration_samples: int = 512

    # Speculative Decoding
    speculative_enabled: bool = True
    draft_model_name: str = "meta-llama/Meta-Llama-3.1-8B"
    draft_model_path: Path | None = None
    acceptance_threshold: float = 0.9
    max_draft_tokens: int = 5

    # KV Cache
    kv_cache_enabled: bool = True
    kv_cache_dtype: str = "float16"
    max_kv_cache_size: int = 1024 * 1024 * 1024  # 1GB
    kv_cache_offload: bool = False

    # Batching
    continuous_batching: bool = True
    max_batch_size: int = 32
    max_waiting_tokens: int = 20

    # Benchmark
    benchmark_warmup: int = 3
    benchmark_runs: int = 10
    benchmark_prompt_lengths: list[int] = Field(default=[128, 512, 1024, 2048])
    benchmark_gen_lengths: list[int] = Field(default=[128, 256, 512])

    # Model Storage
    model_cache_dir: Path = Field(default=Path("./models"))
    quant_output_dir: Path = Field(default=Path("./models/quantized"))


settings = Settings()