"""CLI for Inference Optimization."""

import asyncio
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt

from app.core.config import settings
from app.services.inference_service import inference_service
from app.services.quantization_service import quantization_service
from app.models.schemas import QuantizationMethod, GenerationRequest, SpeculativeConfig

app = typer.Typer(help="Inference Optimization CLI")
console = Console()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind"),
    port: int = typer.Option(8003, "--port", "-p", help="Port to bind"),
):
    """Start the API server."""
    import uvicorn
    uvicorn.run("app.main:app", host=host, port=port, reload=False)


@app.command()
def quantize(
    model_name: str = typer.Argument(..., help="Model name to quantize"),
    method: QuantizationMethod = typer.Option(QuantizationMethod.AWQ, "--method", "-m", help="Quantization method"),
    bits: int = typer.Option(4, "--bits", "-b", help="Quantization bits (2-8)"),
    group_size: int = typer.Option(128, "--group-size", "-g", help="Group size"),
    calibration: int = typer.Option(512, "--calibration", "-c", help="Calibration samples"),
    output_dir: str | None = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Quantize a model."""
    asyncio.run(_quantize_async(model_name, method, bits, group_size, calibration, output_dir))


async def _quantize_async(
    model_name: str,
    method: QuantizationMethod,
    bits: int,
    group_size: int,
    calibration: int,
    output_dir: str | None,
):
    console.print(Panel.fit(f"[bold blue]Quantizing {model_name}[/bold blue]"))
    console.print(f"Method: [cyan]{method.value}[/cyan]")
    console.print(f"Bits: [cyan]{bits}[/cyan]")
    console.print(f"Group Size: [cyan]{group_size}[/cyan]")
    console.print()

    from app.models.schemas import QuantizationRequest

    request = QuantizationRequest(
        model_name=model_name,
        method=method,
        bits=bits,
        group_size=group_size,
        calibration_samples=calibration,
        output_dir=output_dir,
    )

    with console.status("[bold yellow]Loading model...[/bold yellow]", spinner="dots"):
        quantization_service.load_model(model_name)

    with console.status("[bold yellow]Quantizing...[/bold yellow]", spinner="dots"):
        result = quantization_service.quantize(request)

    console.print(f"[green]Quantization complete![/green]")
    console.print(f"Output: [cyan]{result.output_path}[/cyan]")
    console.print(f"Original: [cyan]{result.original_size_mb:.1f} MB[/cyan]")
    console.print(f"Quantized: [cyan]{result.quantized_size_mb:.1f} MB[/cyan]")
    console.print(f"Compression: [cyan]{result.compression_ratio:.2f}x[/cyan]")
    console.print(f"Time: [cyan]{result.quantization_time_seconds:.1f}s[/cyan]")


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Prompt for generation"),
    max_tokens: int = typer.Option(256, "--max-tokens", "-t", help="Max tokens to generate"),
    temperature: float = typer.Option(0.7, "--temperature", help="Sampling temperature"),
    top_p: float = typer.Option(0.9, "--top-p", help="Top-p sampling"),
    speculative: bool = typer.Option(True, "--speculative/--no-speculative", help="Use speculative decoding"),
):
    """Generate text."""
    asyncio.run(_generate_async(prompt, max_tokens, temperature, top_p, speculative))


async def _generate_async(prompt: str, max_tokens: int, temperature: float, top_p: float, use_speculative: bool):
    console.print(Panel.fit(f"[bold blue]Generating text[/bold blue]"))
    console.print(f"Prompt: [cyan]{prompt[:80]}{'...' if len(prompt) > 80 else ''}[/cyan]")
    console.print()

    await inference_service.initialize()

    speculative_config = SpeculativeConfig(enabled=use_speculative) if use_speculative else None

    request = GenerationRequest(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        speculative=speculative_config,
    )

    with console.status("[bold yellow]Generating...[/bold yellow]", spinner="dots"):
        response = inference_service.generate(request)

    console.print(f"[green]Generated:[/green] {response.text}")
    console.print(f"Tokens: [cyan]{response.tokens_generated}[/cyan]")
    console.print(f"Time: [cyan]{response.generation_time_seconds:.2f}s[/cyan]")
    console.print(f"Speed: [cyan]{response.tokens_per_second:.1f} tok/s[/cyan]")

    if response.speculative_stats and response.speculative_stats.get("enabled"):
        stats = response.speculative_stats
        console.print(f"Speculative: [cyan]{stats['acceptance_rate']:.1%} acceptance[/cyan]")


@app.command()
def benchmark(
    model_name: str = typer.Argument(..., help="Model name to benchmark"),
    preset: str = typer.Option("standard", "--preset", "-p", help="Benchmark preset: quick, standard, comprehensive"),
):
    """Run inference benchmark."""
    asyncio.run(_benchmark_async(model_name, preset))


async def _benchmark_async(model_name: str, preset: str):
    console.print(Panel.fit(f"[bold blue]Benchmarking {model_name}[/bold blue]"))
    console.print(f"Preset: [cyan]{preset}[/cyan]")
    console.print()

    await inference_service.initialize()

    presets = {
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

    if preset not in presets:
        console.print(f"[red]Unknown preset: {preset}[/red]")
        return

    config = presets[preset]

    from app.models.schemas import BenchmarkRequest

    request = BenchmarkRequest(
        model_name=model_name,
        **config,
    )

    with console.status("[bold yellow]Running benchmark...[/bold yellow]", spinner="dots"):
        result = inference_service.run_benchmark(request)

    _print_benchmark_results(result)


def _print_benchmark_results(result):
    console.print(f"[green]Benchmark complete![/green]")
    console.print(f"Model: [cyan]{result.model_name}[/cyan]")
    console.print(f"Device: [cyan]{result.device}[/cyan]")
    console.print(f"Dtype: [cyan]{result.dtype}[/cyan]")
    console.print()

    # Summary table
    table = Table(title="Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in result.summary.items():
        table.add_row(key, f"{value:.2f}" if isinstance(value, float) else str(value))

    console.print(table)

    # Detailed results
    detail_table = Table(title="Detailed Results")
    detail_table.add_column("Prompt", style="cyan")
    detail_table.add_column("Gen", style="cyan")
    detail_table.add_column("Batch", style="cyan")
    detail_table.add_column("Latency (ms)", style="green")
    detail_table.add_column("First Token (ms)", style="green")
    detail_table.add_column("Throughput (tok/s)", style="green")

    for r in result.results:
        detail_table.add_row(
            str(r["prompt_length"]),
            str(r["generation_length"]),
            str(r["batch_size"]),
            f"{r['avg_latency_ms']:.1f}",
            f"{r['avg_first_token_ms']:.1f}",
            f"{r['tokens_per_second']:.1f}",
        )

    console.print(detail_table)


@app.command()
def health():
    """Check service health."""
    asyncio.run(_health_async())


async def _health_async():
    console.print(Panel.fit(f"[bold blue]{settings.app_name}[/bold blue] - Health Check"))
    health = await inference_service.get_health()

    table = Table(title="Health Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")

    table.add_row("Overall", "✅ Healthy" if health.status == "healthy" else "❌ " + health.status, "")
    table.add_row("Model Loaded", "✅" if health.model_loaded else "❌", health.model_name or "N/A")
    table.add_row("Device", "", health.device)
    if health.gpu_memory_allocated_mb:
        table.add_row("GPU Allocated", "", f"{health.gpu_memory_allocated_mb:.1f} MB")
    if health.gpu_memory_reserved_mb:
        table.add_row("GPU Reserved", "", f"{health.gpu_memory_reserved_mb:.1f} MB")

    console.print(table)

    if health.kv_cache_stats:
        kv = health.kv_cache_stats
        console.print(f"\nKV Cache: {kv.cache_size_mb:.1f}/{kv.cache_size_mb + kv.cache_size_mb * 0.1:.1f} MB, Hit Rate: {kv.hit_rate:.1%}")


@app.command()
def kv_cache():
    """Show KV cache statistics."""
    asyncio.run(_kv_cache_async())


async def _kv_cache_async():
    console.print(Panel.fit(f"[bold blue]{settings.app_name}[/bold blue] - KV Cache Stats"))
    await inference_service.initialize()

    stats = inference_service.get_kv_cache_stats()

    table = Table(title="KV Cache Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Tokens", str(stats.total_tokens))
    table.add_row("Cache Size", f"{stats.cache_size_mb:.1f} MB")
    table.add_row("Hit Rate", f"{stats.hit_rate:.1%}")
    table.add_row("Evictions", str(stats.eviction_count))
    table.add_row("Offloaded", str(stats.offloaded_blocks))

    console.print(table)


if __name__ == "__main__":
    app()