"""Speculative decoding service for faster inference."""

import logging
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.core.config import settings
from app.models.schemas import SpeculativeConfig, GenerationRequest, GenerationResponse

logger = logging.getLogger(__name__)


class SpeculativeDecodingService:
    """Service for speculative decoding with draft model."""

    def __init__(self) -> None:
        self._target_model = None
        self._target_tokenizer = None
        self._draft_model = None
        self._draft_tokenizer = None
        self._loaded = False

    def load_models(self, target_model: str, draft_model: str | None = None) -> None:
        """Load target and draft models."""
        logger.info(f"Loading target model: {target_model}")

        device_map = "auto" if settings.device == "auto" else settings.device
        torch_dtype = getattr(torch, settings.dtype)

        # Load target model
        self._target_tokenizer = AutoTokenizer.from_pretrained(
            target_model,
            cache_dir=settings.model_cache_dir,
            trust_remote_code=True,
        )
        self._target_model = AutoModelForCausalLM.from_pretrained(
            target_model,
            cache_dir=settings.model_cache_dir,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # Load draft model (can be same as target for testing)
        draft_name = draft_model or target_model
        if draft_name != target_model:
            logger.info(f"Loading draft model: {draft_name}")
            self._draft_tokenizer = AutoTokenizer.from_pretrained(
                draft_name,
                cache_dir=settings.model_cache_dir,
                trust_remote_code=True,
            )
            self._draft_model = AutoModelForCausalLM.from_pretrained(
                draft_name,
                cache_dir=settings.model_cache_dir,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        else:
            # Use target model as draft (for testing)
            self._draft_model = self._target_model
            self._draft_tokenizer = self._target_tokenizer

        self._loaded = True
        logger.info("Models loaded for speculative decoding")

    @torch.inference_mode()
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text with speculative decoding."""
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        start_time = time.time()

        # Tokenize input
        input_ids = self._target_tokenizer.encode(request.prompt, return_tensors="pt").to(self._target_model.device)
        prompt_tokens = input_ids.shape[1]

        # Generate with speculative decoding
        if request.speculative and request.speculative.enabled:
            output_ids, speculative_stats = self._speculative_generate(
                input_ids,
                request.max_tokens,
                request.speculative,
            )
        else:
            output_ids = self._target_model.generate(
                input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                do_sample=request.temperature > 0,
                pad_token_id=self._target_tokenizer.eos_token_id,
            )
            speculative_stats = {"enabled": False}

        generation_time = time.time() - start_time

        # Decode output (only new tokens)
        new_tokens = output_ids[0][prompt_tokens:]
        generated_text = self._target_tokenizer.decode(new_tokens, skip_special_tokens=True)
        tokens_generated = len(new_tokens)

        return GenerationResponse(
            text=generated_text,
            tokens_generated=tokens_generated,
            prompt_tokens=prompt_tokens,
            generation_time_seconds=generation_time,
            tokens_per_second=tokens_generated / generation_time if generation_time > 0 else 0,
            speculative_stats=speculative_stats,
        )

    def _speculative_generate(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        config: SpeculativeConfig,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Generate using speculative decoding."""
        device = self._target_model.device
        acceptance_threshold = config.acceptance_threshold
        max_draft_tokens = config.max_draft_tokens

        current_ids = input_ids.clone()
        total_drafted = 0
        total_accepted = 0
        total_rejected = 0

        for step in range(max_tokens):
            # Draft phase: generate multiple tokens with draft model
            draft_ids = self._draft_model.generate(
                current_ids,
                max_new_tokens=max_draft_tokens,
                do_sample=True,
                temperature=0.5,  # Lower temperature for draft
                top_p=0.9,
                pad_token_id=self._draft_tokenizer.eos_token_id,
            )

            drafted_tokens = draft_ids[0][current_ids.shape[1]:]
            total_drafted += len(drafted_tokens)

            # Verification phase: check each drafted token with target model
            accepted_tokens = []
            for i, token in enumerate(drafted_tokens):
                # Get target model probability for this token
                with torch.no_grad():
                    # Prepare input for verification
                    verify_input = torch.cat([current_ids, draft_ids[:, :i+1]], dim=1)
                    outputs = self._target_model(verify_input)
                    logits = outputs.logits[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    token_prob = probs[0, token].item()

                if token_prob >= acceptance_threshold:
                    accepted_tokens.append(token)
                    total_accepted += 1
                else:
                    total_rejected += 1
                    break  # Stop at first rejection

            # Update current_ids with accepted tokens
            if accepted_tokens:
                accepted_tensor = torch.tensor([accepted_tokens], device=device)
                current_ids = torch.cat([current_ids, accepted_tensor], dim=1)
            else:
                # If no tokens accepted, generate one with target model
                with torch.no_grad():
                    outputs = self._target_model(current_ids)
                    logits = outputs.logits[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    token = torch.multinomial(probs, 1)
                    current_ids = torch.cat([current_ids, token], dim=1)
                    total_rejected += 1

            # Check for EOS
            if current_ids[0, -1].item() == self._target_tokenizer.eos_token_id:
                break

        acceptance_rate = total_accepted / total_drafted if total_drafted > 0 else 0

        return current_ids, {
            "enabled": True,
            "total_drafted": total_drafted,
            "total_accepted": total_accepted,
            "total_rejected": total_rejected,
            "acceptance_rate": acceptance_rate,
        }

    def unload(self) -> None:
        """Unload models to free memory."""
        self._target_model = None
        self._target_tokenizer = None
        self._draft_model = None
        self._draft_tokenizer = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Global instance
speculative_service = SpeculativeDecodingService()