"""
LLM-as-a-Judge: use a stronger model (e.g. GPT-4, Claude, or a Hugging Face model) to grade
model outputs on criteria such as accuracy, relevance, and tone.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Optional: OpenAI
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    OpenAI = None

# Optional: Anthropic
try:
    from anthropic import Anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False
    Anthropic = None

# Optional: Hugging Face (transformers)
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for text generation. Score the model's completion relative to the reference answer and the user prompt.
Output only a single JSON object with one number (0-10) per criterion. No other text.
Example: {"accuracy": 8, "relevance": 9, "tone": 7}"""


def _build_judge_prompt(prompt: str, reference: str, candidate: str, criteria: list[str]) -> str:
    criteria_str = ", ".join(criteria)
    return f"""## User prompt
{prompt}

## Reference (ground truth) completion
{reference}

## Model completion to score
{candidate}

## Task
Score the model completion from 0 to 10 for each criterion: {criteria_str}.
Output only a JSON object with keys: {json.dumps(criteria)}. One number per key."""


def _call_openai(
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("openai package not installed; pip install openai")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    return (resp.choices[0].message.content or "").strip()


def _call_anthropic(
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    if not _ANTHROPIC_AVAILABLE:
        raise RuntimeError("anthropic package not installed; pip install anthropic")
    client = Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=256,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = resp.content[0].text if resp.content else ""
    return text.strip()


def load_huggingface_judge(
    model_name: str,
    token: str | None = None,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    logger_instance: logging.Logger | None = None,
) -> tuple[Any, Any]:
    """
    Load a Hugging Face model and tokenizer for use as judge.
    Returns (model, tokenizer). Use with score_batch(..., preloaded_judge=(model, tokenizer)).
    """
    if not _HF_AVAILABLE:
        raise RuntimeError("transformers and torch required for Hugging Face judge; pip install transformers torch")
    log = logger_instance or logger
    log.info("Loading Hugging Face judge model: %s", model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    kwargs = {"device_map": device_map}
    if torch_dtype != "auto":
        kwargs["torch_dtype"] = getattr(torch, torch_dtype, torch.float32) if isinstance(torch_dtype, str) else torch_dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        **kwargs,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def _call_huggingface(
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 256,
) -> str:
    """Run one judge query through a loaded Hugging Face model. Uses chat template if available."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = f"{system_prompt}\n\n{user_prompt}\n\nOutput only a JSON object:"
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
    gen_ids = out[:, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()


def _parse_judge_response(raw: str, criteria: list[str]) -> dict[str, float]:
    """Extract JSON from judge output and return criterion -> score."""
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    out = json.loads(raw)
    return {c: float(out.get(c, 0.0)) for c in criteria}


def score_one(
    prompt: str,
    reference: str,
    candidate: str,
    criteria: list[str],
    provider: str,
    model: str,
    api_key: str | None = None,
    env_api_key_name: str = "OPENAI_API_KEY",
    preloaded_judge: tuple[Any, Any] | None = None,
) -> dict[str, float]:
    """
    Score a single (prompt, reference, candidate) with the judge LLM.
    Returns a dict of criterion -> score (0-10).
    For provider "huggingface", pass preloaded_judge=(model, tokenizer) to avoid loading per call.
    """
    user_prompt = _build_judge_prompt(prompt, reference, candidate, criteria)
    provider_l = provider.lower()

    if provider_l == "huggingface":
        if preloaded_judge is None:
            token = api_key or os.environ.get(env_api_key_name)
            if not token:
                logger.warning("Hugging Face token not set (%s); returning zeros", env_api_key_name)
                return {c: 0.0 for c in criteria}
            hf_model, hf_tokenizer = load_huggingface_judge(model, token=token)
            preloaded_judge = (hf_model, hf_tokenizer)
        hf_model, hf_tokenizer = preloaded_judge
        try:
            raw = _call_huggingface(hf_model, hf_tokenizer, JUDGE_SYSTEM_PROMPT, user_prompt)
            return _parse_judge_response(raw, criteria)
        except (json.JSONDecodeError, KeyError, Exception) as e:
            logger.warning("Hugging Face judge parse/error: %s", e)
            return {c: 0.0 for c in criteria}

    api_key = api_key or os.environ.get(env_api_key_name)
    if not api_key:
        logger.warning("Judge API key not set (%s); returning zeros", env_api_key_name)
        return {c: 0.0 for c in criteria}
    try:
        if provider_l == "openai":
            raw = _call_openai(model, api_key, JUDGE_SYSTEM_PROMPT, user_prompt)
        elif provider_l == "anthropic":
            raw = _call_anthropic(model, api_key, JUDGE_SYSTEM_PROMPT, user_prompt)
        else:
            logger.warning("Unknown judge provider %s; use openai, anthropic, or huggingface", provider)
            return {c: 0.0 for c in criteria}
        return _parse_judge_response(raw, criteria)
    except (json.JSONDecodeError, KeyError, Exception) as e:
        logger.warning("Judge parse/API error: %s", e)
        return {c: 0.0 for c in criteria}


def score_batch(
    prompts: list[str],
    references: list[str],
    candidates: list[str],
    criteria: list[str],
    provider: str,
    model: str,
    env_api_key_name: str = "OPENAI_API_KEY",
    api_key: str | None = None,
    preloaded_judge: tuple[Any, Any] | None = None,
    logger_instance: logging.Logger | None = None,
) -> list[dict[str, float]]:
    """
    Score each (prompt, reference, candidate) and return list of score dicts.
    For provider "huggingface", pass preloaded_judge=(model, tokenizer) to reuse one loaded model
    (otherwise loads once at start of batch). Use env_api_key_name for Hugging Face token env when
    provider is huggingface (e.g. HUG_ACCESS_TOKEN).
    """
    if provider.lower() == "huggingface" and preloaded_judge is None:
        token = api_key or os.environ.get(env_api_key_name)
        if token:
            preloaded_judge = load_huggingface_judge(model, token=token, logger_instance=logger_instance)
    return [
        score_one(
            p, r, c, criteria, provider, model,
            api_key=api_key,
            env_api_key_name=env_api_key_name,
            preloaded_judge=preloaded_judge,
        )
        for p, r, c in zip(prompts, references, candidates)
    ]


def aggregate_judge_scores(per_sample: list[dict[str, float]], criteria: list[str]) -> dict[str, float]:
    """Average per-sample judge scores into one dict (e.g. for one model)."""
    if not per_sample:
        return {f"judge_{c}": 0.0 for c in criteria}
    n = len(per_sample)
    agg: dict[str, float] = {}
    for c in criteria:
        key = f"judge_{c}"
        agg[key] = sum(s.get(c, 0.0) for s in per_sample) / n
    return agg
