"""
Core evaluation module: compare fine-tuned model vs baseline (PreFineTuned) over the eval dataset
using ROUGE, BLEU, BERTScore, and optional LLM-as-a-Judge.
Config is inherited from run_config.yaml, settings.py, and eval_config.yaml.
"""
from __future__ import annotations

import copy
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# Project root must be on path when running this module (e.g. run from FineTuning/)
from src.Evaluate.config_loader import get_project_root, load_and_merge_config
from src.Evaluate.judge import aggregate_judge_scores, load_huggingface_judge, score_batch
from src.Evaluate.metrics import compute_all

# FineTuning paths and dataset
from src.FineTuning.tools.LoadDataset import MobileActionsDS
from src.FineTuning.tools.LoadModel import Load_Hugging_Face_Model
from src.FineTuning.tools.utils import get_model_dir_path


def _setup_logging(results_dir: Path) -> logging.Logger:
    log_dir = get_project_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(ZoneInfo("America/Toronto")).strftime("%Y-%m-%dT%H:%M:%S")
    log_file = log_dir / f"Evaluate_{ts}.log"
    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
    )
    return logging.getLogger(__name__)


def _load_finetune_config() -> dict:
    import yaml
    path = get_project_root() / "src" / "FineTuning" / "FunctionGemma" / "finetune_config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _get_baseline_model_path(config: dict) -> Path:
    cfg = copy.deepcopy(config)
    cfg["FineTune"]["FROM"] = "PreFineTuned"
    return get_model_dir_path(cfg, to_save=False)


def _get_finetuned_model_path(config: dict) -> Path:
    return get_model_dir_path(config, eval_fine_tuned = True, to_save=False)


def _generate(model, tokenizer, prompts: list[str], generation_config: dict, device, logger) -> list[str]:
    import torch
    outputs: list[str] = []
    max_new_tokens = generation_config.get("max_new_tokens", 1024)
    do_sample = generation_config.get("do_sample", False)
    temperature = generation_config.get("temperature", 0.0)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    model.eval()
    for i, prompt in enumerate(prompts):
        if (i + 1) % 50 == 0:
            logger.info("Generated %d / %d", i + 1, len(prompts))
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=pad_token_id,
            )
        # decode only the new tokens
        gen_ids = out[:, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=False)
        outputs.append(text.strip())
    return outputs


def run_evaluation(
    run_config_path: Path | None = None,
    eval_config_path: Path | None = None,
) -> dict:
    """
    Load configs, models, and eval dataset; run inference; compute metrics and optional judge; save results.
    """
    config = load_and_merge_config(run_config_path, eval_config_path)
    results_dir = Path(config["SolidConfig"]["EVAL_RESULTS_DIR_PATH"])
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logging(results_dir)

    # Env
    load_dotenv(config["SolidConfig"]["ENV_DIR"])
    access_token = os.environ.get(config["SolidConfig"]["HUGGINGFACE_ACCESS_TOKEN_ENV_NAME"], "")

    eval_cfg = config.get("Evaluate", {})
    use_finetuned = eval_cfg.get("use_fine_tuned", True)
    use_baseline = eval_cfg.get("use_baseline", True)
    max_samples = eval_cfg.get("max_eval_samples")
    metrics_cfg = eval_cfg.get("metrics", {})
    judge_cfg = eval_cfg.get("llm_judge", {})
    gen_cfg = eval_cfg.get("generation", {})

    # Paths
    finetuned_path = _get_finetuned_model_path(config)
    baseline_path = _get_baseline_model_path(config)

    # Load tokenizer and dataset (use tokenizer from fine-tuned or baseline)
    finetune_yaml = _load_finetune_config()
    auto_cfg = finetune_yaml.get("AutoModelForCausalLM", {})

    tokenizer_path = str(finetuned_path if use_finetuned and finetuned_path.exists() else baseline_path)
    print(f"tokenizer_path: {tokenizer_path}")
    from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ds_helper = MobileActionsDS(config, tokenizer, logger)
    ds_helper.Load_Data()
    _, eval_ds = ds_helper.train_eval_split()
    eval_ds = list(eval_ds)
    if max_samples is not None:
        eval_ds = eval_ds[: max_samples]
    logger.info("Eval dataset size: %d", len(eval_ds))

    prompts = [x["prompt"] for x in eval_ds]
    references = [x["completion"] for x in eval_ds]

    results = {
        "config": {
            "use_fine_tuned": use_finetuned,
            "use_baseline": use_baseline,
            "num_eval_samples": len(eval_ds),
        },
        "models": {},
        "timestamp": datetime.now(ZoneInfo("America/Toronto")).isoformat(),
    }

    # Optional: preload Hugging Face judge once (reused for both models)
    preloaded_judge = None
    if judge_cfg.get("enabled") and judge_cfg.get("provider", "").lower() == "huggingface":
        hf_token = os.environ.get(judge_cfg.get("env_api_key_name", "HUG_ACCESS_TOKEN"))
        if hf_token:
            preloaded_judge = load_huggingface_judge(
                judge_cfg.get("model", ""),
                token=hf_token,
                logger_instance=logger,
            )
        else:
            logger.warning("Hugging Face judge enabled but token not set; judge scores will be skipped")

    # Load and run each model
    if use_finetuned and finetuned_path.exists():
        logger.info("Loading fine-tuned model from %s", finetuned_path)
        model_ft, _, tok_ft, device_ft = Load_Hugging_Face_Model(
            access_token=access_token,
            model_name=str(finetuned_path),
            model_path=str(finetuned_path),
            AutoModelForCausalLM_setting=auto_cfg,
            logger=logger,
        )
        model_ft.config.pad_token_id = tok_ft.pad_token_id or tok_ft.eos_token_id
        pred_ft = _generate(model_ft, tok_ft, prompts, gen_cfg, device_ft, logger)
        results["models"]["fine_tuned"] = compute_all(
            references, pred_ft,
            rouge=metrics_cfg.get("rouge", True),
            bleu=metrics_cfg.get("bleu", True),
            bertscore=metrics_cfg.get("bertscore", True),
        )
        if judge_cfg.get("enabled"):
            criteria = judge_cfg.get("criteria", ["accuracy", "relevance", "tone"])
            judge_scores = score_batch(
                prompts, references, pred_ft,
                criteria=criteria,
                provider=judge_cfg.get("provider", "openai"),
                model=judge_cfg.get("model", "gpt-4o-mini"),
                env_api_key_name=judge_cfg.get("env_api_key_name", "OPENAI_API_KEY"),
                api_key=access_token if judge_cfg.get("provider", "").lower() == "huggingface" else None,
                preloaded_judge=preloaded_judge,
                logger_instance=logger,
            )
            results["models"]["fine_tuned"].update(aggregate_judge_scores(judge_scores, criteria))
        del model_ft
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if use_baseline:
        logger.info("Loading baseline (PreFineTuned) from %s", baseline_path)
        hug_name = config["FineTune"]["Model"]["PreFineTuned"]["hug_base_model_name"]
        model_bl, _, tok_bl, device_bl = Load_Hugging_Face_Model(
            access_token=access_token,
            model_name=hug_name,
            model_path=str(baseline_path),
            AutoModelForCausalLM_setting=auto_cfg,
            logger=logger,
        )
        model_bl.config.pad_token_id = tok_bl.pad_token_id or tok_bl.eos_token_id
        pred_bl = _generate(model_bl, tok_bl, prompts, gen_cfg, device_bl, logger)
        results["models"]["baseline"] = compute_all(
            references, pred_bl,
            rouge=metrics_cfg.get("rouge", True),
            bleu=metrics_cfg.get("bleu", True),
            bertscore=metrics_cfg.get("bertscore", True),
        )
        if judge_cfg.get("enabled"):
            criteria = judge_cfg.get("criteria", ["accuracy", "relevance", "tone"])
            judge_scores = score_batch(
                prompts, references, pred_bl,
                criteria=criteria,
                provider=judge_cfg.get("provider", "openai"),
                model=judge_cfg.get("model", "gpt-4o-mini"),
                env_api_key_name=judge_cfg.get("env_api_key_name", "OPENAI_API_KEY"),
                api_key=access_token if judge_cfg.get("provider", "").lower() == "huggingface" else None,
                preloaded_judge=preloaded_judge,
                logger_instance=logger,
            )
            results["models"]["baseline"].update(aggregate_judge_scores(judge_scores, criteria))
        del model_bl
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save
    out_file = results_dir / f"eval_results_{datetime.now(ZoneInfo('America/Toronto')).strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_file)
    return results


if __name__ == "__main__":
    # Run from project root (FineTuning/):  python -m src.Evaluate.Evaluation
    run_evaluation()
