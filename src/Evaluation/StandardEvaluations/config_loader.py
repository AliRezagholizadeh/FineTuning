"""
Load and merge run_config.yaml, settings, and eval_config.yaml for the evaluation module.
Paths and model/dataset names come from run_config; eval-specific options from eval_config.yaml.
"""
import yaml
from pathlib import Path

# Project root (FineTuning/) when running from project or as python -m src.Evaluate.Evaluation
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_project_root() -> Path:
    return _PROJECT_ROOT


def load_run_config(path: Path | None = None) -> dict:
    if path is None:
        path = _PROJECT_ROOT / "run_config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_eval_config(path: Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parent / "eval_config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_and_merge_config(run_config_path: Path | None = None, eval_config_path: Path | None = None) -> dict:
    """
    Load run_config, apply run_settings (paths), then merge Evaluate section from eval_config.
    Caller must run from project context so settings.run_settings can resolve paths.
    """
    run_config = load_run_config(run_config_path)
    # Apply path resolution (requires settings.run_settings)
    from settings import run_settings
    run_config = run_settings(run_config)

    eval_cfg = load_eval_config(eval_config_path)
    if "Evaluate" in eval_cfg:
        run_config["Evaluate"] = eval_cfg["Evaluate"]
    else:
        run_config["Evaluate"] = {}

    return run_config
