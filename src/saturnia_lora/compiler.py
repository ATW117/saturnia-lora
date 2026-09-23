from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def compile_ai_toolkit(
    root: Path,
    experiment: dict[str, Any],
    model: dict[str, Any],
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    run_name: str,
) -> tuple[Path, dict[str, Any]]:
    exp = experiment["experiment"]
    mod = model["model"]
    train = protocol["train"]
    sample = model["sample"]
    train_dir = (root / ".runs" / "datasets" / exp["name"] / "train").resolve()
    output_dir = (root / ".runs" / "output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = []
    with (root / "evaluation" / "prompts.toml").open("rb") as handle:
        import tomllib
        prompts_data = tomllib.load(handle)["prompt"]
    for prompt in prompts_data:
        prompts.append(prompt["text"].replace("[trigger]", exp["trigger"]))

    process = {
        "type": "sd_trainer",
        "training_folder": str(output_dir),
        "device": mod.get("device", "cuda:0"),
        # The generated captions already contain the trigger. Keeping the global
        # trigger unset makes cached text embeddings safe and deterministic.
        "network": {
            "type": "lora",
            "linear": train["rank"],
            "linear_alpha": train["alpha"],
        },
        "save": {
            "dtype": train["dtype"],
            "save_every": train["save_every"],
            "max_step_saves_to_keep": train["max_step_saves_to_keep"],
        },
        "datasets": [{
            "folder_path": str(train_dir),
            "caption_ext": "txt",
            "caption_dropout_rate": train["caption_dropout_rate"],
            "shuffle_tokens": False,
            "cache_latents_to_disk": True,
            "resolution": train["resolutions"],
        }],
        "train": {
            "batch_size": train["batch_size"],
            "steps": train["steps"],
            "gradient_accumulation": train["gradient_accumulation"],
            "train_unet": True,
            "train_text_encoder": False,
            "gradient_checkpointing": train["gradient_checkpointing"],
            "noise_scheduler": "flowmatch",
            "optimizer": train["optimizer"],
            "lr": train["learning_rate"],
            "timestep_type": train["timestep_type"],
            "content_or_style": "balanced",
            "dtype": train["dtype"],
            "cache_text_embeddings": bool(mod.get("cache_text_embeddings", False)),
        },
        "model": {
            "name_or_path": mod["name_or_path"],
            "arch": mod["arch"],
            "quantize": bool(mod.get("quantize", False)),
            "quantize_te": bool(mod.get("quantize_te", False)),
            "low_vram": bool(mod.get("low_vram", False)),
        },
        "sample": {
            "sampler": "flowmatch",
            "sample_every": train["sample_every"],
            "sample_start_step": 0,
            "width": sample["width"],
            "height": sample["height"],
            "prompts": prompts,
            "neg": "",
            "seed": train["sample_seed"],
            "walk_seed": False,
            "guidance_scale": sample["guidance_scale"],
            "sample_steps": sample["steps"],
        },
    }
    config = {
        "job": "extension",
        "config": {
            "name": run_name,
            "process": [process],
        },
        "meta": {
            "name": "[name]",
            "version": "1.0",
            "saturnia_experiment": exp["name"],
            "saturnia_model": mod["name"],
            "train_items": manifest["counts"].get("train", 0),
            "holdout_items": manifest["counts"].get("holdout", 0),
        },
    }
    config_dir = root / ".runs" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{run_name}.yaml"
    # JSON is a strict YAML subset and avoids a runtime YAML dependency here.
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path, config
