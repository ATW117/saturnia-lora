from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

from .compiler import compile_ai_toolkit
from .config import Catalog, repo_root
from .dataset import build_dataset, source_counts


def safe_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_.-")
    if not value:
        raise ValueError("run name is empty after sanitizing")
    return value


def prepare_one(catalog: Catalog, name: str, force: bool = False) -> dict:
    manifest = build_dataset(catalog.root, catalog.experiment(name), force=force)
    counts = manifest["counts"]
    print(f"{name}: {counts.get('train', 0)} train, {counts.get('holdout', 0)} holdout")
    return manifest


def compile_one(catalog: Catalog, experiment_name: str, model_name: str, run_name: str | None = None):
    experiment = catalog.experiment(experiment_name)
    manifest = build_dataset(catalog.root, experiment)
    run_name = safe_name(run_name or f"{experiment_name}__{model_name}")
    path, config = compile_ai_toolkit(
        catalog.root,
        experiment,
        catalog.model(model_name),
        catalog.protocol(),
        manifest,
        run_name,
    )
    return path, config, manifest


def git_revision(directory: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def command_list(catalog: Catalog) -> int:
    print("Experiments")
    for name in catalog.names("experiments"):
        cfg = catalog.experiment(name)["experiment"]
        try:
            counts = source_counts(catalog.root, cfg["sources"])
            status = f"{sum(counts.values())} images"
        except FileNotFoundError:
            status = "local dataset needed"
        print(f"  {name:24} {status:20} | {', '.join(cfg['sources'])}")
    print("Models")
    for name in catalog.names("models"):
        model = catalog.model(name)["model"]
        print(f"  {name:24} {model['name_or_path']}")
    return 0


def command_run(catalog: Catalog, args: argparse.Namespace) -> int:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"{args.experiment}__{args.model}__{stamp}"
    path, config, manifest = compile_one(catalog, args.experiment, args.model, run_name)
    toolkit = Path(args.ai_toolkit_dir).expanduser().resolve()
    runner = toolkit / "run.py"
    if not runner.is_file():
        raise FileNotFoundError(f"AI Toolkit run.py not found at {runner}")
    metadata = {
        "run_name": config["config"]["name"],
        "created_at": stamp,
        "config_path": str(path),
        "dataset_manifest": str(catalog.root / ".runs" / "datasets" / args.experiment / "manifest.json"),
        "ai_toolkit_dir": str(toolkit),
        "ai_toolkit_revision": git_revision(toolkit),
        "python": sys.executable,
        "command": [sys.executable, str(runner), str(path)],
        "train_items": manifest["counts"].get("train", 0),
    }
    meta_path = catalog.root / ".runs" / "configs" / f"{config['config']['name']}.run.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Running {config['config']['name']}")
    print(f"Config: {path}")
    return subprocess.run(metadata["command"], cwd=toolkit, check=False).returncode


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build and run reproducible Saturnia style-LoRA experiments")
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="show experiment and model catalog")

    prepare = sub.add_parser("prepare", help="materialize deterministic training datasets")
    prepare.add_argument("experiment", nargs="?")
    prepare.add_argument("--all", action="store_true")
    prepare.add_argument("--force", action="store_true")

    compile_cmd = sub.add_parser("compile", help="emit an AI Toolkit YAML config")
    compile_cmd.add_argument("--experiment", required=True)
    compile_cmd.add_argument("--model", required=True)
    compile_cmd.add_argument("--run-name")

    matrix = sub.add_parser("matrix", help="print every experiment/model command")
    matrix.add_argument("--ai-toolkit-dir", default="${AI_TOOLKIT_DIR}")
    matrix.add_argument("--model", choices=Catalog(repo_root()).names("models"))

    run = sub.add_parser("run", help="compile then execute one AI Toolkit job")
    run.add_argument("--experiment", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--ai-toolkit-dir", required=True)
    run.add_argument("--run-name")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    catalog = Catalog(repo_root())
    try:
        if args.command == "list":
            return command_list(catalog)
        if args.command == "prepare":
            if args.all:
                for name in catalog.names("experiments"):
                    prepare_one(catalog, name, args.force)
                return 0
            if not args.experiment:
                raise ValueError("provide an experiment name or --all")
            prepare_one(catalog, args.experiment, args.force)
            return 0
        if args.command == "compile":
            path, _, _ = compile_one(catalog, args.experiment, args.model, args.run_name)
            print(path)
            return 0
        if args.command == "matrix":
            for experiment in catalog.names("experiments"):
                for model in ([args.model] if args.model else catalog.names("models")):
                    print(
                        "PYTHONPATH=src python3 -m saturnia_lora run "
                        f"--experiment {experiment} --model {model} "
                        f"--ai-toolkit-dir {args.ai_toolkit_dir}"
                    )
            return 0
        if args.command == "run":
            return command_run(catalog, args)
        raise AssertionError(args.command)
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
