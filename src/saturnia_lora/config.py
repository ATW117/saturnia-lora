from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?}")


def repo_root() -> Path:
    override = os.environ.get("SATURNIA_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise ValueError(f"environment variable {name} is required")

        return ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class Catalog:
    root: Path

    def names(self, kind: str) -> list[str]:
        return sorted(path.stem for path in (self.root / "configs" / kind).glob("*.toml"))

    def load(self, kind: str, name: str) -> dict[str, Any]:
        if not re.fullmatch(r"[a-z0-9_]+", name):
            raise ValueError(f"invalid {kind} name: {name!r}")
        path = self.root / "configs" / kind / f"{name}.toml"
        if not path.is_file():
            known = ", ".join(self.names(kind))
            raise FileNotFoundError(f"unknown {kind} {name!r}; choose one of: {known}")
        return expand_env(load_toml(path))

    def experiment(self, name: str) -> dict[str, Any]:
        return self.load("experiments", name)

    def model(self, name: str) -> dict[str, Any]:
        return self.load("models", name)

    def protocol(self, name: str = "style_v1") -> dict[str, Any]:
        return self.load("protocols", name)

    def prompts(self) -> list[dict[str, Any]]:
        data = load_toml(self.root / "evaluation" / "prompts.toml")
        return data["prompt"]
