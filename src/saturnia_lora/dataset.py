from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Item:
    source: str
    source_image: str
    source_caption: str
    prepared_name: str
    split: str
    width: int
    height: int
    sha256: str
    caption: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_size(path: Path) -> tuple[int, int]:
    """Read PNG/JPEG dimensions without pulling training dependencies into this tool."""
    with path.open("rb") as handle:
        head = handle.read(24)
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            return struct.unpack(">II", head[16:24])
        if head[:2] != b"\xff\xd8":
            raise ValueError(f"unsupported image format: {path}")
        handle.seek(2)
        while True:
            byte = handle.read(1)
            if not byte:
                break
            if byte != b"\xff":
                continue
            marker = handle.read(1)
            while marker == b"\xff":
                marker = handle.read(1)
            if marker in {b"\xd8", b"\xd9"}:
                continue
            length_raw = handle.read(2)
            if len(length_raw) != 2:
                break
            length = struct.unpack(">H", length_raw)[0]
            if marker and marker[0] in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                payload = handle.read(5)
                height, width = struct.unpack(">HH", payload[1:5])
                return width, height
            handle.seek(length - 2, 1)
    raise ValueError(f"could not read image dimensions: {path}")


def content_only_caption(raw: str, trigger: str) -> str:
    normalized = " ".join(raw.split())
    for separator in (",", "."):
        prefix = f"{trigger}{separator}"
        if normalized.casefold().startswith(prefix.casefold()):
            normalized = normalized[len(prefix):].lstrip()
            break
    first, marker, _rest = normalized.partition(". ")
    content = first.rstrip(". ").strip()
    if not content:
        raise ValueError("caption has no content sentence")
    return f"{trigger}. {content}."


def transform_caption(raw: str, trigger: str, policy: str) -> str:
    if policy == "content_only":
        return content_only_caption(raw, trigger)
    if policy == "as_is":
        normalized = " ".join(raw.split())
        return normalized if normalized.casefold().startswith(trigger.casefold()) else f"{trigger}. {normalized}"
    raise ValueError(f"unknown caption policy: {policy}")


def scan_source(root: Path, source: str) -> list[tuple[Path, Path]]:
    directory = (root / source).resolve()
    try:
        directory.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"source escapes repository: {source}") from error
    if not directory.is_dir():
        raise FileNotFoundError(f"dataset source is not a directory: {directory}")
    images = sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"dataset source has no images: {directory}")
    pairs = []
    for image in images:
        caption = image.with_suffix(".txt")
        if not caption.is_file():
            raise FileNotFoundError(f"missing caption for {image}")
        pairs.append((image, caption))
    return pairs


def _holdouts(pairs: list[tuple[Path, Path]], source: str, seed: int, fraction: float) -> set[Path]:
    if not 0 <= fraction < 1:
        raise ValueError("holdout_fraction must be in [0, 1)")
    count = round(len(pairs) * fraction)
    if fraction > 0 and len(pairs) > 1:
        count = max(1, count)
    ranked = sorted(
        pairs,
        key=lambda pair: hashlib.sha256(f"{seed}:{source}:{pair[0].name}".encode()).digest(),
    )
    return {image for image, _ in ranked[:count]}


def _safe_reset(path: Path, runs_root: Path) -> None:
    resolved = path.resolve()
    resolved.relative_to(runs_root.resolve())
    if resolved.exists():
        shutil.rmtree(resolved)


def input_fingerprint(root: Path, cfg: dict) -> str:
    """Cheaply detect source/config changes without invalidating latent caches."""
    records: list[object] = [cfg]
    for source in cfg["sources"]:
        for image, caption in scan_source(root, source):
            image_stat = image.stat()
            caption_stat = caption.stat()
            records.append([
                str(image.relative_to(root)),
                image_stat.st_size,
                image_stat.st_mtime_ns,
                str(caption.relative_to(root)),
                caption_stat.st_size,
                caption_stat.st_mtime_ns,
            ])
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_dataset(root: Path, experiment: dict, force: bool = False) -> dict:
    cfg = experiment["experiment"]
    name = cfg["name"]
    runs_root = root / ".runs"
    target = runs_root / "datasets" / name
    manifest_path = target / "manifest.json"
    fingerprint = input_fingerprint(root, cfg)
    if manifest_path.is_file() and not force:
        existing = json.loads(manifest_path.read_text())
        if existing.get("input_fingerprint") == fingerprint:
            return existing
    _safe_reset(target, runs_root)
    train_dir = target / "train"
    train_dir.mkdir(parents=True)

    items: list[Item] = []
    seen_names: set[str] = set()
    for source in cfg["sources"]:
        pairs = scan_source(root, source)
        holdouts = _holdouts(pairs, source, int(cfg["split_seed"]), float(cfg["holdout_fraction"]))
        source_tag = source.replace("/", "__")
        for image, caption_path in pairs:
            prepared_name = f"{source_tag}__{image.name}"
            if prepared_name in seen_names:
                raise ValueError(f"prepared filename collision: {prepared_name}")
            seen_names.add(prepared_name)
            caption = transform_caption(
                caption_path.read_text(encoding="utf-8"),
                cfg["trigger"],
                cfg["caption_policy"],
            )
            split = "holdout" if image in holdouts else "train"
            width, height = image_size(image)
            item = Item(
                source=source,
                source_image=str(image.relative_to(root)),
                source_caption=str(caption_path.relative_to(root)),
                prepared_name=prepared_name,
                split=split,
                width=width,
                height=height,
                sha256=sha256(image),
                caption=caption,
            )
            items.append(item)
            if split == "train":
                destination = train_dir / prepared_name
                try:
                    os.symlink(image.resolve(), destination)
                except OSError:
                    os.link(image.resolve(), destination)
                destination.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")

    hashes = Counter(item.sha256 for item in items)
    manifest = {
        "schema_version": 1,
        "input_fingerprint": fingerprint,
        "experiment": cfg,
        "counts": dict(Counter(item.split for item in items)),
        "duplicate_groups": sum(1 for count in hashes.values() if count > 1),
        "items": [asdict(item) for item in items],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def source_counts(root: Path, sources: Iterable[str]) -> dict[str, int]:
    return {source: len(scan_source(root, source)) for source in sources}
