"""Configuration: load, validate, hash.

Every downstream module receives a :class:`Config`, never a raw dict. A configuration is
identified by content (``Config.hash``), not by filename.

``load_config`` merges, in order: ``configs/paths.yaml``, the experiment config, and the
model and eval configs the experiment references (``includes:``, relative to the experiment
file). ``overrides`` is applied last.

This module imports neither numpy nor torch, so it can be imported before
``runlog.parallel.setup_threads`` has run.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import psutil
import yaml

DATA_ROOT_PLACEHOLDER = "<data_root>"
# paths.yaml key -> dotted key in the resolved config
_PATHS_KEYS = {"data_root": "data.root", "contract_version": "data.contract_version"}
# (dotted key, allowed types)
_REQUIRED = [
    ("experiment", (str,)),
    ("data.contract_version", (str,)),
    ("data.root", (str,)),
    ("run.seed", (int,)),
    ("run.n_jobs", (int, type(None))),
    ("eval.pairs.n", (int,)),
    ("eval.pairs.seed", (int,)),
    ("eval.permutations.B", (int,)),
]


class ConfigError(ValueError):
    """A configuration fails validation."""


def _lookup(cfg: Mapping[str, Any], dotted: str) -> tuple[bool, Any]:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return False, None
        node = node[part]
    return True, node


class Config(Mapping):
    """Mapping wrapper around a fully resolved, JSON-serializable config."""

    def __init__(self, raw: Mapping[str, Any]) -> None:
        self._raw: dict[str, Any] = json.loads(json.dumps(raw))
        self._hash = config_hash(self._raw)

    def __getitem__(self, key: str) -> Any:
        return self._raw[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._raw)

    def __len__(self) -> int:
        return len(self._raw)

    def get(self, key: str, default: Any = None) -> Any:
        """Look up ``key``; dotted paths are allowed (``"eval.pairs.n"``)."""
        found, value = _lookup(self._raw, key)
        return value if found else default

    @property
    def raw(self) -> dict[str, Any]:
        """The fully resolved config as a plain dict (a copy)."""
        return copy.deepcopy(self._raw)

    @property
    def hash(self) -> str:
        return self._hash

    @property
    def n_jobs(self) -> int:
        return resolve_n_jobs(self._raw)

    @property
    def data_root(self) -> Path:
        return Path(self._raw["data"]["root"])

    @property
    def experiment(self) -> str:
        return self._raw["experiment"]


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _deep_merge(base: dict[str, Any], new: Mapping[str, Any]) -> None:
    for key, value in new.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def _set_dotted(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    node = cfg
    for part in parents:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(f"{dotted}: {part!r} is not a mapping")
    node[leaf] = copy.deepcopy(value)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Merge ``configs/paths.yaml``, the experiment config, and the configs it references.

    ``overrides`` is a flat ``{"a.b.c": value}`` mapping applied last. Raises if
    ``configs/paths.yaml`` is missing.
    """
    exp_path = Path(path).resolve()
    paths_file = exp_path.parent.parent / "paths.yaml"  # configs/experiments/x.yaml -> configs/paths.yaml
    if not paths_file.is_file():
        raise FileNotFoundError(
            f"{paths_file} not found: copy configs/paths.example.yaml to configs/paths.yaml")

    paths = _read_yaml(paths_file)
    merged: dict[str, Any] = {}
    for key, dotted in _PATHS_KEYS.items():
        if key in paths:
            _set_dotted(merged, dotted, paths.pop(key))
    _deep_merge(merged, paths)

    experiment = _read_yaml(exp_path)
    includes = experiment.pop("includes", None) or {}
    _deep_merge(merged, experiment)
    for key, relative in includes.items():
        _deep_merge(merged, {key: _read_yaml(exp_path.parent / relative)})

    for dotted, value in (overrides or {}).items():
        _set_dotted(merged, dotted, value)

    validate_config(merged)
    return Config(merged)


def validate_config(cfg: dict) -> None:
    """Check required keys and types; raise :class:`ConfigError` naming the key path.

    ``eval.pairs.n == 500`` and ``eval.permutations.B == 10000`` are required unless
    ``run.debug`` is set.
    """
    for dotted, types in _REQUIRED:
        expected = " or ".join("null" if t is type(None) else t.__name__ for t in types)
        found, value = _lookup(cfg, dotted)
        if not found:
            raise ConfigError(f"{dotted}: required key is missing (expected {expected})")
        if not isinstance(value, types):
            raise ConfigError(f"{dotted}: expected {expected}, got {type(value).__name__} ({value!r})")

    if not (cfg.get("run") or {}).get("debug"):
        for dotted, required in (("eval.pairs.n", 500), ("eval.permutations.B", 10000)):
            _, value = _lookup(cfg, dotted)
            if value != required:
                raise ConfigError(
                    f"{dotted}: expected {required} unless run.debug is set, got {type(value).__name__} ({value!r})")


def config_hash(cfg: dict) -> str:
    """SHA-256 over ``json.dumps(cfg, sort_keys=True, separators=(",", ":"))``.

    ``data.root`` is canonicalized to the literal ``"<data_root>"`` first, so the same
    resolved config hashes identically on any machine.
    """
    canonical = copy.deepcopy(cfg)
    if isinstance(canonical.get("data"), dict) and "root" in canonical["data"]:
        canonical["data"]["root"] = DATA_ROOT_PLACEHOLDER
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def resolve_n_jobs(cfg: dict) -> int:
    """``cfg["run"]["n_jobs"]`` if set, else ``psutil.cpu_count(logical=False)``."""
    n_jobs = cfg["run"].get("n_jobs")
    return n_jobs if n_jobs is not None else psutil.cpu_count(logical=False)
