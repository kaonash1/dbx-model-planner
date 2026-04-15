from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..config import AppConfig, ProfileNames


@dataclass(slots=True, frozen=True)
class RuntimePaths:
    """Resolved filesystem paths used by the planner runtime."""

    config_path: Path
    config_dir: Path
    data_dir: Path
    snapshot_db_path: Path


@dataclass(slots=True, frozen=True)
class RuntimeContext:
    """Local runtime context: active profiles plus resolved paths."""

    profiles: ProfileNames
    paths: RuntimePaths


def build_runtime_context(
    config: AppConfig | None = None,
    *,
    config_path: Path | str | None = None,
    data_dir: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> RuntimeContext:
    """Build a runtime context from config and local path defaults."""

    env_map = dict(os.environ if env is None else env)
    resolved_config_path = _resolve_config_path(config_path, env_map)
    resolved_data_dir = _resolve_data_dir(data_dir, env_map)
    resolved_data_dir.mkdir(parents=True, exist_ok=True)

    runtime_config = config or AppConfig()
    paths = RuntimePaths(
        config_path=resolved_config_path,
        config_dir=resolved_config_path.parent,
        data_dir=resolved_data_dir,
        snapshot_db_path=resolved_data_dir / "snapshots.sqlite3",
    )
    return RuntimeContext(profiles=runtime_config.profiles, paths=paths)


def _resolve_config_path(config_path: Path | str | None, env: Mapping[str, str]) -> Path:
    if config_path is not None:
        return Path(config_path).expanduser()

    for env_name in ("DBX_MODEL_PLANNER_CONFIG", "DBX_MODEL_PLANNER_CONFIG_PATH"):
        value = env.get(env_name)
        if value:
            return Path(value).expanduser()

    config_home = Path(env.get("XDG_CONFIG_HOME", Path.home() / ".config")).expanduser()
    return config_home / "dbx-model-planner" / "config.toml"


def _resolve_data_dir(data_dir: Path | str | None, env: Mapping[str, str]) -> Path:
    if data_dir is not None:
        return Path(data_dir).expanduser()

    if value := env.get("DBX_MODEL_PLANNER_DATA_DIR"):
        return Path(value).expanduser()

    data_home = Path(env.get("XDG_DATA_HOME", Path.home() / ".local" / "share")).expanduser()
    return data_home / "dbx-model-planner"

