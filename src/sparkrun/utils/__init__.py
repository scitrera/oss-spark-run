"""Shared utility functions for sparkrun.

Small, self-contained helpers that are used across multiple modules.
Keeping them here avoids circular imports and reduces duplication.
"""

from __future__ import annotations


def coerce_value(value: str):
    """Coerce a string value to int, float, or bool where possible."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def resolve_ssh_user(cluster_user: str | None, config, fallback: str = "root") -> str:
    """Resolve SSH user from cluster definition, config, or OS environment."""
    import os
    return cluster_user or config.ssh_user or os.environ.get("USER", fallback)


def is_valid_ip(ip: str) -> bool:
    """Basic check if a string looks like an IPv4 address."""
    parts = ip.strip().split(".")
    if len(parts) != 4:
        return False
    return all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)


def parse_kv_output(output: str) -> dict[str, str]:
    """Parse key=value lines from script output.

    Lines starting with ``#`` are ignored. Leading/trailing whitespace
    on keys and values is stripped.

    Args:
        output: Raw stdout containing key=value lines.

    Returns:
        Dictionary of parsed key=value pairs.
    """
    result: dict[str, str] = {}
    for line in output.strip().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result


def load_yaml(path) -> dict:
    """Load a YAML file, returning an empty dict on parse failure."""
    from pathlib import Path as _Path
    import yaml
    with _Path(path).open() as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}
