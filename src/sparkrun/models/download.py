"""HuggingFace model download utilities.

Supports both standard HuggingFace models (full repo download via
``snapshot_download``) and GGUF quantized models (selective download
of a specific quant variant).

GGUF model specs use colon syntax: ``"Qwen/Qwen3-1.7B-GGUF:Q4_K_M"``
where the part after ``:`` selects which quantization files to download.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sparkrun.config import DEFAULT_HF_CACHE_DIR

logger = logging.getLogger(__name__)

# Container-side mount point for the HuggingFace cache (set by build_volumes)
CONTAINER_HF_CACHE = "/root/.cache/huggingface"


def _hub_cache(cache_dir: str | None = None) -> str:
    """Return the HuggingFace Hub cache directory.

    ``snapshot_download(cache_dir=X)`` stores files at ``X/models--{name}/``.
    The standard HF Hub cache is ``~/.cache/huggingface/hub/``, so we must
    pass the ``hub/`` subdirectory to ``snapshot_download`` (and to
    ``huggingface-cli download --cache-dir``) to keep downloads consistent
    with the default HF cache layout.

    Our volume mount maps ``DEFAULT_HF_CACHE_DIR`` (``~/.cache/huggingface``)
    to ``/root/.cache/huggingface`` inside containers, so the ``hub/``
    subdirectory is preserved on both sides.
    """
    base = cache_dir or str(DEFAULT_HF_CACHE_DIR)
    return base + "/hub"


# ---------------------------------------------------------------------------
# GGUF model spec helpers
# ---------------------------------------------------------------------------

def parse_gguf_model_spec(model_id: str) -> tuple[str, str | None]:
    """Parse a GGUF model specification into (repo_id, quant_variant).

    The colon syntax ``repo:quant`` selects a specific quantization
    variant from a GGUF repository on HuggingFace.

    Examples::

        >>> parse_gguf_model_spec("Qwen/Qwen3-1.7B-GGUF:Q4_K_M")
        ('Qwen/Qwen3-1.7B-GGUF', 'Q4_K_M')
        >>> parse_gguf_model_spec("Qwen/Qwen3-1.7B-GGUF")
        ('Qwen/Qwen3-1.7B-GGUF', None)
        >>> parse_gguf_model_spec("meta-llama/Llama-3-8B")
        ('meta-llama/Llama-3-8B', None)
    """
    if ":" in model_id:
        repo_id, quant = model_id.rsplit(":", 1)
        return repo_id, quant
    return model_id, None


def is_gguf_model(model_id: str) -> bool:
    """Check if a model spec refers to a GGUF model.

    Returns True when the repo name contains ``GGUF`` (case-insensitive)
    or when a quant variant is specified via colon syntax (``repo:quant``).
    """
    repo_id, quant = parse_gguf_model_spec(model_id)
    if quant is not None:
        return True
    return "gguf" in repo_id.lower()


def resolve_gguf_path(
    model_id: str,
    cache_dir: str | None = None,
) -> str | None:
    """Resolve the local cache path to a GGUF file.

    Searches the HuggingFace cache structure for a ``.gguf`` file
    matching the model specification.

    Args:
        model_id: GGUF model spec (e.g. ``"Qwen/Qwen3-1.7B-GGUF:Q4_K_M"``).
        cache_dir: Override for the HuggingFace cache directory.

    Returns:
        Absolute path to the ``.gguf`` file, or ``None`` if not found.
    """
    repo_id, quant = parse_gguf_model_spec(model_id)
    cache = Path(cache_dir or str(DEFAULT_HF_CACHE_DIR))
    safe_name = repo_id.replace("/", "--")
    model_cache = cache / "hub" / ("models--" + safe_name)

    if not model_cache.exists():
        return None

    snapshots = model_cache / "snapshots"
    if not snapshots.exists():
        return None

    # Search snapshot directories (newest first) for matching .gguf files
    for snapshot_dir in sorted(snapshots.iterdir(), reverse=True):
        if not snapshot_dir.is_dir():
            continue
        for gguf_file in snapshot_dir.glob("*.gguf"):
            if quant is None:
                return str(gguf_file)
            if quant.lower() in gguf_file.name.lower():
                return str(gguf_file)

    return None


def resolve_gguf_container_path(
    model_id: str,
    cache_dir: str | None = None,
    container_cache: str = CONTAINER_HF_CACHE,
) -> str | None:
    """Resolve the container-internal path to a cached GGUF file.

    Translates the host cache path to the container mount path by
    replacing the host cache prefix with the container mount point.

    Args:
        model_id: GGUF model spec.
        cache_dir: Host-side HuggingFace cache directory.
        container_cache: Container-side cache mount point.

    Returns:
        Container-internal path to the ``.gguf`` file, or ``None``.
    """
    host_path = resolve_gguf_path(model_id, cache_dir)
    if not host_path:
        return None

    cache = cache_dir or str(DEFAULT_HF_CACHE_DIR)
    # Replace host cache prefix with container mount path
    if host_path.startswith(cache):
        return container_cache + host_path[len(cache):]
    return None


# ---------------------------------------------------------------------------
# Cache checking
# ---------------------------------------------------------------------------

def is_model_cached(model_id: str, cache_dir: str | None = None) -> bool:
    """Check if a model is already cached locally.

    Inspects the HuggingFace cache directory structure to determine
    whether the model has been previously downloaded.

    For GGUF models with a quant variant (``repo:quant``), checks
    whether a matching ``.gguf`` file exists in the cache.

    Args:
        model_id: HuggingFace model identifier (e.g. ``"meta-llama/Llama-3-8B"``)
            or GGUF spec (e.g. ``"Qwen/Qwen3-1.7B-GGUF:Q4_K_M"``).
        cache_dir: Override for the HuggingFace cache directory.

    Returns:
        True if the model cache directory exists and is non-empty.
    """
    # For GGUF models, check for the specific quant file
    if is_gguf_model(model_id):
        return resolve_gguf_path(model_id, cache_dir) is not None

    cache = Path(cache_dir or str(DEFAULT_HF_CACHE_DIR))
    # HF cache structure: hub/models--org--name/
    safe_name = model_id.replace("/", "--")
    model_cache = cache / "hub" / f"models--{safe_name}"
    return model_cache.exists() and any(model_cache.iterdir())


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_model(
    model_id: str,
    cache_dir: str | None = None,
    token: str | None = None,
    dry_run: bool = False,
) -> int:
    """Download a model from HuggingFace Hub.

    Automatically detects GGUF model specs (``repo:quant``) and
    downloads only the matching quantization files instead of the
    entire repository.

    Args:
        model_id: HuggingFace model identifier or GGUF spec.
        cache_dir: Override for the HuggingFace cache directory.
        token: Optional HuggingFace API token for gated models.
        dry_run: If True, show what would be done without executing.

    Returns:
        Exit code (0 = success).
    """
    if is_gguf_model(model_id):
        return _download_gguf(model_id, cache_dir=cache_dir, token=token, dry_run=dry_run)

    cache = cache_dir or str(DEFAULT_HF_CACHE_DIR)

    if dry_run:
        logger.info("[dry-run] Would download model: %s to %s", model_id, cache)
        return 0

    if is_model_cached(model_id, cache):
        logger.info("Model %s already cached at %s", model_id, cache)
        return 0

    logger.info("Downloading model: %s...", model_id)

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_id, cache_dir=_hub_cache(cache), token=token)
        logger.info("Model downloaded successfully: %s", model_id)
        return 0
    except Exception as e:
        logger.error("Failed to download model %s: %s", model_id, e)
        return 1


def _download_gguf(
    model_id: str,
    cache_dir: str | None = None,
    token: str | None = None,
    dry_run: bool = False,
) -> int:
    """Download a GGUF model, fetching only the matching quant files.

    Uses ``snapshot_download`` with ``allow_patterns`` to avoid
    downloading every quantization variant in the repository.

    Args:
        model_id: GGUF model spec (e.g. ``"Qwen/Qwen3-1.7B-GGUF:Q4_K_M"``).
        cache_dir: Override for the HuggingFace cache directory.
        token: Optional HuggingFace API token.
        dry_run: If True, show what would be done without executing.

    Returns:
        Exit code (0 = success).
    """
    repo_id, quant = parse_gguf_model_spec(model_id)
    cache = cache_dir or str(DEFAULT_HF_CACHE_DIR)

    if dry_run:
        logger.info("[dry-run] Would download GGUF model: %s (quant=%s) to %s",
                     repo_id, quant, cache)
        return 0

    # Check if matching GGUF file already cached
    if resolve_gguf_path(model_id, cache) is not None:
        logger.info("GGUF model %s already cached", model_id)
        return 0

    logger.info("Downloading GGUF model: %s (quant=%s)...", repo_id, quant or "any")

    try:
        from huggingface_hub import snapshot_download

        kwargs: dict = {"repo_id": repo_id, "cache_dir": _hub_cache(cache), "token": token}
        if quant:
            # Download only files matching the quant variant
            kwargs["allow_patterns"] = ["*%s*" % quant]

        snapshot_download(**kwargs)
        logger.info("GGUF model downloaded successfully: %s", model_id)
        return 0
    except Exception as e:
        logger.error("Failed to download GGUF model %s: %s", model_id, e)
        return 1
