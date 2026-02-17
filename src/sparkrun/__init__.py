"""sparkrun - Launch and manage Docker-based inference workloads on NVIDIA DGX Spark systems."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("sparkrun")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
