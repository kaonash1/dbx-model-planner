"""dbx-model-planner package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("dbx-model-planner")
except PackageNotFoundError:  # pragma: no cover – editable / dev install
    __version__ = "0.0.0"

