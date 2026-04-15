"""Storage backends persist inventory snapshots and normalized model data."""

from .sqlite import SQLiteSnapshotStore, SnapshotRecord

__all__ = [
    "SQLiteSnapshotStore",
    "SnapshotRecord",
]

