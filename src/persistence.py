from __future__ import annotations

import json
from pathlib import Path

_STATE_FILE = Path("data/indexed_files.json")


def _load() -> dict[str, list[str]]:
    """Load the full state file. Returns {index_name: [filenames]}."""
    if not _STATE_FILE.exists():
        return {}
    try:
        return json.loads(_STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save(state: dict[str, list[str]]) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state, indent=2))


def load_files(index_name: str) -> set[str]:
    """Return the set of filenames already indexed under *index_name*."""
    return set(_load().get(index_name, []))


def add_file(index_name: str, filename: str) -> None:
    """Record *filename* as indexed under *index_name*."""
    state = _load()
    files = set(state.get(index_name, []))
    files.add(filename)
    state[index_name] = sorted(files)
    _save(state)


def remove_file(index_name: str, filename: str) -> None:
    """Remove *filename* from the persisted list for *index_name*."""
    state = _load()
    files = set(state.get(index_name, []))
    files.discard(filename)
    state[index_name] = sorted(files)
    _save(state)


def clear_files(index_name: str) -> None:
    """Remove all persisted filenames for *index_name*."""
    state = _load()
    state[index_name] = []
    _save(state)


def remove_index(index_name: str) -> None:
    """Remove the entire entry for a deleted index."""
    state = _load()
    state.pop(index_name, None)
    _save(state)
