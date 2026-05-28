"""Tests for the on-disk layout types.

Pure data types: `Checkpoint` (per-checkpoint `ckpt-NNNNN.json`)
and `SampleManifest` (per-sample `sample.json`).
Verifies the contract: required fields, validation strictness on
triggers, JSON round-trip, and ``extra="allow"`` forward-compat.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from inspect_ai.util._checkpoint._layout.schemas import (
    Checkpoint,
    SampleManifest,
    SolverDone,
)


def _info(
    snapshot_id: str, size_bytes: int = 0, duration_ms: int = 0
) -> dict[str, object]:
    return {
        "snapshot_id": snapshot_id,
        "size_bytes": size_bytes,
        "duration_ms": duration_ms,
    }


_BASE_CHECKPOINT = {
    "checkpoint_id": 1,
    "trigger": "time",
    "turn": 42,
    "created_at": "2026-04-26T14:23:11Z",
    "duration_ms": 842,
    "size_bytes": 1834291,
    "host": _info("abc123"),
    "sandboxes": {
        "default": _info("def456"),
        "tools": _info("ghi789"),
    },
}

_BASE_SAMPLE = {
    "restic_password": "s3cr3t",
}


# --- Checkpoint -----------------------------------------------------------


def test_checkpoint_basic_round_trip() -> None:
    checkpoint = Checkpoint.model_validate(_BASE_CHECKPOINT)
    assert checkpoint.checkpoint_id == 1
    assert checkpoint.trigger == "time"
    assert checkpoint.created_at == datetime(
        2026, 4, 26, 14, 23, 11, tzinfo=timezone.utc
    )
    assert checkpoint.host.snapshot_id == "abc123"
    assert {name: info.snapshot_id for name, info in checkpoint.sandboxes.items()} == {
        "default": "def456",
        "tools": "ghi789",
    }

    # JSON round-trip preserves all fields.
    rehydrated = Checkpoint.model_validate_json(checkpoint.model_dump_json())
    assert rehydrated == checkpoint


@pytest.mark.parametrize("trigger", ["time", "turn", "manual"])
def test_checkpoint_accepts_all_documented_triggers(trigger: str) -> None:
    payload = {**_BASE_CHECKPOINT, "trigger": trigger}
    checkpoint = Checkpoint.model_validate(payload)
    assert checkpoint.trigger == trigger


def test_checkpoint_rejects_unknown_trigger() -> None:
    payload = {**_BASE_CHECKPOINT, "trigger": "bogus"}
    with pytest.raises(ValidationError):
        Checkpoint.model_validate(payload)


def test_checkpoint_extra_fields_preserved_round_trip() -> None:
    """Forward-compat for unknown fields.

    A future inspect adds a field; older inspect must keep that field
    intact when round-tripping the JSON.
    """
    payload = {**_BASE_CHECKPOINT, "future_field": {"nested": [1, 2, 3]}}
    checkpoint = Checkpoint.model_validate(payload)
    dumped = json.loads(checkpoint.model_dump_json())
    assert dumped["future_field"] == {"nested": [1, 2, 3]}


def test_checkpoint_empty_sandboxes_is_valid() -> None:
    """Host-only checkpointing case: no sandbox snapshots."""
    payload = {**_BASE_CHECKPOINT, "sandboxes": {}}
    checkpoint = Checkpoint.model_validate(payload)
    assert checkpoint.sandboxes == {}


# --- SampleManifest ---------------------------------------------------


def test_manifest_basic_round_trip() -> None:
    manifest = SampleManifest.model_validate(_BASE_SAMPLE)
    assert manifest.restic_password == "s3cr3t"
    assert manifest.solver_done is None

    rehydrated = SampleManifest.model_validate_json(manifest.model_dump_json())
    assert rehydrated == manifest


def test_manifest_requires_password() -> None:
    with pytest.raises(ValidationError):
        SampleManifest.model_validate({})


def test_manifest_extra_fields_preserved_round_trip() -> None:
    payload = {**_BASE_SAMPLE, "future_field": "later"}
    manifest = SampleManifest.model_validate(payload)
    dumped = json.loads(manifest.model_dump_json())
    assert dumped["future_field"] == "later"


def test_manifest_with_solver_done() -> None:
    payload = {
        **_BASE_SAMPLE,
        "solver_done": {
            "checkpoint_id": 7,
            "created_at": "2026-04-26T14:23:11Z",
        },
    }
    manifest = SampleManifest.model_validate(payload)
    assert isinstance(manifest.solver_done, SolverDone)
    assert manifest.solver_done.checkpoint_id == 7
    assert manifest.solver_done.created_at == datetime(
        2026, 4, 26, 14, 23, 11, tzinfo=timezone.utc
    )

    rehydrated = SampleManifest.model_validate_json(manifest.model_dump_json())
    assert rehydrated == manifest
