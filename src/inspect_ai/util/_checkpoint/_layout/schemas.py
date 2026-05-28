"""Pydantic models for the on-disk checkpoint layout.

Defines the shape of the per-sample ``sample.json`` manifest and the
per-checkpoint ``ckpt-NNNNN.json`` checkpoint files. See
``design/plans/checkpointing-working.md`` §1 for the full layout
description. These are pure data types — read/write helpers live with
the Phase 3 write code.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from .._triggers import CheckpointTriggerKind


class SnapshotDetails(BaseModel):
    """Per-backup stats captured in the checkpoint file.

    One per repo (host repo + one per active sandbox repo). Values come
    from restic's backup summary — see :class:`ResticBackupSummary`.
    """

    model_config = ConfigDict(extra="allow")

    snapshot_id: str
    """Restic snapshot id for this backup."""

    size_bytes: int
    """Bytes this snapshot added to its repo, after compression
    (restic's ``data_added_packed``)."""

    duration_ms: int
    """How long the restic invocation took, in milliseconds."""


class Checkpoint(BaseModel):
    """Per-checkpoint metadata file (``<attempt>/ckpt-NNNNN.json``).

    Written atomically at each successful checkpoint. This file's
    existence is the commit point — the checkpoint is visible to
    resume only when this file is in place. See §1 and §4d.
    """

    model_config = ConfigDict(extra="allow")

    checkpoint_id: int
    """Ordinal integer (1, 2, 3, …) chosen by inspect at write time."""

    trigger: CheckpointTriggerKind
    """The policy that fired this checkpoint."""

    turn: int
    """Agent turn index at which this checkpoint was taken."""

    created_at: datetime
    """When the checkpoint was committed."""

    duration_ms: int
    """How long the checkpoint cycle took, in milliseconds."""

    size_bytes: int
    """Total on-disk size added by this checkpoint (sum of host + sandboxes)."""

    host: SnapshotDetails
    """Stats for the host repo backup this cycle."""

    sandboxes: dict[str, SnapshotDetails] = Field(default_factory=dict)
    """Per-sandbox stats keyed by sandbox name. Empty when checkpointing is
    host-only."""


class SolverDone(BaseModel):
    """Marker that the solver/agent finished and scoring is the next thing.

    Set on :class:`SampleManifest` by ``_CheckpointerSetup.__aexit__``
    after a successful final fire on clean agent exit. Its presence on
    the on-disk manifest is the "skip the agent loop on retry" signal —
    the sample source reads it and tags
    :class:`ResumeCheckpoint.attempt` as
    :attr:`Attempt.RETRY_FOR_SCORING`.
    """

    model_config = ConfigDict(extra="allow")

    checkpoint_id: int
    """Ordinal of the final checkpoint this marker rides on top of."""

    created_at: datetime
    """When the agent returned and the marker was written."""


class SampleManifest(BaseModel):
    """Per-sample manifest file (``<sample-root>/sample.json``).

    Single per-sample state file that carries everything the sample's
    checkpoint subtree needs in addition to the restic repos themselves:
    the restic password (written once at first checkpoint setup) and an
    optional :class:`SolverDone` marker (written on clean agent exit).

    Preserved across retries of the same sample via the FS copy at
    resume — so the same password unlocks the FS-copied ``host/`` and
    ``sandboxes/<name>/`` repos, and ``solver_done`` rides forward
    automatically.
    """

    model_config = ConfigDict(extra="allow")

    restic_password: str
    """Password used by every repo (host + each sandbox) under this
    sample. See ``design/plans/checkpointing-hydration.md`` for how it
    reaches sandbox-side restic without being persisted in the sandbox."""

    solver_done: SolverDone | None = None
    """Present iff the agent completed cleanly and the harness-driven
    final fire succeeded. Drives scoring-phase resume."""
