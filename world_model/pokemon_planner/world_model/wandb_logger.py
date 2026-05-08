"""W&B logger wrapper — no-op when WANDB_API_KEY env var unset.

Per spec Section 6.6. Trainer code calls logger.log(metrics, step) without
caring whether wandb is configured; this class handles the conditional.
"""
from __future__ import annotations

import os
from typing import Any


class WandbLogger:
    """Conditional W&B logger — silent when WANDB_API_KEY unset."""

    def __init__(
        self,
        project: str,
        run_name: str,
        config: dict[str, Any],
        run_id: str | None = None,
    ):
        if not os.environ.get("WANDB_API_KEY"):
            self.enabled = False
            self.run = None
            return

        import wandb
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            id=run_id,
            resume="allow" if run_id else None,
        )
        self.enabled = True

    @property
    def run_id(self) -> str | None:
        if not self.enabled or self.run is None:
            return None
        return getattr(self.run, "id", None)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if not self.enabled or self.run is None:
            return
        self.run.log(metrics, step=step)

    def log_artifact(self, path: str, name: str, kind: str = "checkpoint") -> None:
        if not self.enabled or self.run is None:
            return
        import wandb
        artifact = wandb.Artifact(name=name, type=kind)
        artifact.add_file(path)
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        if not self.enabled or self.run is None:
            return
        self.run.finish()
