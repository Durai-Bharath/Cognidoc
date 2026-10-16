"""Append-only JSONL queue for items needing human review.

Every "flag for manual review / verification / inspection" branch writes here.
Writing never raises into the pipeline: an I/O failure is logged and swallowed so
the review path can never block extraction.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from common.logging import get_logger

_log = get_logger(__name__)


class ReviewItem(BaseModel):
    review_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    document_id: str | None = None
    agent: str
    reason: str
    item_ref: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ReviewQueue:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._items: list[ReviewItem] = []

    @property
    def items(self) -> list[ReviewItem]:
        """Items flagged through this queue instance (in-memory view)."""
        return list(self._items)

    def flag(
        self,
        *,
        agent: str,
        reason: str,
        document_id: str | None = None,
        item_ref: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> ReviewItem:
        item = ReviewItem(
            document_id=document_id,
            agent=agent,
            reason=reason,
            item_ref=item_ref,
            details=details or {},
        )
        with self._lock:
            self._items.append(item)
            if self.path is not None:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a", encoding="utf-8") as fh:
                        fh.write(item.model_dump_json() + "\n")
                except OSError:
                    _log.exception("review_queue_write_failed", extra={"path": str(self.path)})
        _log.info(
            "flagged_for_review",
            extra={"agent": agent, "reason": reason, "item_ref": item_ref, "doc": document_id},
        )
        return item

    def read_all(self) -> list[ReviewItem]:
        """Read every item persisted at :attr:`path` (all runs, not just this instance)."""
        if self.path is None or not self.path.exists():
            return self.items
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [ReviewItem.model_validate(json.loads(line)) for line in lines if line.strip()]
