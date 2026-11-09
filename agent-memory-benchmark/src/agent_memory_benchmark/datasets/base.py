"""Dataset adapter protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agent_memory_benchmark.datasets.models import DatasetExample


class DatasetAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable dataset name."""

    @abstractmethod
    def load(self) -> list[DatasetExample]:
        """Load typed benchmark examples."""
