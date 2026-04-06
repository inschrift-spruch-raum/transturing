"""Abstract executor backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from .isa import Instruction, Trace


class ExecutorBackend(ABC):
    """Abstract base class for all executor backends."""

    name: ClassVar[str]  # Backend identifier (e.g. 'numpy', 'torch')

    @abstractmethod
    def execute(self, prog: list[Instruction], max_steps: int = 50000) -> Trace:
        """Execute a program and return its full trace."""
        ...
