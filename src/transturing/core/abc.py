"""Abstract executor backend interface."""

from abc import ABC, abstractmethod

from .isa import Instruction, Trace


class ExecutorBackend(ABC):
    """Abstract base class for all executor backends.

    Both NumPyExecutor and TorchExecutor implement this interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier, e.g. 'numpy' or 'torch'."""
        ...

    @abstractmethod
    def execute(self, prog: list[Instruction], max_steps: int = 50000) -> Trace:
        """Execute a program and return its full trace."""
        ...
