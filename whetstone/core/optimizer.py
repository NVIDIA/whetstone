# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, Generic, Iterator, List, TypeVar
from pydantic import BaseModel, ConfigDict, Field

from whetstone.core.module import BaseState, StatefulModule
from whetstone.core.objective import DifferentiableObjective, Objective
from whetstone.utils import JSONType

from .corpus import Corpus


class Iteration(BaseModel):
    """A single iteration of an optimizer."""
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict = Field(default_factory=dict)

    def to_json(self) -> JSONType:
        return self.model_dump_json()
    
    @classmethod
    def from_json(cls, data: JSONType):
        return cls.model_validate_json(data)


class Optimizer[StateT: BaseState](StatefulModule[StateT], ABC):
    """Base class for optimizers.
    
    Optimizers generate new inputs based on previous evaluation results
    stored in a corpus. They may maintain internal state that can be
    serialized.
    """
    @abstractmethod
    def step(self, objective: Objective, corpus: Corpus) -> Iteration | None:
        """Generate a new iteration of the optimizer."""
        raise NotImplementedError

class GradientOptimizer[StateT: BaseState](Optimizer[StateT], ABC):
    """Base class for optimizers that use gradients.
    """
    @abstractmethod
    def step(self, objective: DifferentiableObjective, corpus: Corpus) -> Iteration | None:
        """Generate a new iteration of the optimizer."""
        raise NotImplementedError
