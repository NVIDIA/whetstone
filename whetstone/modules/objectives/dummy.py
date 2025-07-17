# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
import random
import time
from functools import lru_cache
from typing import Any, List

import torch
from whetstone.core.model import Model
from whetstone.core.module import BaseState, ModuleRegistry, StatefulModule
from whetstone.core.objective import DifferentiableObjective, Objective, Sample
from whetstone.utils import JSONType


@dataclass
class DummyObjectiveState(BaseState):
    def persist(self) -> JSONType:
        return {
            "score": self.score
        }
    
    def __init__(self, instance: "DummyObjective", data: JSONType | None = None):
        super().__init__(instance)
        if data:
            self.score = data["score"]
        else:
            self.score = 1.0

@ModuleRegistry.register
class DummyObjective(DifferentiableObjective, StatefulModule[DummyObjectiveState]):
    model: Model
    jitter: float = 0.1

    def _evaluate_batch(self, inputs: List[str]) -> List[Sample]:
        return [self._evaluate(input) for input in inputs]
    
    def _evaluate(self, input: str) -> Sample:
        self.state.score *= 0.99
        time.sleep(0.1)
        # We add some random jitter to the score
        score = self.state.score + random.uniform(-self.jitter, self.jitter)
        # and trunc it to 0-1
        score = max(0.0, min(1.0, score))

        return Sample(input=input, score=score, output={"output": self.model.generate(input)})
    
    def _gradient_batch(self, inputs: List[str]) -> List[tuple[Sample, torch.Tensor]]:
        return [
            (self.evaluate(input), torch.randn(len(input), self.model.vocab_size()))
            for input in inputs
        ]