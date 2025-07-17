# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Set, Type, TypeVar

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field

from whetstone.core.module import BaseModule, BaseState, ModuleRegistry, StatefulModule
from whetstone.utils import JSONType

from .constraint import Constraint
from .corpus import Corpus
from .objective import Objective, Sample
from .optimizer import Optimizer, Iteration

class JobState(BaseState):
    def persist(self) -> JSONType:
        return {
            "started_at": self.started_at.isoformat(),
            "total_samples": self.total_samples,
            "iterations": [iteration.to_json() for iteration in self.iterations]
        }
    
    def __init__(self, instance: "Job", persisted_data: JSONType | None = None) -> "JobState":
        super().__init__(instance)
        if persisted_data:
            self.started_at = datetime.fromisoformat(persisted_data["started_at"])
            self.total_samples = persisted_data["total_samples"]
            self.iterations = [Iteration.from_json(iteration) for iteration in persisted_data["iterations"]]
        else:
            self.started_at = datetime.now()
            self.total_samples = 0
            self.iterations = []


@ModuleRegistry.register
class Job(StatefulModule[JobState]):
    """
    Represents a single run of a target triple (objective, optimizer, corpus).

    This is also where you can configure the termination criterion (number of iterations, time, etc.)
    """
    @ModuleRegistry.register(make_dataclass=False)
    class Target(BaseModule):
        objective: Objective
        optimizer: Optimizer
        corpus: Corpus | None = None

        def __init__(self, **kwargs):
            # We allow defining other objects or dependencies in the target, but only these three will be used
            # by higher level functionality.
            self.objective = kwargs.pop("objective")
            self.optimizer = kwargs.pop("optimizer")
            self.corpus = kwargs.pop("corpus")

    
    target: Target
    iterations: int = 100
    state_location: str = "state.json"
    save_interval: int = 10

    @property
    def completed(self) -> bool:
        """
        Whether the job has completed.
        """
        return len(self.state.iterations) >= self.iterations


    def run(self, sample_callback: Callable[[List[Sample]], None] | None = None) -> Iterator[Iteration]:
        """
        Run the job.
        """
        if self.target.corpus is None:
            self.target.corpus = self.target.initial_corpus

        def callback(samples: List[Sample]):
            """
            Callback to be called when an input is evaluated.
            """
            for sample in samples:
                self.target.corpus.add(sample)
                self.state.total_samples += 1

            if sample_callback is not None:
                sample_callback(samples)

        self.target.objective._register_callback(callback)

        while not self.completed:
            iteration = self.target.optimizer.step(self.target.objective, self.target.corpus) or Iteration()
            self.state.iterations.append(iteration)
            yield iteration
    
    def save_state(self):
        """Save the state of the job to the state file and also the corpus."""
        temp_file = f"{self.state_location}.tmp"
        try:
            with open(temp_file, "w") as f:
                json.dump(
                    ModuleRegistry.suspend_state(),
                    f,
                    indent=4
                )
            # If successful, move the temp file to the actual location
            os.replace(temp_file, self.state_location)
        except Exception as e:
            # Clean up the temp file if there was an error
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise e

        self.target.corpus.save()
