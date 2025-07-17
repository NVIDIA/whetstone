# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, Generic, List, TypeVar
from pydantic import BaseModel, ConfigDict, Field
import torch

from whetstone.core.module import BaseModule, ModuleRegistry
from whetstone.utils import JSONType

@dataclass
class Sample:
    """Result of evaluating an objective function."""
    input: str
    score: float  # Normalized score in [0, 1], where 0 means objective achieved
    output: dict | None = field(default=None)

    def to_json(self) -> JSONType:
        return {
            "input": self.input,
            "score": self.score,
            "output": self.output
        }
    
    @classmethod
    def from_json(cls, data: JSONType) -> "Sample":
        return cls(
            input=data["input"],
            score=data["score"],
            output=data["output"]
        )

@dataclass
class Objective(BaseModule, ABC):
    """Base class for objective functions.
    
    Objectives evaluate inputs and return a normalized score in [0, 1],
    where 0 indicates the objective has been achieved. They may also return
    additional metadata about the evaluation.

    Objective results are always cached by default, as it is assumed that objective eval is expensive but also deterministic.
    """
    _callbacks: list[tuple[Callable[[list[Sample]], None], bool]] = field(default_factory=list, init=False, repr=False, hash=False, compare=False)
    _cache: dict[str, Sample] = field(default_factory=dict, init=False, repr=False, hash=False, compare=False)

    @abstractmethod
    def _evaluate_batch(self, inputs: List[str]) -> List[Sample]:
        pass
    
    def evaluate_batch(self, inputs: List[str]) -> List[Sample]:
        """Evaluate a batch of inputs and return their scores and metadata."""
        old_results = []

        inputs_uncached = []
        for input in inputs:
            if input in self._cache:
                old_results.append(self._cache[input])
            else:
                inputs_uncached.append(input)

        new_results = []
        if inputs_uncached:
            new_results.extend(self._evaluate_batch(inputs_uncached))

            for result in new_results:
                self._cache[result.input] = result

         
        all_results = new_results + old_results

        for callback, call_on_cached in self._callbacks:
            if call_on_cached:
                callback(all_results)
            elif new_results:
                callback(new_results)
        
        return all_results
    
    def evaluate(self, input_value: str) -> Sample:
        """Evaluate a single input (convenience method)."""
        result = self.evaluate_batch([input_value])
        if len(result) == 0:
            raise ValueError(f"evaluate_batch impl of {self.__class__.__name__} returned no results for input {input_value}")
        return result[0]
    
    def is_achieved(self, result: Sample, threshold: float = 0.0) -> bool:
        """Check if the objective has been achieved based on a result."""
        return result.score <= threshold
    
    def _register_callback(self, callback: Callable[[list[Sample]], None], call_on_cached: bool = False):
        """Install a callback to be called when an input is evaluated."""
        self._callbacks.append((callback, call_on_cached))

    
@dataclass
class DifferentiableObjective(Objective, ABC):
    """An objective that is differentiable with respect to the input token representation and model used."""
    _gradient_cache: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False, hash=False, compare=False)
    
    @abstractmethod
    def _gradient_batch(self, inputs: List[str]) -> List[tuple[Sample, torch.Tensor]]:
        """Compute the gradient of the objective with respect to the input.
        """
        pass

    def _evaluate_batch(self, inputs: List[str]) -> List[Sample]:
        """Default implementation of evaluate_batch that uses gradient_batch."""
        return [sample for sample, _ in self._gradient_batch(inputs)]

    def gradient_batch(self, inputs: List[str]) -> List[tuple[Sample, torch.Tensor]]:
        """Compute the gradient of the objective with respect to the input."""
        old_results = []

        inputs_uncached = []
        for input in inputs:
            if input in self._cache and input in self._gradient_cache:
                old_results.append((self._cache[input], self._gradient_cache[input]))
            else:
                inputs_uncached.append(input)

        new_results = []
        if inputs_uncached:
            new_results.extend(self._gradient_batch(inputs_uncached))

            for result in new_results:
                self._cache[result[0].input] = result[0]
                self._gradient_cache[result[0].input] = result[1]

        all_results = new_results + old_results

        for callback, call_on_cached in self._callbacks:
            if call_on_cached:
                callback([sample for sample, _ in all_results])
            elif new_results:
                callback([sample for sample, _ in new_results])

        return all_results

    def gradient(self, input_value: str) -> tuple[Sample, torch.Tensor]:
        """Compute the gradient of the objective with respect to the input."""
        return self.gradient_batch([input_value])[0]
