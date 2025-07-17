# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from functools import lru_cache
import json
import logging
import os
import random
from typing import Dict, Generic, Iterator, List, Set, TypeVar
import typing

from whetstone.core.module import BaseModule, BaseState, ModuleRegistry, StatefulModule
from whetstone.utils import JSONType


from .constraint import Constraint
from .objective import Objective, Sample

log = logging.getLogger(__name__)


class CorpusState(BaseState):
    def __init__(self, instance: "Corpus", persisted_data: JSONType | None = None):
        super().__init__(instance, persisted_data)
        self.samples: List[Sample] = []
        self.input_map: Dict[str, Sample] = {}
    

@ModuleRegistry.register
class Corpus(StatefulModule[CorpusState]):
    """A collection of the best inputs and their evaluation results.
    
    The corpus maintains a set of unique inputs and their corresponding
    evaluation results. All inputs must satisfy the corpus's constraints.
    This default corpus will only maintain the top-n samples by objective value.

    Depending on whether the objective provides additional output metadata, the corpus
    can be configured to only keep the best sample for each unique objective output.
    This way a more diverse input corpus can be maintained over the course of the optimization.

    The corpus implementation also provides a simple epsilon-greedy sampling strategy
    to balance exploration and exploitation.
    """
    objective: Objective
    
    initial_inputs: List[str] = field(default_factory=list("placeholder"))
    constraints: List[Constraint] = field(default_factory=list)
    max_samples: int = 100
    corpus_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "corpus"))
    epsilon: float = 0.3
    collapse_objective_outputs: bool = False # Keeps only the best input with a specific output dict

    def __post_init__(self):
        # Load existing samples from the corpus directory
        if os.path.exists(self.corpus_dir):
            self.state.samples = []
            for f in os.listdir(self.corpus_dir):
                with open(os.path.join(self.corpus_dir, f), "r") as file:
                    self.add(Sample.from_json(json.loads(file.read())))
        
        if len(self.state.samples) == 0:
            initial_inputs = self.initial_inputs
            self.add_many(self.objective.evaluate_batch(initial_inputs))

    def save(self):
        """Save the corpus to the corpus directory."""
        if not os.path.exists(self.corpus_dir):
            os.makedirs(self.corpus_dir)
        
        # Clear existing files
        for file in os.listdir(self.corpus_dir):
            os.remove(os.path.join(self.corpus_dir, file))
        
        # Save each input with a filename based on its score
        for sample in self.state.samples:
            score = sample.score
            
            # Format score to 4 digits (e.g., 0.001 -> 0001)
            score_str = str(int(score * 1000)).zfill(4)
            
            # Create a unique filename
            import hashlib
            hash_obj = hashlib.sha256(str(sample).encode())
            unique_id = hash_obj.hexdigest()[:5]
            filename = f"{score_str}_{unique_id}.json"
            filepath = os.path.join(self.corpus_dir, filename)
            
            # Write the input to the file
            with open(filepath, "w") as f:
                f.write(json.dumps(sample.to_json(), indent=4))

    def validate_input(self, input: str) -> bool:
        """Check if the input satisfies the constraints."""
        return all(constraint(input) for constraint in self.constraints)

    def add(self, item: Sample | str) -> bool:
        """Add a sample to the corpus.
        
        Returns True if the sample was added.
        Raises ValueError if the sample violates a constraint.
        """

        if isinstance(item, Sample):
            sample = item
        else:
            if self.objective is None:
                raise ValueError(f"Objective is not set, need to evaluate input {item}")

            sample = self.objective.evaluate(item)
        
        if sample.input in self.state.input_map:
            return False

        if not self.validate_input(sample.input):
            log.warning(f"Sample {sample.input} violates constraints")
            return False
        
        # If we already have an input that generates the same output with a lower loss, don't add this one, otherwise replace it
        if sample.output and len(sample.output) > 0 and self.collapse_objective_outputs:
            for existing_sample in self.state.samples:
                if existing_sample.output == sample.output:
                    if existing_sample.score > sample.score:
                        self.state.samples[self.state.samples.index(existing_sample)] = sample
                        self.state.input_map[sample.input] = sample
                        return True
        
        if len(self.state.samples) < self.max_samples:
            self.state.samples.append(sample)
            self.state.input_map[sample.input] = sample
            return True
        
        # Find the worst sample
        if self.worst_sample.score > sample.score:
            self.state.samples[self.state.samples.index(self.worst_sample)] = sample
            self.state.input_map[sample.input] = sample
            return True

        return False

    def add_many(self, samples: List[Sample]) -> List[bool]:
        """Add multiple samples to the corpus.
        
        Returns a list of booleans indicating whether each sample was added.
        """
        return [self.add(sample) for sample in samples]
    
    def __next__(self) -> str:
        """Let the corpus suggest the next input to optimize.
        
        Override this function in a subclass to implement a custom corpus sampling strategy.
        """
        # With probability epsilon, return a random sample
        if random.random() < self.epsilon:
            return random.choice(self.state.samples).input

        return self.best_input
    

    @property
    def worst_sample(self) -> Sample | None:
        """The worst sample in the corpus."""
        return max(self.state.samples, key=lambda x: x.score) if self.state.samples else None
    
    @property 
    def best_sample(self) -> Sample | None:
        """The best sample in the corpus."""
        return min(self.state.samples, key=lambda x: x.score) if self.state.samples else None
    
    @property
    def best_input(self) -> str | None:   
        """The best input in the corpus."""
        return self.best_sample.input if self.best_sample else None
    
    @property
    def best_score(self) -> float | None:
        """The best score in the corpus."""
        return self.best_sample.score if self.best_sample else None
    
    @property
    def inputs(self) -> List[str]:
        """The inputs in the corpus."""
        return [sample.input for sample in self.state.samples]

    def __len__(self) -> int:
        """The number of samples in the corpus."""
        return len(self.state.samples)