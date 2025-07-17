# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Dict, List, TypeVar
import random
import time
import logging

from whetstone.core.corpus import Corpus
from whetstone.core.module import BaseState, ModuleRegistry, StatefulModule
from whetstone.core.objective import Objective
from whetstone.core.optimizer import Iteration, Optimizer
from whetstone.utils import JSONType

log = logging.getLogger(__name__)

@dataclass
class OptimizerPerformance:
    """Track performance metrics for an optimizer."""
    total_samples: int = 0
    total_time: float = 0.0
    total_score_improvement: float = 0.0
    last_score: float | None = None

    def update(self, score: float, elapsed_time: float):
        if self.last_score is not None:
            improvement = max(0.0, self.last_score - score)  # Remember: lower score is better
            self.total_score_improvement += improvement
        self.last_score = score
        self.total_time += elapsed_time
        self.total_samples += 1

    @property
    def avg_improvement_per_second(self) -> float:
        if self.total_time == 0:
            return float('inf')
        return self.total_score_improvement / self.total_time

class MultiOptimizerState(BaseState):
    def __init__(self, instance: "MultiOptimizer", persisted_data: JSONType | None = None):
        super().__init__(instance, persisted_data)
        if persisted_data:
            self.optimizer_performance = {
                int(idx): OptimizerPerformance(
                    total_samples=perf["total_samples"],
                    total_time=perf["total_time"],
                    total_score_improvement=perf["total_score_improvement"],
                    last_score=perf["last_score"]
                )
                for idx, perf in persisted_data["optimizer_performance"].items()
            }
            self.last_iteration_time = persisted_data["last_iteration_time"]
        else:
            self.optimizer_performance: Dict[int, OptimizerPerformance] = dict()
            self.last_iteration_time: float = 0.0

    def persist(self) -> JSONType:
        return {
            "optimizer_performance": {
                str(idx): {
                    "total_samples": perf.total_samples,
                    "total_time": perf.total_time,
                    "total_score_improvement": perf.total_score_improvement,
                    "last_score": perf.last_score
                }
                for idx, perf in self.optimizer_performance.items()
            },
            "last_iteration_time": self.last_iteration_time
        }

@ModuleRegistry.register
class MultiOptimizer(Optimizer, StatefulModule[MultiOptimizerState]):
    """An optimizer that combines multiple optimizers using an epsilon-greedy strategy."""
    optimizers: List[Optimizer]
    epsilon: float = 0.3
    window_size: int = 5

    def step(self, objective: Objective, corpus: Corpus) -> Iteration:
        start_time = time.time()
        
        # Get current best score
        best_score = corpus.best_score if corpus.best_score is not None else float('inf')
        
        # Initialize performance tracking for new optimizers
        for idx in range(len(self.optimizers)):
            if idx not in self.state.optimizer_performance:
                self.state.optimizer_performance[idx] = OptimizerPerformance()
        
        # Choose optimizer using epsilon-greedy
        if random.random() < self.epsilon:
            # Exploration: choose random optimizer
            chosen_idx = random.randrange(len(self.optimizers))
            chosen_optimizer = self.optimizers[chosen_idx]
            log.info(f"Exploring with {chosen_optimizer.__class__.__name__}")
        else:
            # Exploitation: choose best performing optimizer
            chosen_idx = max(
                range(len(self.optimizers)),
                key=lambda idx: self.state.optimizer_performance[idx].avg_improvement_per_second
            )
            chosen_optimizer = self.optimizers[chosen_idx]
            log.info(f"Exploiting with {chosen_optimizer.__class__.__name__} at {self.state.optimizer_performance[chosen_idx].avg_improvement_per_second:.4f} avg loss drop per second")

        # Run the chosen optimizer
        iteration = chosen_optimizer.step(objective, corpus) or Iteration()
        
        # Update performance metrics
        elapsed_time = time.time() - start_time
        self.state.optimizer_performance[chosen_idx].update(
            best_score,
            elapsed_time
        )
        
        # Add metadata about optimizer choice and performance
        iteration.metadata.update({
            "chosen_optimizer_index": chosen_idx,
            "chosen_optimizer_class": chosen_optimizer.__class__.__name__,
            "optimizer_performance": {
                str(idx): {
                    "avg_improvement_per_second": perf.avg_improvement_per_second,
                    "total_samples": perf.total_samples
                }
                for idx, perf in self.state.optimizer_performance.items()
            }
        })
        
        return iteration 