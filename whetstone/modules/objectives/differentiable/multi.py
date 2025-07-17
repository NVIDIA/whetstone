# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import List, TypeVar

import torch
from whetstone.core.module import BaseModule, ModuleRegistry
from whetstone.core.objective import DifferentiableObjective, Objective, Sample
from whetstone.utils import JSONType

@dataclass
class WeightedObjective:
    weight: float
    objective: Objective

@ModuleRegistry.register
class MultiObjective(DifferentiableObjective, BaseModule):
    """A weighted combination of multiple objectives."""
    weighted_objectives: List[WeightedObjective]

    def __post_init__(self):
        self.weighted_objectives = [WeightedObjective(weight=wo["weight"], objective=wo["objective"]) for wo in self.weighted_objectives]

        if len(self.weighted_objectives) == 0:
            raise ValueError("weighted_objectives must contain at least one objective")

    def _evaluate_batch(self, inputs: List[str]) -> List[Sample]:
        all_samples = []
        for wo in self.weighted_objectives:
            samples = wo.objective.evaluate_batch(inputs)
            if wo.weight < 0:
                samples = [Sample(input=sample.input, score=1 - sample.score, output=sample.output) for sample in samples]
            all_samples.append(samples)
        
        # Combine results
        results = []
        for i in range(len(inputs)):
            total_score = 0.0
            total_weight = 0.0
            outputs = {}
            
            for j, wo in enumerate(self.weighted_objectives):
                sample = all_samples[j][i]
                total_score += abs(wo.weight) * sample.score
                total_weight += abs(wo.weight)
                outputs[j] = sample.output
            
            # Normalize by total weight
            normalized_score = total_score / total_weight if total_weight > 0 else 0.0
            
            results.append(Sample(
                input=inputs[i],
                score=normalized_score,
                output=outputs
            ))
        
        return results
    
    def __str__(self):
        s = "MultiObjective\n"
        for wo in self.weighted_objectives:
            s += f"Subobjective with weight {wo.weight}:\n{wo.objective}\n"
        return s
    
    def _gradient_batch(self, inputs: List[str]) -> List[tuple[Sample, torch.Tensor]]:
        # Get gradients from all differentiable objectives in parallel
        all_gradients = []
        for wo in self.weighted_objectives:
            if isinstance(wo.objective, DifferentiableObjective):
                gradients = wo.objective.gradient_batch(inputs)
                if wo.weight < 0:
                    gradients = [(Sample(input=gradient[0].input, score=1 - gradient[0].score, output=gradient[0].output), -gradient[1]) for gradient in gradients]
                all_gradients.append((wo, gradients))
        
        # Combine results
        results = []
        for i in range(len(inputs)):
            total_grad = None
            total_weight = 0.0
            outputs = {}
            
            # Process gradients from differentiable objectives
            for j, (wo, gradients) in enumerate(all_gradients):
                sample, grad = gradients[i]
                if total_grad is None:
                    total_grad = abs(wo.weight) * grad
                else:
                    total_grad += abs(wo.weight) * grad
                total_weight += abs(wo.weight)
                outputs[j] = sample.output
            
            # Normalize gradient by total weight
            if total_grad is not None and total_weight > 0:
                total_grad = total_grad / total_weight
            
            # Get scores from all objectives
            total_score = 0.0
            for j, wo in enumerate(self.weighted_objectives):
                sample = wo.objective.evaluate(inputs[i])
                total_score += abs(wo.weight) * sample.score
                outputs[j] = sample.output
            
            normalized_score = total_score / total_weight if total_weight > 0 else 0.0
            sample = Sample(input=inputs[i], score=normalized_score, output=outputs)
            
            results.append((sample, total_grad))
        
        return results 