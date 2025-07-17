# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass, field
from typing import List, TypeVar

import torch
from torch import Tensor
from transformers import PreTrainedTokenizer

from whetstone.core.module import ModuleRegistry
from whetstone.core.objective import Objective, Sample

log = logging.getLogger(__name__)

@ModuleRegistry.register
class LengthObjective(Objective):
    """
    Objective to minimize the string length.
    Score is the raw string length (lower is better).
    NOTE: Score is not normalized to [0, 1].
    """
    def _evaluate_batch(self, inputs: List[str]) -> List[Sample]:
        
        all_samples = []
        for text in inputs:
            length = float(len(text))

            def norm(length):
                # [0, inf] -> [0, 1], with most values between 0 and 50 filling most of the range (empirical)
                return max(min(1, length**0.5/(1+length**0.5)), 0)

            all_samples.append(Sample(text, norm(length)))
            
        return all_samples 