# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
import string
from typing import List, Optional

from whetstone.core.corpus import Corpus
from whetstone.core.module import ModuleRegistry
from whetstone.core.objective import Objective
from whetstone.core.optimizer import Iteration, Optimizer


@ModuleRegistry.register
class RandomOptimizer(Optimizer):
    """
    A random optimizer that replaces n characters of an input
    with random characters sampled from a configurable charset.

    Replacement can occur either at the end of the string (suffix mode)
    or at random positions within the string.
    """
    num_chars_to_replace: int = 5
    charset: str = string.ascii_letters + string.digits
    replace_mode: str = "suffix"  # Options: "suffix", "random_positions"
    
    def step(self, objective: Objective, corpus: Corpus) -> Iteration | None:
        # If the corpus is empty, create a random input
        if len(corpus) == 0:
            input_str = "".join(random.choices(self.charset, k=10))
        else:
            input_str = next(corpus)
            
            # Get the length of the input
            input_len = len(input_str)
            
            # Determine how many characters to replace
            n = min(self.num_chars_to_replace, input_len)
            
            if n > 0:
                if self.replace_mode == "suffix":
                    # Keep the first (input_len - n) characters and replace the last n characters
                    prefix = input_str[:-n] if n < input_len else ""
                    suffix = "".join(random.choices(self.charset, k=n))
                    input_str = prefix + suffix
                elif self.replace_mode == "random_positions":
                    # Convert string to list for mutable operations
                    input_list = list(input_str)
                    
                    # Choose n random indices to replace
                    indices_to_replace = random.sample(range(input_len), k=n)
                    
                    # Replace characters at chosen indices
                    for i in indices_to_replace:
                        input_list[i] = random.choice(self.charset)
                    
                    # Convert back to string
                    input_str = "".join(input_list)
                else:
                    raise ValueError(f"Unknown replace_mode: {self.replace_mode}")
        
        # Evaluate the new input
        sample = objective.evaluate(input_str)
        corpus.add(sample)
        
        return Iteration()
