# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor
from whetstone.core.module import BaseModule, BaseState


class Model(BaseModule, ABC):
    """Abstract base class for all language models."""
    @abstractmethod
    def generate(self, x: str, max_tokens: int = 100, **kwargs) -> str:
        """Generates text based on the input prompt.

        Args:
            x: The input prompt string. This might be a single prompt or a formatted conversation.
            max_tokens: The maximum number of new tokens to generate. Defaults to 100.
            **kwargs: Additional generation parameters specific to the model implementation.

        Returns:
            The generated text string.
        """
        pass

    def tokenize(self, x: str) -> Tensor:
        """Tokenizes a string into a tensor of token IDs.

        Args:
            x: The input string to tokenize.

        Returns:
            A Tensor containing the token IDs.
        
        Raises:
            NotImplementedError: If the model does not support tokenization.
        """
        raise NotImplementedError(f"Class {self.__class__.__name__} does not support tokenizing strings")

    def detokenize(self, x: Tensor) -> str | list[str]:
        """Detokenizes a tensor of token IDs back into a string or list of strings.

        Args:
            x: A Tensor of token IDs. Can be 1D (single sequence) or 2D (batch).

        Returns:
            The detokenized string, or a list of strings if the input was a batch.
        
        Raises:
            NotImplementedError: If the model does not support detokenization.
        """
        raise NotImplementedError(f"Class {self.__class__.__name__} does not support detokenizing tokens")

    def vocab_size(self) -> int:
        """Returns the size of the model's vocabulary.

        Returns:
            The vocabulary size as an integer.
        
        Raises:
            NotImplementedError: If the model does not have a defined vocabulary size.
        """
        raise NotImplementedError(f"Class {self.__class__.__name__} has no vocab size")
    
    def render_conversation_template(self, conversation: list[dict[str, str]]) -> str:
        """Renders a structured conversation into a single string suitable for the model's input.

        Args:
            conversation: A list of dictionaries, where each dictionary has 'role' and 'content' keys.
                          Example: [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there!"}]

        Returns:
            A formatted string representing the conversation.
        
        Raises:
            NotImplementedError: If the model does not support rendering conversation templates.
        """
        # TODO: We can provide a sensible default implementation for this
        raise NotImplementedError(f"Class {self.__class__.__name__} does not support rendering conversation templates")