# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor
import tiktoken
import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI
from whetstone.core.model import Model
from whetstone.core.module import BaseState, ModuleRegistry, StatefulModule
from whetstone.utils import JSONType

import logging
log = logging.getLogger(__name__)

class OpenAIModelState(BaseState):
    """State associated with an OpenAIModel instance."""
    _client: OpenAI = field(init=False)
    _encoder: Any = field(init=False)  # tiktoken encoder instance

    def __init__(self, instance: Any, persisted_data: JSONType | None = None):
        super().__init__(instance, persisted_data)
        self._client = OpenAI(**instance.client_parameters)
        try:
            self._encoder = tiktoken.encoding_for_model(instance.model)
        except KeyError:
            print(f"Warning: Encoding for model '{instance.model}' not found. Using 'cl100k_base'.")

@ModuleRegistry.register
class OpenAIModel(Model, StatefulModule[OpenAIModelState]):
    """Implements the Model interface using the OpenAI API.

    Uses the `openai` library to interact with models like GPT-3.5, GPT-4, etc.
    Handles state management for the OpenAI client and tokenizer.
    """
    model: str
    """The specific OpenAI model to use (e.g., 'gpt-3.5-turbo')."""
    client_parameters: dict = field(default_factory=dict)
    """Parameters to initialize the OpenAI client (e.g., api_key, base_url)."""
    generation_parameters: dict = field(default_factory=dict)
    """Default parameters for the chat completion API calls (e.g., temperature, top_p)."""

    def generate(self, x: str, max_tokens: int = 100, **kwargs) -> str:
        """Generates a response using the OpenAI Chat Completions API.
        
        Accepts either a plain string prompt (treated as a single user message)
        or a JSON string representing a list of message dictionaries conforming to the
        OpenAI API schema.

        Args:
            x: The input prompt string or a JSON string list of message dicts.
            max_tokens: The maximum number of tokens to generate in the completion.
                        This overrides the default if not already set in `generation_parameters`.
            **kwargs: Additional parameters passed directly to the `client.chat.completions.create` method.

        Returns:
            The generated content string from the model's response.

        Raises:
            ValueError: If the input `x` is not a string or a valid JSON representation
                        of the expected message list format.
        """
        messages: list[dict[str, str]]
        try:
            # Attempt to parse x as a JSON list of messages
            parsed_x = [json.loads(item) for item in x]
            if isinstance(parsed_x, list) and all(isinstance(item, dict) for item in parsed_x):
                messages = parsed_x
            else:
                # If x is valid JSON but not the expected format, treat as plain string
                messages = [{"role": "user", "content": x}]
        except json.JSONDecodeError:
            # If x is not valid JSON, treat as plain string
            messages = [{"role": "user", "content": x}]
        except TypeError:
            if isinstance(x, list) and all(isinstance(item, dict) for item in x):
                messages = x
            elif isinstance(x, str):
                messages = [{"role": "user", "content": x}]
            else:
                raise ValueError(f"Invalid input type: {type(x)}")
        
        if "max_tokens" not in self.generation_parameters and "max_completion_tokens" not in self.generation_parameters:
            self.generation_parameters["max_tokens"] = max_tokens

        # Combine generation parameters
        gen_params = {**self.generation_parameters, **kwargs}
        response = self.state._client.chat.completions.create(
            model=self.model,
            messages=messages,
            **gen_params
        )
        if hasattr(response, "choices") and len(response.choices) > 0:
            content = response.choices[0].message.content
            return content if content is not None else ""
        else:
            log.warning(f"No content found in response: {response}")
            return ""
        
    def tokenize(self, x: str) -> Tensor:
        """Tokenizes a string using the appropriate tiktoken encoder for the model.

        Args:
            x: The input string to tokenize.

        Returns:
            A 2D Tensor of shape (1, num_tokens) containing the token IDs.
        """
        token_ids = self.state._encoder.encode(x)
        return torch.tensor([token_ids])

    def detokenize(self, x: Tensor) -> str:
        """Detokenizes a tensor of token IDs using the tiktoken encoder.

        Expects a 1D or 2D Tensor. If 2D, it squeezes it first.

        Args:
            x: A Tensor containing token IDs.

        Returns:
            The detokenized string.
        """
        token_ids = x.cpu().squeeze().tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return self.state._encoder.decode(token_ids)

    def vocab_size(self) -> int:
        """Returns the vocabulary size of the tiktoken encoder."""
        return self.state._encoder.n_vocab

    def render_conversation_template(self, conversation: list[dict[str, str]]) -> str:
        """Renders a conversation list into a JSON string.
        
        The OpenAI API consumes the message list directly. This method serializes it
        to a JSON string to fit the `generate` method's expectation of a string input.

        Args:
            conversation: A list of message dictionaries (e.g., [{'role': 'user', 'content': '...'}]).

        Returns:
            A JSON string representation of the conversation list.
        """
        # OpenAI API takes the list directly, but to fit the LLM interface 
        # where generate expects a string, we serialize the conversation here.
        return json.dumps(conversation)
