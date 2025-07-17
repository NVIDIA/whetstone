# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass
from typing import Optional, Any

import torch
import transformers
from torch import Tensor
from transformers import PreTrainedTokenizer

from whetstone.core import Corpus, ModuleRegistry, Optimizer
from whetstone.core import DifferentiableObjective
from whetstone.core.model import Model
from whetstone.core.optimizer import GradientOptimizer, Iteration

log = logging.getLogger(__name__)

@ModuleRegistry.register
class GCG(GradientOptimizer):
    model: Model

    search_width: int = 200
    topk: int = 256
    n_replace: int = 1
    filter_ids: bool = True
    add_space_before_target: bool = False
    not_allowed_ids: Optional[list[int]] = None

    def step(self, objective: DifferentiableObjective, corpus: Corpus) -> Iteration | None:
        input_str = next(corpus)

        # Compute the token gradient
        optim_ids_onehot_grad = objective.gradient(input_str)[1]

        with torch.no_grad():
            input_ids = self.model.tokenize(input_str)
            # Sample candidate token sequences based on the token gradient
            sampled_ids = sample_ids_from_grad(
                input_ids,
                optim_ids_onehot_grad.squeeze(0),
                self.search_width,
                self.topk,
                self.n_replace,
                not_allowed_ids=self.not_allowed_ids,
            )

            if self.filter_ids:
                filtered = self._filter_ids(sampled_ids)
                if filtered is not None:
                    sampled_ids = filtered

            # Evaluate the sampled token sequences
            sampled_strs = self.model.detokenize(sampled_ids)
            objective.evaluate_batch(sampled_strs)


    def _filter_ids(self, ids: Tensor):
        """Filters out sequeneces of token ids that change after retokenization.

        Args:
            ids : Tensor, shape = (search_width, n_optim_ids)
                token ids

        Returns:
            filtered_ids : Tensor, shape = (new_search_width, n_optim_ids)
                all token ids that are the same after retokenization
        """
        ids_decoded = self.model.detokenize(ids)
        filtered_ids = []

        for i in range(len(ids_decoded)):
            # Retokenize the decoded token ids
            ids_encoded = self.model.tokenize(ids_decoded[i], add_special_tokens=False)
            if torch.equal(ids[i], ids_encoded):
                filtered_ids.append(ids[i])

        if not filtered_ids:
            # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
            log.warning(
                "No token sequences are the same after decoding and re-encoding. "
                "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
            )
            return None

        return torch.stack(filtered_ids)


def sample_ids_from_grad(
        ids: Tensor,
        grad: Tensor,
        search_width: int,
        topk: int = 256,
        n_replace: int = 1,
        not_allowed_ids: Tensor = False,
):
    """Returns `search_width` combinations of token ids based on the token gradient.

    Args:
        ids : Tensor, shape = (n_optim_ids)
            the sequence of token ids that are being optimized
        grad : Tensor, shape = (n_optim_ids, vocab_size)
            the gradient of the GCG loss computed with respect to the one-hot token embeddings
        search_width : int
            the number of candidate sequences to return
        topk : int
            the topk to be used when sampling from the gradient
        n_replace: int
            the number of token positions to update per sequence
        not_allowed_ids: Tensor, shape = (n_ids)
            the token ids that should not be used in optimization

    Returns:
        sampled_ids : Tensor, shape = (search_width, n_optim_ids)
            sampled token ids
    """
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1).to(grad.device)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices

    sampled_ids_pos = torch.argsort(torch.rand((search_width, n_optim_tokens), device=grad.device))[..., :n_replace]
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device)
    ).squeeze(2)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

    return new_ids
