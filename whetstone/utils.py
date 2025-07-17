# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
import functools
import gc
import inspect
import logging
import math
import random
import string
from importlib import import_module
from typing import Dict, Callable, Any, Type

from pydantic import BaseModel, ConfigDict, ValidationError
import torch
from omegaconf import DictConfig, ListConfig
from omegaconf.basecontainer import BaseContainer
from torch import Tensor

INIT_CHARS = [
    ".", ",", "!", "?", ";", ":", "(", ")", "[", "]", "{", "}",
    "@", "#", "$", "%", "&", "*",
    "w", "x", "y", "z",
]

logger = logging.getLogger(__name__)

JSONType = dict[str, "JSONType"] | list["JSONType"] | str | int | float | bool
JSONDict = dict[str, JSONType]

class AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value


def instantiate(cfg: DictConfig, obj_global_cache: Dict = None):
    """
    DANGEROUS: Untrusted cfg can lead to arbitrary code execution.

    Instantiates all/this component of a config, using dependency injection.
    Requires dependencies to be in a DAG.
    Allows you to define things like the model once and then reference them repeatedly.
    """
    if obj_global_cache is None:
        obj_global_cache = dict()

    if not isinstance(cfg, BaseContainer):
        return cfg

    target = cfg.get("_target_", None)

    local_objs = AttrDict()
    for key, sub_cfg in cfg.items():
        if key.startswith("_") and key.endswith("_"):
            continue

        if instance := obj_global_cache.get(str(sub_cfg)):
            local_objs[key] = instance
        elif isinstance(sub_cfg, DictConfig):
            instance = instantiate(sub_cfg, obj_global_cache)

            if not isinstance(instance, BaseContainer):
                obj_global_cache[str(sub_cfg)] = instance
                local_objs[key] = instance
        elif isinstance(sub_cfg, ListConfig):
            instances = []
            for subsub_cfg in sub_cfg:
                if reference := obj_global_cache.get(str(subsub_cfg)):
                    instances.append(reference)
                else:
                    instances.append(instantiate(subsub_cfg, obj_global_cache))

            local_objs[key] = list()
            for subsub_cfg, instance in zip(sub_cfg, instances):
                if not isinstance(instance, BaseContainer):
                    obj_global_cache[str(subsub_cfg)] = instance

                local_objs[key].append(instance)

    for key, sub_cfg in cfg.items():
        if not (key.startswith("_") and key.endswith("_")):
            local_objs.setdefault(key, sub_cfg)

    if target:
        path = ('whetstone.core.' + cfg._target_).split(".")

        # problem is, path could point to a method, class or method of a class!
        mod_len = 1
        while True:
            try:
                target = import_module(".".join(path[:mod_len]))
                mod_len += 1
            except ModuleNotFoundError:
                break

        while True:
            try:
                target = getattr(target, path[mod_len - 1])
                mod_len += 1
            except (AttributeError, IndexError):
                break

        args = cfg._args_ if "_args_" in cfg else []

        if not callable(target):
            raise ModuleNotFoundError(cfg._target_)

        return target(*args, **local_objs)

    return local_objs


def mellowmax(t: Tensor, alpha=1.0, dim=-1):
    return 1.0 / alpha * (torch.logsumexp(alpha * t, dim=dim) - torch.log(
        torch.tensor(t.shape[-1], dtype=t.dtype, device=t.device)))


def generate_random_ascii(n):
    """Generate a random ASCII string of length n."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("Length must be a non-negative integer")

    # Using all printable ASCII characters
    ascii_chars = string.printable
    return ''.join(random.choice(ascii_chars) for _ in range(n))


def get_nonascii_toks(tokenizer, device="cpu"):
    def is_ascii(s):
        return s.isascii() and s.isprintable()

    nonascii_toks = []
    for i in range(tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            nonascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        nonascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        nonascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        nonascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        nonascii_toks.append(tokenizer.unk_token_id)

    return torch.tensor(nonascii_toks, device=device)

class SingletonMeta(type):
    """A metaclass for creating singleton classes."""
    _instances: Dict[Type, object] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> object:
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

def mellowmax(t: Tensor, alpha=1.0, dim=-1):
    return 1.0 / alpha * (torch.logsumexp(alpha * t, dim=dim) - torch.log(
        torch.tensor(t.shape[-1], dtype=t.dtype, device=t.device)))


logger = logging.getLogger(__name__)


def should_reduce_batch_size(exception: Exception) -> bool:
    """Check if exception indicates memory-related issues"""
    cuda_errors = [
        "CUDA out of memory.",
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",
        "DefaultCPUAllocator: can't allocate memory",
    ]

    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in cuda_errors)
    return False


def check_memory_pressure(device: torch.device) -> tuple[bool, float]:
    """Check memory pressure and return (is_pressured, current_usage_ratio)"""
    if device.type == 'mps':
        current_mem = torch.mps.current_allocated_memory()
        max_mem = torch.mps.recommended_max_memory()
        driver_mem = torch.mps.driver_allocated_memory()

        usage_ratio = current_mem / max_mem
        driver_ratio = driver_mem / max_mem
        return usage_ratio > 0.95 or driver_ratio > 0.98, usage_ratio

    elif device.type == 'cuda':
        memory_allocated = torch.cuda.memory_allocated(device)
        memory_reserved = torch.cuda.memory_reserved(device)

        if memory_reserved > 0:
            usage_ratio = memory_allocated / memory_reserved
            return usage_ratio > 0.95, usage_ratio

    return False, 0.0


def clear_memory():
    """Clear GPU memory cache and run garbage collection"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()


def batching(starting_batch_size: int = 128, fixed_batch_size=None):
    """
    Decorator that handles adaptive batch sizing for PyTorch functions.
    Ensures exact input-output correspondence with no duplicates or skips.
    """

    def decorator(func: Callable) -> Callable:
        current_batch_size = fixed_batch_size or starting_batch_size
        successful_runs = 0
        required_successful_runs = 5 if not fixed_batch_size else math.inf
        last_memory_ratio = 0.0
        optimal_memory_ratio = 0.85

        @functools.wraps(func)
        def wrapper(batch: torch.Tensor, *args, **kwargs) -> Any:
            nonlocal current_batch_size, successful_runs, last_memory_ratio

            device = batch.device
            total_size = batch.size(0)
            results = []
            processed_idx = 0

            while processed_idx < total_size:

                end_idx = min(processed_idx + current_batch_size, total_size)
                sub_batch = batch[processed_idx:end_idx]

                try:
                    result = func(sub_batch, *args, **kwargs)

                    # Check memory pressure before attempting next batch
                    is_pressured, memory_ratio = check_memory_pressure(device)

                    # On success, store result and update processed_idx
                    results.append(result)
                    processed_idx = end_idx

                    # Update success metrics
                    successful_runs += 1
                    last_memory_ratio = memory_ratio

                    # Consider increasing batch size
                    if ((successful_runs >= required_successful_runs and
                         last_memory_ratio < optimal_memory_ratio)
                            and current_batch_size < total_size):
                        headroom = 1 - last_memory_ratio
                        increase_factor = 1 + (headroom * 0.5)
                        current_batch_size = min(
                            int(current_batch_size * increase_factor),
                            total_size
                        )
                        successful_runs = 0
                        logger.info(f"Increased batch size to {current_batch_size}")

                except Exception as e:
                    if should_reduce_batch_size(e):
                        # On memory error, reduce batch size but don't advance processed_idx
                        successful_runs = 0
                        current_batch_size = max(1, int(current_batch_size * 0.5))
                        clear_memory()
                        logger.warning(f"Reduced batch size to {current_batch_size} due to memory error")
                    else:
                        raise

                clear_memory()

            # All items have been processed exactly once
            assert sum(r.size(0) for r in results) == total_size

            # Combine results
            if isinstance(results[0], torch.Tensor):
                return torch.cat(results, dim=0)
            return results

        return wrapper

    return decorator
