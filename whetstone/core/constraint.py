# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar

from pydantic import BaseModel

from whetstone.core.module import BaseModule, ModuleRegistry

class Constraint(BaseModule, ABC, Callable[[str], bool]):
    """
    Base class for input constraints.
    
    Constraints define validation rules that inputs must satisfy to be included
    in a corpus.
    """

    @abstractmethod
    def __call__(self, input: str) -> bool:
        """Check if the input satisfies the constraint."""
        pass