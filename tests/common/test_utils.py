from unittest.mock import Mock

import torch

from whetstone.utils import clear_memory


def test_clear_memory_skips_unavailable_accelerators(monkeypatch):
    cuda_empty_cache = Mock()
    mps_empty_cache = Mock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", cuda_empty_cache)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.mps, "empty_cache", mps_empty_cache)

    clear_memory()

    cuda_empty_cache.assert_not_called()
    mps_empty_cache.assert_not_called()
