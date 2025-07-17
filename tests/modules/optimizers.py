import pytest

from whetstone.modules.optimizers.gcg import GCG
from whetstone.modules.optimizers.llm import LLMOptimizer
from whetstone.modules.optimizers.multi import MultiOptimizer
from whetstone.modules.optimizers.random import RandomOptimizer


def random_optimizer():
    return RandomOptimizer(num_chars_to_replace=1)

def gcg_optimizer(model):
    return GCG(model, topk=1, search_width=1, filter_ids=False)

def llm_optimizer(model):
    return LLMOptimizer(model, depth=1)

def multi_optimizer():
    return MultiOptimizer([random_optimizer(), random_optimizer()])

@pytest.fixture(scope="session", params=[random_optimizer, llm_optimizer, multi_optimizer])
def optimizer(request, model):
    if request.param == gcg_optimizer or request.param == llm_optimizer:
        return request.param(model)
    else:
        return request.param()

@pytest.fixture(scope="session", params=[gcg_optimizer])
def diff_optimizer(request, model):
    return request.param(model)
