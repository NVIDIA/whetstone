import pytest

from whetstone.modules.models.hf_transformers import HFTransformersModel
from whetstone.modules.objectives.differentiable.str_match import StrMatchObjective
from whetstone.modules.objectives.dummy import DummyObjective
from whetstone.modules.objectives.length import LengthObjective
from whetstone.modules.objectives.llm_judge import LLMJudge


def dummy_objective(model):
    return DummyObjective(model)

def length_objective():
    return LengthObjective()

def llm_judge_objective(model):
    return LLMJudge(model, model)

def strmatch_objective(model):
    return StrMatchObjective(model)
 
@pytest.fixture(scope="session", params=
                [dummy_objective, 
                 length_objective, 
                 llm_judge_objective, 
                 strmatch_objective])
def objective(request, model):
    if request.param == strmatch_objective and not isinstance(model, HFTransformersModel):
        pytest.skip("StrMatchObjective requires a HFTransformersModel")

    if request.param in [dummy_objective, llm_judge_objective, strmatch_objective]:
        return request.param(model)
    else:
        return request.param()

@pytest.fixture(scope="session", params=[dummy_objective, strmatch_objective])
def diff_objective(request, model):
    if request.param == strmatch_objective and not isinstance(model, HFTransformersModel):
        pytest.skip("StrMatchObjective requires a HFTransformersModel")
        
    return request.param(model)