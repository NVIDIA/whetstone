
import pytest

from whetstone.modules.models.dummy import DummyModel
from whetstone.modules.models.hf_transformers import HFTransformersModel


def dummy_model():
    return DummyModel("test")


def hf_model():
    return HFTransformersModel("arnir0/Tiny-LLM")


@pytest.fixture(scope="session", params=[dummy_model, hf_model])
def model(request):
    return request.param()