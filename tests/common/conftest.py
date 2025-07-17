import pytest

from whetstone.core import *
from whetstone.modules import *

@pytest.fixture
def dummy_model():
    return DummyModel("test")

@pytest.fixture
def dummy_objective(dummy_model):
    return DummyObjective(model=dummy_model)

@pytest.fixture
def dummy_optimizer():
    return RandomOptimizer()

@pytest.fixture
def basic_input_strs():
    return ["a_hello", "a_world", "b_foo", "d_bar"]

@pytest.fixture
def basic_constraint():
    return lambda x: x[0] in ["a", "b"]

@pytest.fixture
def corpus(dummy_objective, basic_input_strs, tmp_path, basic_constraint):
    return Corpus(objective=dummy_objective, initial_inputs=basic_input_strs, corpus_dir=tmp_path / "corpus", constraints=[basic_constraint])

@pytest.fixture
def job(corpus, dummy_objective, dummy_optimizer, tmp_path):
    return Job(
        target=Job.Target(
            objective=dummy_objective,
            optimizer=dummy_optimizer,
            corpus=corpus
        ),
        iterations=5,
        state_location=tmp_path / "state.json"
    )