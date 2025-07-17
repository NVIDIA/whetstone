from .conftest import *


def test_corpus_initial_inputs(corpus, basic_constraint, basic_input_strs):
    assert len(corpus) == 3
    for input in basic_input_strs:
        if basic_constraint(input):
            assert input in corpus.inputs

def test_corpus_add(corpus):
    corpus.add("a_hello")
    assert len(corpus) == 3
    assert "a_hello" in corpus.inputs

    corpus.add("c_baz")
    assert len(corpus) == 3
    assert "c_baz" not in corpus.inputs

    corpus.add("b_zoo")
    assert len(corpus) == 4
    assert "b_zoo" in corpus.inputs

def test_corpus_add_many(corpus):
    corpus.add_many([
        "a_hello",
        "c_baz",
        "b_zoo",
    ])
    assert len(corpus) == 4
    assert "a_hello" in corpus.inputs
    assert "c_baz" not in corpus.inputs
    assert "b_zoo" in corpus.inputs



