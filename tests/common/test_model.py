
import torch
from whetstone.core.model import Model

def test_model_generate(model: Model):
    i_str = "Hello, world!"
    i_tokens = model.tokenize(i_str)
    limited_generation = model.generate(i_str, max_tokens=5)
    output_tokens = model.tokenize(i_str + limited_generation)

    # Important to check that the tokenizer isn't adding any special tokens
    assert len(output_tokens) <= len(i_tokens) + 5
    
    # Check for default repeatability
    repeated_generation = model.generate(i_str, max_tokens=10)
    assert repeated_generation.startswith(limited_generation)
    assert len(model.tokenize(i_str + repeated_generation)) <= len(i_tokens) + 10

def test_model_tokenize(model: Model):
    tokens = model.tokenize("Hello, world!")
    assert len(tokens) > 0
    assert isinstance(tokens, torch.Tensor)

def test_model_detokenize(model: Model):
    tokens = model.tokenize("Hello, world!")
    assert len(model.detokenize(tokens)) > 0

    str_tokens = model.detokenize(tokens)
    assert isinstance(str_tokens, str)
    assert str_tokens == "Hello, world!"


def test_model_vocab_size(model: Model):
    try:
        assert model.vocab_size() > 0
    except NotImplementedError:
        pass

def test_model_render_conversation_template(model: Model):
    conversation = [
        {"role": "user", "content": "Hello, world!"},
        {"role": "assistant", "content": "Hello, world!"},
    ]
    
    try:
        output = model.render_conversation_template(conversation)
        assert len(output) > 0
        assert isinstance(output, str)
    except NotImplementedError:
        pass

