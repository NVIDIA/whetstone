import torch

from whetstone.core.objective import Objective, Sample, DifferentiableObjective


def test_objective_evaluate_batch(objective: Objective):
    """Tests the basic functionality of evaluate_batch."""
    results = objective.evaluate_batch(["a", "b", "c"])
    assert len(results) == 3
    assert results[0].input == "a"
    assert results[1].input == "b"
    assert results[2].input == "c"


def test_objective_evaluate(objective: Objective):
    """Tests the basic functionality of evaluate and its consistency with evaluate_batch."""
    results_batch = objective.evaluate_batch(["a", "b", "c"])
    result_a = objective.evaluate("a")
    result_b = objective.evaluate("b")
    result_c = objective.evaluate("c")

    assert result_a.input == "a"
    assert result_b.input == "b"
    assert result_c.input == "c"

    # Ensure the scores are repeatable (at least for the score) via caching
    assert results_batch[0].score == result_a.score
    assert results_batch[1].score == result_b.score
    assert results_batch[2].score == result_c.score


def test_objective_callback(objective: Objective):
    """Tests the callback registration and invocation."""
    objective._cache.clear()
    callback_calls_new = []
    callback_calls_all = []

    def callback(results: list[Sample]):
        callback_calls_new.append(results)

    def callback_cached(results: list[Sample]):
        callback_calls_all.append(results)

    objective._register_callback(callback, call_on_cached=False)
    objective._register_callback(callback_cached, call_on_cached=True)

    # Evaluate batch should trigger callback once for the batch
    objective.evaluate_batch(["a", "b", "c"])
    assert len(callback_calls_all) == 1
    assert len(callback_calls_all[0]) == 3
    assert callback_calls_all[0][0].input == "a"
    assert callback_calls_all[0][1].input == "b"
    assert callback_calls_all[0][2].input == "c"

    objective.evaluate("d")
    objective.evaluate("d")
    assert len(callback_calls_new) == 2
    assert len(callback_calls_all) == 3

def test_objective_score_range(objective: Objective):
    """Ensures the scores are in the range [0, 1]."""
    results = objective.evaluate_batch(["x", "y", "z"])
    assert all(0 <= result.score <= 1 for result in results)


def test_objective_is_achieved(objective: Objective):
    """Tests the is_achieved method."""
    results = objective.evaluate_batch(["p", "q"])

    # Assuming scores might vary, let's test thresholds
    results[0].score = 0.0
    assert objective.is_achieved(results[0], 0.0)
    assert objective.is_achieved(results[0], 0.1)

    results[1].score = 0.5
    assert not objective.is_achieved(results[1], 0.0)
    assert not objective.is_achieved(results[1], 0.4)
    assert objective.is_achieved(results[1], 0.5)
    assert objective.is_achieved(results[1], 1.0)


def test_diff_objective_gradient_batch(diff_objective: DifferentiableObjective):
    """Tests the gradient_batch method for DifferentiableObjective."""
    inputs = ["grad_a", "grad_b"]
    results = diff_objective.gradient_batch(inputs)

    assert len(results) == 2
    assert isinstance(results[0], tuple)
    assert len(results[0]) == 2
    assert isinstance(results[0][0], Sample)
    assert isinstance(results[0][1], torch.Tensor)
    assert results[0][0].input == "grad_a"

    assert isinstance(results[1], tuple)
    assert len(results[1]) == 2
    assert isinstance(results[1][0], Sample)
    assert isinstance(results[1][1], torch.Tensor)
    assert results[1][0].input == "grad_b"

    # Check score range inherited from Objective evaluation via _gradient_batch
    assert 0 <= results[0][0].score <= 1
    assert 0 <= results[1][0].score <= 1


def test_diff_objective_gradient(diff_objective: DifferentiableObjective):
    """Tests the gradient method and its consistency with gradient_batch."""
    # Clear caches to ensure consistency check is valid
    diff_objective._cache.clear()
    diff_objective._gradient_cache.clear()

    batch_results = diff_objective.gradient_batch(["grad_c"])
    single_result = diff_objective.gradient("grad_c")

    assert isinstance(single_result, tuple)
    assert len(single_result) == 2
    assert isinstance(single_result[0], Sample)
    assert isinstance(single_result[1], torch.Tensor)
    assert single_result[0].input == "grad_c"

    # Ensure consistency between batch and single evaluation (both sample and gradient)
    assert batch_results[0][0] == single_result[0]
    assert torch.equal(batch_results[0][1], single_result[1])



