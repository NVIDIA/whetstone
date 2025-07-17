import torch

from whetstone.core.optimizer import GradientOptimizer, Optimizer, Iteration


def test_optimizer_step(optimizer: Optimizer, objective, corpus):
    try:
        iteration = optimizer.step(objective, corpus)
    except ValueError as e:
        assert "Unsupported model" in str(e)
        return
    if iteration is not None:
        assert isinstance(iteration, Iteration)
        assert iteration.timestamp is not None
   
def test_gradient_optimizer_step(diff_optimizer: GradientOptimizer, diff_objective, corpus):
    try:
        iteration = diff_optimizer.step(diff_objective, corpus)
    except ValueError as e:
        assert "Unsupported model" in str(e)
        return
    if iteration is not None:   
        assert isinstance(iteration, Iteration)
        assert iteration.timestamp is not None
