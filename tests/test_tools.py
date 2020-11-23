import itertools
import random
import timeit
from typing import Iterable, Tuple

import numpy
import torch

from tools import calculate_broadcasted_elementwise_result_shape, estimate_cost_of_sequence, get_optimal_sequence, tensor_sum


def test_calculate_broadcasted_elementwise_result_shape():
    """Test calculate_broadcasted_elementwise_result_shape."""
    max_dim = 64
    for n_dim, i in itertools.product(range(2, 5), range(10)):
        a_shape = [1 for _ in range(n_dim)]
        b_shape = [1 for _ in range(n_dim)]
        for j in range(n_dim):
            dim = 2 + random.randrange(max_dim)
            mod = random.randrange(3)
            if mod % 2 == 0:
                a_shape[j] = dim
            if mod > 0:
                b_shape[j] = dim
            a = torch.empty(*a_shape)
            b = torch.empty(*b_shape)
            shape = calculate_broadcasted_elementwise_result_shape(first=a.shape, second=b.shape)
            c = a + b
            exp_shape = c.shape
            assert shape == exp_shape


def _generate_shapes(
    n_dim: int = 5,
    n_terms: int = 4,
    iterations: int = 64,
) -> Iterable[Tuple[Tuple[int, ...], ...]]:
    """Generate shapes."""
    max_shape = torch.randint(low=2, high=32, size=(128,))
    for _i in range(iterations):
        # create broadcastable shapes
        idx = torch.randperm(max_shape.shape[0])[:n_dim]
        this_max_shape = max_shape[idx]
        this_min_shape = torch.ones_like(this_max_shape)
        shapes = []
        for _j in range(n_terms):
            mask = this_min_shape
            while not (1 < mask.sum() < n_dim):
                mask = torch.as_tensor(torch.rand(size=(n_dim,)) < 0.3, dtype=max_shape.dtype)
            this_array_shape = this_max_shape * mask + this_min_shape * (1 - mask)
            shapes.append(tuple(this_array_shape.tolist()))
        yield tuple(shapes)


def test_estimate_cost_of_add_sequence():
    """Test ``estimate_cost_of_add_sequence()``."""
    # create random array, estimate the costs of addition, and measure some execution times.
    # then, compute correlation between the estimated cost, and the measured time.
    data = []
    for shapes in _generate_shapes():
        arrays = [torch.empty(*shape) for shape in shapes]
        cost = estimate_cost_of_sequence(*(a.shape for a in arrays))
        consumption = timeit.timeit(stmt='sum(arrays)', globals=locals(), number=25)
        data.append((cost, consumption))
    a = numpy.asarray(data)

    # check for strong correlation between estimated costs and measured execution time
    assert (numpy.corrcoef(x=a[:, 0], y=a[:, 1])[0, 1]) > 0.8


def test_get_optimal_add_sequence():
    """Test ``get_optimal_add_sequence()``."""
    for shapes in _generate_shapes():
        # get optimal sequence
        opt_cost, opt_seq = get_optimal_sequence(*shapes)

        # check correct cost
        exp_opt_cost = estimate_cost_of_sequence(*(shapes[i] for i in opt_seq))
        assert exp_opt_cost == opt_cost

        # check optimality
        for perm in itertools.permutations(list(range(len(shapes)))):
            cost = estimate_cost_of_sequence(*(shapes[i] for i in perm))
            assert cost >= opt_cost


def test_tensor_sum():
    """Test tensor_sum."""
    for shapes in _generate_shapes():
        tensors = [torch.rand(*shape) for shape in shapes]
        result = tensor_sum(*tensors)

        # compare result to sequential addition
        assert torch.allclose(result, sum(tensors))
