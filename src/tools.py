import functools
import itertools
import operator
from typing import Callable, SupportsFloat, Tuple, Union

import numpy
import torch


def calculate_broadcasted_elementwise_result_shape(
    first: Tuple[int, ...],
    second: Tuple[int, ...],
) -> Tuple[int, ...]:
    """Determine the return shape of a broadcasted elementwise operation."""
    return tuple(max(a, b) for a, b in zip(first, second))


def estimate_cost_of_sequence(
    shape: Tuple[int, ...],
    *other_shapes: Tuple[int, ...],
) -> int:
    """Cost of a sequence of broadcasted element-wise operations of tensors, given their shapes."""
    return sum(
        map(
            numpy.prod,
            itertools.islice(
                itertools.accumulate(
                    other_shapes,
                    calculate_broadcasted_elementwise_result_shape,
                    initial=shape,
                ),
                1,
                None,
            )
        )
    )


@functools.lru_cache(maxsize=32)
def _get_optimal_sequence(
    *sorted_shapes: Tuple[int, ...],
) -> Tuple[int, Tuple[int, ...]]:
    """Find the optimal sequence in which to combine tensors element-wise based on the shapes.

    The shapes should be sorted to enable efficient caching.

    :param sorted_shapes:
        The shapes of the tensors to combine.

    :return:
        The optimal execution order (as indices), and the cost.
    """
    return min(
        (estimate_cost_of_sequence(*(sorted_shapes[i] for i in p)), p)
        for p in itertools.permutations(list(range(len(sorted_shapes))))
    )


def get_optimal_sequence(*shapes: Tuple[int, ...]) -> Tuple[int, Tuple[int, ...]]:
    """Find the optimal sequence in which to combine tensors elementwise based on the shapes.

    :param shapes:
        The shapes of the tensors to combine.

    :return:
        The optimal execution order (as indices), and the cost.
    """
    # create sorted list of shapes to allow utilization of lru cache (optimal execution order does not depend on the
    # input sorting, as the order is determined by re-ordering the sequence anyway)
    arg_sort = sorted(range(len(shapes)), key=shapes.__getitem__)

    # Determine optimal order and cost
    cost, optimal_order = _get_optimal_sequence(*(shapes[new_index] for new_index in arg_sort))

    # translate back to original order
    optimal_order = tuple(arg_sort[i] for i in optimal_order)

    return cost, optimal_order


def _multi_combine(
    tensors: Tuple[torch.FloatTensor, ...],
    op: Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor],
    initial: Union[None, torch.FloatTensor, SupportsFloat] = None,
) -> torch.FloatTensor:
    """Broadcasted element-wise combination of tensors.

    The optimal execution plan gets cached so that the optimization is only performed once for a fixed set of shapes.

    :param tensors:
        The tensors, in broadcastable shape.
    :param op:
        The elementwise operator.
    :param initial:
        An initial value.

    :return:
        The elementwise combination evaluated in optimal processing order.
    """
    # determine optimal processing order
    order = get_optimal_sequence(*(t.shape for t in tensors))[1]
    if initial is None:
        initial = tensors[order[0]]
        order = order[1:]
    return functools.reduce(op, [tensors[i] for i in order], initial)


def tensor_sum(*tensors: torch.FloatTensor) -> torch.FloatTensor:
    """Broadcasted addition of tensors in optimal order.

    The optimal execution plan gets cached so that the optimization is only performed once for a fixed set of shapes.

    :param tensors:
        The tensors to add. They have to be in a broadcastable shape.

    :return:
        sum(*tensors) evaluated in an optimal processing order.
    """
    return _multi_combine(tensors=tensors, op=operator.add)
