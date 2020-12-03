import functools
import inspect
import itertools
import logging
import operator
from typing import Callable, SupportsFloat, Tuple, TypeVar, Union

import numpy
import torch

logger = logging.getLogger(__name__)


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


def _is_oom_error(error: RuntimeError) -> bool:
    """Check whether a runtime error was caused by insufficient memory."""
    message = error.args[0]
    logger.debug(f"Checking error for OOM: {message}")

    # CUDA out of memory
    if 'CUDA out of memory.' in message:
        return True

    # CUDA error (dimension was larger than int limit)
    if "RuntimeError: CUDA error: invalid configuration argument" in message:
        return True

    # CPU out of memory
    if "[enforce fail at CPUAllocator.cpp:64] . DefaultCPUAllocator: can't allocate memory:" in message:
        return True

    return False


R = TypeVar("R")


def maximize_memory_utilization_(
    parameter_name: str,
    q: int = 32,
    cpu_warning: bool = True,
) -> Callable[[Callable[..., R]], Callable[..., Tuple[R, int]]]:
    """
    A decorator factory to create methods for memory utilization maximization.

    :param parameter_name:
        The parameter name.
    :param q:
        Prefer multiples of q as size.
    :param cpu_warning:
        Whether to check the input for CPU tensors and warn about potential CPU OOM problems.

    :return:
        A decorator for functions.
    """

    if cpu_warning:
        def check_for_cpu_tensors(*args, **kwargs):
            if any(
                (torch.is_tensor(obj) and obj.device.type == "cpu")
                for obj in itertools.chain(args, kwargs.values())
            ):
                logger.warning(
                    "Using maximize_memory_utilization on non-CUDA tensors. This may lead to undocumented crashes due to CPU OOM killer."
                )
    else:
        def check_for_cpu_tensors(*args, **kwargs):
            pass

    def decorator_maximize_memory_utilization(func: Callable[..., R]) -> Callable[..., Tuple[R, int]]:
        """
        A decorator to maximize memory utilization.

        :param func:
            The function to decorate.

        :return:
            The decorated function.
        """
        # Input validation
        signature = inspect.signature(func)
        if parameter_name not in signature.parameters.keys():
            raise ValueError(f"{func} does not have a parameter {parameter_name}.")
        _parameter = signature.parameters[parameter_name]
        if _parameter.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
            # TODO: we could also support positional ones by saving the position
            raise ValueError(f"{parameter_name} must be a keyword based parameter.")
        if _parameter.annotation != inspect.Parameter.empty and _parameter.annotation != int:
            logger.warning(f"Memory utilization maximization is written for integer parameters, but the {parameter_name} is annotated as {_parameter.annotation}")

        @functools.wraps(func)
        def wrapper_maximize_memory_utilization(*args, **kwargs) -> Tuple[R, int]:
            """
            A wrapper around the function to maximize memory utilization by successive halving.

            :param args:
                The positional arguments.
            :param kwargs:
                The key-word based arguments.

            :return:
                A tuple (result, max_value).
            """
            check_for_cpu_tensors(*args, **kwargs)
            max_value = kwargs.pop(parameter_name)
            while max_value > 0:
                p_kwargs = {
                    parameter_name: max_value
                }
                try:
                    return func(*args, **p_kwargs, **kwargs), max_value
                except RuntimeError as runtime_error:
                    # clear cache
                    torch.cuda.empty_cache()

                    # check whether the error is an out-of-memory error
                    if not _is_oom_error(error=runtime_error):
                        raise runtime_error

                    logger.info(f"Execution failed with {parameter_name}={max_value}")
                    max_value //= 2
                    if max_value > q:
                        max_value = max_value // q * q
            raise MemoryError(f"Execution did not even succeed with {parameter_name}=1.")

        return wrapper_maximize_memory_utilization

    return decorator_maximize_memory_utilization
