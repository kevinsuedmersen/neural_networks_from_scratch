from typing import Callable

import numpy.typing as npt


def simplify_init_error(
        output_activation_: str,
        task_: str
) -> Callable:
    def decorator(method) -> Callable:
        def method_wrapper(
                loss_instance,
                ytrue: npt.NDArray,
                dendritic_potentials: npt.NDArray,
                activations: npt.NDArray
        ) -> npt.NDArray:
            if (loss_instance.output_activation == output_activation_) and (loss_instance.task == task_):
                error = ytrue - activations
            else:
                error = method(loss_instance, ytrue, dendritic_potentials, activations)

            return error

        return method_wrapper

    return decorator
