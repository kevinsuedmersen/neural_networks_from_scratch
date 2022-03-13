from typing import Callable

import numpy.typing as npt


def simplify_init_error(
        output_activation_: str,
        task_: str
) -> Callable:
    """This decorator can be used, if the initialization of the error at the output layer (error_L) can be
    simplified to `-(ytrue - ypred)`. If it cannot be simplified, error_L is calculated by the loss's
    `init_error` method.

    `simplify_init_error` is the outer decorator which returns a parametrized, inner decorator
    """
    def _decorator(method) -> Callable:
        """This is the actual, inner decorator which is returned by calling `simplify_init_error`"""
        def _method_wrapper(
                loss_instance,
                ytrue: npt.NDArray,
                dendritic_potentials_out: npt.NDArray,
                activations_out: npt.NDArray
        ) -> npt.NDArray:
            """This is the actual method wrapper which implements the logic of the decorator"""
            if (loss_instance.output_activation == output_activation_) and (loss_instance.task == task_):
                # Notice that -(ytrue - activations_out) = (activations_out - ytrue)
                error = activations_out - ytrue
            else:
                error = method(loss_instance, ytrue, dendritic_potentials_out, activations_out)

            return error

        return _method_wrapper

    return _decorator
