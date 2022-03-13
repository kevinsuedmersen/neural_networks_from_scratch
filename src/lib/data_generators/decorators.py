from typing import Callable, Generator, Tuple

import numpy.typing as npt


def control_looping_behavior(method) -> Callable:
    """Decorator which controls the looping behavior of any data generator"""
    def _method_wrapper(data_gen_instance, *args, **kwargs) -> Generator[Tuple[npt.NDArray, npt.NDArray], None, None]:
        """Making sure the data generator is exhausted after one epoch or continues to loop forever"""
        if data_gen_instance.loop_forever:
            while True:
                data_gen = method(data_gen_instance, *args, **kwargs)
                return data_gen
        else:
            data_gen = method(data_gen_instance, *args, **kwargs)
            return data_gen

    return _method_wrapper
