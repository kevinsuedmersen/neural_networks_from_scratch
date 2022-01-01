import numpy.typing as npt


def tanh_forward(dendritic_potentials: npt.NDArray) -> npt.NDArray:
    raise NotImplementedError


def tanh_backward(dendritic_potentials: npt.NDArray, activations: npt.NDArray) -> npt.NDArray:
    raise NotImplementedError
