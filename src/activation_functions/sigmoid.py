import numpy.typing as npt


def sigmoid_forward(dendritic_potentials: npt.NDArray) -> npt.NDArray:
    raise NotImplementedError


def sigmoid_backward(dendritic_potentials: npt.NDArray, activations: npt.NDArray) -> npt.NDArray:
    pass
