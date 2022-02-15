import numpy as np


def compute_euclidean_distance(actual: np.ndarray, expected: np.ndarray):
    """Computes euclidean distance"""
    _actual = actual.ravel()
    _expected = expected.ravel()
    assert _actual.shape == _expected.shape
    squared_distances = (_actual - _expected)**2
    euclidean_distance = np.sqrt(np.sum(squared_distances))

    return euclidean_distance


def assert_euclidean_distance(
        actual: np.ndarray,
        expected: np.ndarray,
        absolute_tolerance: float = 1e-15
):
    """Asserts that the euclidean distance is below a certain threshold"""
    euclidean_distance = compute_euclidean_distance(actual, expected)
    assert euclidean_distance < absolute_tolerance
