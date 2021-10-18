"""Module for testing matrix multiplication with 3D matrices
"""
import pytest
import numpy as np


class Test3DMatmul:
    batch_size = 10
    n = 3
    k = 4
    m = 5

    @pytest.fixture
    def left(self) -> np.ndarray(shape=(n, k)):
        return np.random.randint(0, 3, size=(self.batch_size, self.n, self.k))

    @pytest.fixture
    def right(self) -> np.ndarray(shape=(k, m)):
        return np.random.randint(0, 3, size=(1, self.k, self.m))

    def test_batch_wise_matmul(self, left, right):
        # Use np.matmul to compute batch_size matrix multiplications in parallel
        result_1 = np.matmul(left, right)

        # Fake the broadcasting of the right matrix
        right_broadcasted = np.concatenate([right for _ in range(self.batch_size)], axis=0)

        # Iterate through each element in the batch, do matrix multiplication and combine the intermediate
        # results
        interim_results = []
        for batch_idx in range(self.batch_size):
            current_left = left[batch_idx, ...]
            current_right = right_broadcasted[batch_idx, ...]
            current_result = np.matmul(current_left, current_right)
            # Add batch dimenstion to current result
            current_result = current_result[np.newaxis, ...]
            interim_results.append(current_result)
        result_2 = np.concatenate(interim_results, axis=0)

        # Test whether the two methods provided the same results
        np.testing.assert_array_equal(result_1, result_2)
