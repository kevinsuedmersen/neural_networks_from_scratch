import numpy as np
import pytest

from src.lib.activation_functions import sigmoid_forward
from src.lib.metrics.accuracy import Accuracy
from tests.test_config import TestConfig


class TestMetric(TestConfig):
    threshold = 0.5

    @pytest.fixture
    def ytrue(self):
        random_floats = np.random.randn(self.batch_size, self.n_neurons_out, 1)
        random_ytrue = random_floats.copy()
        random_ytrue[random_floats >= self.threshold] = 1
        random_ytrue[random_floats < self.threshold] = 0

        return random_ytrue

    @pytest.fixture
    def ypred(self):
        random_floats = np.random.randn(self.batch_size, self.n_neurons_out, 1)
        # Make sure predictions are between 0 and 1
        random_ypred = sigmoid_forward(random_floats)

        return random_ypred


class TestAccuracy(TestMetric):
    @pytest.fixture
    def accuracy(self):
        return Accuracy("accuracy")

    def test_update_state_and_return_result(self, accuracy, ytrue, ypred):
        # TODO: Think of a good way to test update of states
        n_correct_preds = []
        n_all_preds = []
        results = []
        for i in range(10):
            accuracy.update_state(ytrue, ypred)
            n_correct_preds.append(accuracy.n_correct_preds)
            n_all_preds.append(accuracy.n_all_preds)
            results.append(accuracy.result())

        assert len(set(results)) == 1
