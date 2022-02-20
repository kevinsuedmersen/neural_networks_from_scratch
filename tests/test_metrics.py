import numpy as np
import pytest

from src.lib.activation_functions import sigmoid_forward
from src.lib.metrics.accuracy import Accuracy
from tests.test_config import TestConfig


class TestMetric(TestConfig):
    threshold = 0.5
    # Overwrite batch_size and n_neurons_out to make it easier to understand whether the metrics
    # work as expected ==> Choose them so that we will have 100 predictions in total
    # batch_size = 50
    # n_neurons_out = 2

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

    @pytest.fixture
    def binarized_ypred(self, ypred):
        binarized_ypred = Accuracy._binarize(ypred, self.threshold)

        return binarized_ypred

    @pytest.fixture
    def n_true_positives(self, ytrue, binarized_ypred):
        n_true_positives = np.sum((binarized_ypred == 1) & (ytrue == 1)).item()

        return n_true_positives

    @pytest.fixture
    def n_false_positives(self, ytrue, binarized_ypred):
        n_false_positives = np.sum((binarized_ypred == 1) & (ytrue == 0)).item()

        return n_false_positives

    @pytest.fixture
    def n_true_negatives(self, ytrue, binarized_ypred):
        n_true_negatives = np.sum((binarized_ypred == 0) & (ytrue == 0)).item()

        return n_true_negatives

    @pytest.fixture
    def n_false_negatives(self, ytrue, binarized_ypred):
        n_false_negatives = np.sum((binarized_ypred == 0) & (ytrue == 1)).item()

        return n_false_negatives

    @pytest.fixture
    def accuracy(self):
        return Accuracy("accuracy", self.threshold)

    def test_accuracy(
            self,
            accuracy,
            ytrue,
            ypred,
            binarized_ypred,
            n_true_positives,
            n_false_positives,
            n_true_negatives,
            n_false_negatives
    ):
        n_correct_preds = np.sum(binarized_ypred == ytrue).item()
        n_all_preds = ytrue.size
        expected_result = n_correct_preds / n_all_preds * 100

        for epoch in range(100):
            accuracy.reset_state()
            for batch in range(100):
                accuracy.update_state(ytrue, ypred)
            actual_result = accuracy.result()
            assert actual_result == expected_result


class TestAccuracy(TestMetric):
    @pytest.fixture
    def accuracy(self):
        return Accuracy("accuracy")

    def test_update_state_and_return_result(self, accuracy, ytrue, ypred):
        n_correct_preds = []
        n_all_preds = []
        results = []
        for i in range(10):
            accuracy.update_state(ytrue, ypred)
            n_correct_preds.append(accuracy.n_correct_preds)
            n_all_preds.append(accuracy.n_all_preds)
            results.append(accuracy.result())

        assert len(set(results)) == 1
