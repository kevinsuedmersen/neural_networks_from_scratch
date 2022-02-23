import numpy as np
import pytest
from sklearn import metrics

from src.lib.activation_functions import softmax_forward
from src.lib.metrics.cost.categorical_crossentropy import CategoricalCrossentropyMetric
from src.lib.metrics.score.accuracy import Accuracy
from src.lib.metrics.score.precision import Precision
from src.lib.metrics.score.recall import Recall
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
        # Make sure predictions are between 0 and 1 and that they all sum up to 1
        random_ypred = softmax_forward(random_floats)

        return random_ypred

    @staticmethod
    def _assert_metric_result(metric, expected_result, ytrue, ypred):
        """Tests whether the metric value is computed correctly 10 consecutive times"""
        for epoch in range(10):
            metric.reset_state()
            for batch in range(10):
                metric.update_state(ytrue, ypred)
            actual_result = metric.result()
            assert actual_result == expected_result


class TestScoreMetric(TestMetric):
    @pytest.fixture
    def binarized_ypred(self, ypred):
        # Make sure our metric calculation is not dependent on how the threshold value is handled
        ypred[ypred == 0.5] = 0.001
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

    def test_accuracy(self, accuracy, ytrue, ypred, binarized_ypred):
        """Test accuracy computation using micro averaging"""
        expected_result = metrics.accuracy_score(ytrue.ravel(), binarized_ypred.ravel())
        self._assert_metric_result(accuracy, expected_result, ytrue, ypred)

    @pytest.fixture
    def precision(self):
        return Precision("precision", self.threshold)

    def test_precision(self, precision, ytrue, ypred, binarized_ypred):
        """Test precision computation using micro averaging"""
        expected_result = metrics.precision_score(ytrue.ravel(), binarized_ypred.ravel())
        self._assert_metric_result(precision, expected_result, ytrue, ypred)

    @pytest.fixture
    def recall(self):
        return Recall("recall", self.threshold)

    def test_recall(self, recall, ytrue, ypred, binarized_ypred):
        """Test recall computation using micro averaging"""
        expected_result = metrics.recall_score(ytrue.ravel(), binarized_ypred.ravel())
        self._assert_metric_result(recall, expected_result, ytrue, ypred)


class TestCostMetric(TestMetric):
    @pytest.fixture
    def cce_metric(self):
        return CategoricalCrossentropyMetric("categorical_crossentropy")

    def test_cce(self, cce_metric, ytrue, ypred):
        """Test categorical cross entroppy computation"""
        expected_result = metrics.log_loss(np.squeeze(ytrue), np.squeeze(ypred))
        self._assert_metric_result(cce_metric, expected_result, ytrue, ypred)
