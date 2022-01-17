import pytest

from src.jobs import MLJob
from tests.test_config import TestConfig


class TestBackwardPropagation(TestConfig):
    """Tests that the backpropagation algorithm computes the correct gradients, i.e. the same gradients
    which are computed using a brute force method
    """
    @pytest.fixture
    def ml_job(self, config_parser):
        ml_job_ = MLJob(config_parser)

        # Train the model with the backpropagation algorithm for 1 epoch
        ml_job_.train_and_evaluate()

        return ml_job_

    def test_weight_gradients(self, ml_job):
        """Tests that the backward propagation algorithm computes the correct weight gradients"""
        pass

    def test_bias_gradients(self, ml_job):
        """Tests that the backward propagation algorithm computes the correct bias gradients"""
        pass
