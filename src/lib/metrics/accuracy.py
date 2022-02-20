import logging

from src.lib.metrics.score_metric import ScoreMetric

logger = logging.getLogger(__name__)


class Accuracy(ScoreMetric):
    def result(self):
        """Returns the percentage of correct predictions from all predictions"""
        accuracy_value = (
                (self.all_true_positives + self.all_true_negatives) /
                (self.all_true_positives + self.all_false_positives + self.all_true_negatives + self.all_false_negatives)
        )

        return accuracy_value
