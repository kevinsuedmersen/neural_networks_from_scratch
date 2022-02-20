import logging

import numpy as np

from src.lib.metrics.score import ScoreMetric

logger = logging.getLogger(__name__)


class Recall(ScoreMetric):
    def result(self):
        """Compute recall and return NaN if there are no instances of the positive class"""
        if (self.all_true_positives + self.all_false_negatives) == 0:
            return np.nan
        else:
            return self.all_true_positives / (self.all_true_positives + self.all_false_negatives)
