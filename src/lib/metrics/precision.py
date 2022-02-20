import logging

import numpy as np

from src.lib.metrics import Metric

logger = logging.getLogger(__name__)


class Precision(Metric):
    def result(self):
        """Compute precision and return NaN if there are no predictions of the positive class"""
        if (self.all_true_positives + self.all_false_positives) == 0:
            return np.nan
        else:
            return self.all_true_positives / (self.all_true_positives + self.all_false_positives)
