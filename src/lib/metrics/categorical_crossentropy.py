import logging

from src.lib.losses.categorical_crossentropy import CategoricalCrossEntropyLoss
from src.lib.metrics.cost_metric import CostMetric

logger = logging.getLogger(__name__)


class CategoricalCrossEntropyMetric(CostMetric):
    def __init__(self, name):
        super().__init__(name)
        # The categorical cross entropy should always be used in a Multi Class Classification scheme
        # and if the output activation function is the softmax function
        self.loss = CategoricalCrossEntropyLoss("softmax", "multi_class_classification")
