import logging
from abc import ABC

logger = logging.getLogger(__name__)


class Metric(ABC):
    pass


class Accuracy(Metric):
    pass
