import logging

from src.metrics.interface import Metric

logger = logging.getLogger(__name__)


class Accuracy(Metric):
    def __init__(self, name: str):
        self.name = name
