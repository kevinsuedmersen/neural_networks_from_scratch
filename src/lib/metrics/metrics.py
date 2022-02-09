import logging

from src.lib.metrics import Metric

logger = logging.getLogger(__name__)


class Accuracy(Metric):
    def __init__(self, name: str):
        self.name = name
