from typing import Union

from src.config_parser.classification import ImageClassificationConfigParser
from src.jobs import MLJob
from src.jobs.classification import ImageClassificationJob


def get_ml_job(job_type: str, cp: Union[ImageClassificationConfigParser]) -> MLJob:
    """Gets an MlJob instance"""
    jobs = {
        "image_classification_job": ImageClassificationJob(cp)
    }
    return jobs[job_type]
