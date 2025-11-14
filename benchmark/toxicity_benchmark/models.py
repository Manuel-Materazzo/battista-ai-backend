"""Data models for test results and metrics"""

from typing import TypedDict, Optional


class TestSample(TypedDict, total=False):
    """A single test sample from dataset"""
    message: str
    toxic: bool
    score: Optional[float]


class TestResult(TypedDict, total=False):
    """Result of testing a single sample"""
    message: str
    true_toxic: bool
    true_score: Optional[float]
    predicted_toxic: bool
    correct: bool
    dataset: str


class Metrics(TypedDict, total=False):
    """Calculated metrics from test results"""
    total: int
    correct: int
    incorrect: int
    accuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
