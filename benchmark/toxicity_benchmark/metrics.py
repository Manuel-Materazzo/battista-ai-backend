"""Metrics calculation utilities"""

from typing import List, Dict

from .models import TestResult, Metrics


class MetricsCalculator:
    """Encapsulates all metrics calculation logic"""

    @staticmethod
    def calculate_metrics(results: List[TestResult]) -> Metrics:
        """
        Calculate all metrics from test results.

        Args:
            results: List of test results

        Returns:
            Dictionary containing all calculated metrics
        """
        if not results:
            return {}

        total = len(results)
        correct = sum(1 for r in results if r["correct"])
        accuracy = (correct / total) * 100

        # Confusion matrix
        tp = sum(1 for r in results if r["true_toxic"] and r["predicted_toxic"])
        fp = sum(1 for r in results if not r["true_toxic"] and r["predicted_toxic"])
        tn = sum(1 for r in results if not r["true_toxic"] and not r["predicted_toxic"])
        fn = sum(1 for r in results if r["true_toxic"] and not r["predicted_toxic"])

        # Precision, Recall, F1
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        return {
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": accuracy,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }
