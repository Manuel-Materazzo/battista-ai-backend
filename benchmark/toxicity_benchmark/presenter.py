"""Results presentation and persistence utilities"""

import json
import os
from pathlib import Path
from typing import List, Dict

from .models import TestResult, Metrics
from .config import (
    MISCLASSIFIED_DISPLAY_LIMIT,
    SEPARATOR_WIDTH,
    METRIC_DECIMAL_PLACES,
    OUTPUT_ENABLE_JSON,
    OUTPUT_ENABLE_CSV,
    OUTPUT_FOLDER,
)


class ResultsPresenter:
    """Handles displaying and saving test results"""

    @staticmethod
    def _ensure_output_folder():
        """Create output folder if it doesn't exist"""
        Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_output_path(filename: str) -> str:
        """Get the full path for an output file"""
        ResultsPresenter._ensure_output_folder()
        return os.path.join(OUTPUT_FOLDER, filename)

    @staticmethod
    def display_metrics(metrics: Metrics):
        """Display metrics to console"""
        if not metrics:
            print("\nNo results to display metrics")
            return

        dp = METRIC_DECIMAL_PLACES
        sep = "=" * SEPARATOR_WIDTH

        print(f"\n{sep}")
        print("TEST RESULTS SUMMARY")
        print(f"{sep}\n")

        print(f"Total Samples:    {metrics['total']}")
        print(f"Correct:          {metrics['correct']}")
        print(f"Incorrect:        {metrics['incorrect']}")
        print(f"Accuracy:         {metrics['accuracy']:.{dp}f}%")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        print(f"\nMetrics:")
        print(f"  Precision:       {metrics['precision']:.{dp}f}%")
        print(f"  Recall:          {metrics['recall']:.{dp}f}%")
        print(f"  F1-Score:        {metrics['f1_score']:.{dp}f}%")

    @staticmethod
    def display_misclassified(results: List[TestResult]):
        """Display misclassified examples to console"""
        misclassified = [r for r in results if not r["correct"]]

        if not misclassified:
            return

        sep = "=" * SEPARATOR_WIDTH
        print(f"\n{sep}")
        print(f"MISCLASSIFIED EXAMPLES ({len(misclassified)}):")
        print(f"{sep}\n")

        for idx, example in enumerate(misclassified[:MISCLASSIFIED_DISPLAY_LIMIT], 1):
            msg = example["message"]
            truncated = f"{msg[:80]}..." if len(msg) > 80 else msg
            print(f"{idx}. {truncated}")
            print(f"   True: {example['true_toxic']} | Predicted: {example['predicted_toxic']}\n")

        if len(misclassified) > MISCLASSIFIED_DISPLAY_LIMIT:
            print(f"... and {len(misclassified) - MISCLASSIFIED_DISPLAY_LIMIT} more\n")

    @staticmethod
    def save_results(dataset_name: str, results: List[TestResult], metrics: Metrics, timestamp: str):
        """Save results to JSON, CSV, and TXT files"""
        ResultsPresenter._save_json(dataset_name, results, timestamp)
        ResultsPresenter._save_csv(dataset_name, results, timestamp)
        ResultsPresenter._save_metrics(dataset_name, metrics, timestamp)

    @staticmethod
    def _save_json(dataset_name: str, results: List[TestResult], timestamp: str):
        """Save detailed results as JSON"""
        filename = f"test_results_{dataset_name}_{timestamp}.json"
        filepath = ResultsPresenter._get_output_path(filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {filepath}")

    @staticmethod
    def _save_csv(dataset_name: str, results: List[TestResult], timestamp: str):
        """Save results as CSV"""
        filename = f"test_results_{dataset_name}_{timestamp}.csv"
        filepath = ResultsPresenter._get_output_path(filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("message,true_toxic,true_score,predicted_toxic,correct\n")
            for r in results:
                message = r["message"].replace('"', '""').replace('\r\n', '\n').replace('\r', '\n')
                f.write(
                    f'"{message}",{r["true_toxic"]},{r.get("true_score", "")}'
                    f',{r["predicted_toxic"]},{r["correct"]}\n'
                )
        print(f"CSV results saved to: {filepath}")

    @staticmethod
    def save_merged_results(results: List[TestResult], timestamp: str):
        """Save merged results from all datasets"""
        if OUTPUT_ENABLE_JSON:
            ResultsPresenter._save_merged_json(results, timestamp)
        if OUTPUT_ENABLE_CSV:
            ResultsPresenter._save_merged_csv(results, timestamp)

    @staticmethod
    def _save_merged_json(results: List[TestResult], timestamp: str):
        """Save all results merged as JSON with dataset field"""
        filename = f"test_results_merged_{timestamp}.json"
        filepath = ResultsPresenter._get_output_path(filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nMerged results saved to: {filepath}")

    @staticmethod
    def _save_merged_csv(results: List[TestResult], timestamp: str):
        """Save all results merged as CSV with dataset column"""
        filename = f"test_results_merged_{timestamp}.csv"
        filepath = ResultsPresenter._get_output_path(filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("dataset,message,true_toxic,true_score,predicted_toxic,correct\n")
            for r in results:
                message = r["message"].replace('"', '""').replace('\r\n', '\n').replace('\r', '\n')
                dataset = r.get("dataset", "unknown").replace('"', '""')
                f.write(
                    f'"{dataset}","{message}",{r["true_toxic"]},{r.get("true_score", "")}'
                    f',{r["predicted_toxic"]},{r["correct"]}\n'
                )
        print(f"Merged CSV results saved to: {filepath}")

    @staticmethod
    def save_summary(dataset_metrics: Dict[str, Metrics], timestamp: str):
        """Save summary metrics as markdown"""
        filename = f"test_summary_{timestamp}.md"
        filepath = ResultsPresenter._get_output_path(filename)
        dp = METRIC_DECIMAL_PLACES

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# Toxicity Model Test Results Summary\n\n")

            # Table 1: Total Samples, Correct, Incorrect, Accuracy
            f.write("## Overview Metrics\n\n")
            f.write("| Dataset | Total Samples | Correct | Incorrect | Accuracy |\n")
            f.write("|---------|---------------|---------|-----------|----------|\n")
            for dataset_name, metrics in dataset_metrics.items():
                if not metrics:
                    f.write(f"| {dataset_name} | 0 | 0 | 0 | 0.00% |\n")
                else:
                    f.write(
                        f"| {dataset_name} | {metrics['total']} | {metrics['correct']} | "
                        f"{metrics['incorrect']} | {metrics['accuracy']:.{dp}f}% |\n"
                    )
            f.write("\n")

            # Table 2: Confusion Matrix
            f.write("## Confusion Matrix\n\n")
            f.write("| Dataset | True Positives | False Positives | True Negatives | False Negatives |\n")
            f.write("|---------|----------------|-----------------|----------------|---------------|\n")
            for dataset_name, metrics in dataset_metrics.items():
                if not metrics:
                    f.write(f"| {dataset_name} | 0 | 0 | 0 | 0 |\n")
                else:
                    f.write(
                        f"| {dataset_name} | {metrics['true_positives']} | {metrics['false_positives']} | "
                        f"{metrics['true_negatives']} | {metrics['false_negatives']} |\n"
                    )
            f.write("\n")

            # Table 3: Precision, Recall, F1-Score
            f.write("## Classification Metrics\n\n")
            f.write("| Dataset | Precision | Recall | F1-Score |\n")
            f.write("|---------|-----------|--------|----------|\n")
            for dataset_name, metrics in dataset_metrics.items():
                if not metrics:
                    f.write(f"| {dataset_name} | 0.00% | 0.00% | 0.00% |\n")
                else:
                    f.write(
                        f"| {dataset_name} | {metrics['precision']:.{dp}f}% | {metrics['recall']:.{dp}f}% | "
                        f"{metrics['f1_score']:.{dp}f}% |\n"
                    )
            f.write("\n")

            # Explanations
            f.write("### Metric Explanations\n\n")
            f.write("**Precision**: Of all the samples predicted as toxic, how many were actually toxic?\n")
            f.write("- Formula: TP / (TP + FP)\n")
            f.write("- High precision means fewer false positives\n\n")
            f.write("**Recall**: Of all the actually toxic samples, how many did the model correctly identify?\n")
            f.write("- Formula: TP / (TP + FN)\n")
            f.write("- High recall means fewer false negatives\n\n")
            f.write("**F1-Score**: The harmonic mean of precision and recall, balancing both metrics.\n")
            f.write("- Formula: 2 * (Precision * Recall) / (Precision + Recall)\n")
            f.write("- Useful when you need a single metric that considers both precision and recall\n")

        print(f"Summary saved to: {filepath}")
