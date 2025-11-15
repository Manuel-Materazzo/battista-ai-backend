"""Main test orchestration logic"""

import time
from datetime import datetime
from typing import List, Dict

from .api import APIClient
from .loader import DatasetLoader
from .metrics import MetricsCalculator
from .presenter import ResultsPresenter
from .models import TestResult, TestSample
from .config import (
    DEFAULT_API_URL,
    DEFAULT_DATA_FOLDER,
    SAMPLES_PER_DATASET_LIMIT,
    TIMESTAMP_FORMAT,
    API_RATE_LIMIT_DELAY,
    SEPARATOR_WIDTH,
)


class ToxicityModelTester:
    """Main test runner orchestrating the testing workflow"""

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        data_folder: str = DEFAULT_DATA_FOLDER,
        samples_limit: int = SAMPLES_PER_DATASET_LIMIT,
    ):
        """
        Initialize the tester.

        Args:
            api_url: URL of the moderation API endpoint
            data_folder: Path to folder containing dataset JSON files
            samples_limit: Maximum number of samples to test per dataset (None for no limit)
        """
        self.api_url = api_url
        self.data_folder = data_folder
        self.samples_limit = samples_limit
        self.all_results: List[TestResult] = []
        self.dataset_metrics: Dict[str, Dict] = {}
        self.api_client = APIClient(api_url)

    def run_tests(self):
        """Execute the full testing workflow"""
        datasets = DatasetLoader.load_datasets(self.data_folder)

        if not datasets:
            print("No data to test!")
            return

        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        for dataset_name, samples in datasets.items():
            self._run_test_suite(dataset_name, samples, timestamp)
        
        # Save merged results and summary after all datasets are tested
        if self.all_results:
            ResultsPresenter.save_merged_results(self.all_results, timestamp)
            ResultsPresenter.save_summary(self.dataset_metrics, timestamp)

    def _run_test_suite(self, dataset_name: str, samples: List[TestSample], timestamp: str):
        """
        Run tests on a single dataset's samples and save results.
        
        Args:
            dataset_name: Name of the dataset
            samples: Samples to test
            timestamp: Timestamp for output filenames
        """
        # Apply limit if configured
        limited_samples = samples
        if self.samples_limit is not None:
            limited_samples = samples[:self.samples_limit]
        
        sep = "=" * SEPARATOR_WIDTH
        print(f"\n{sep}")
        print(f"Testing dataset: {dataset_name}")
        total_samples = len(limited_samples)
        print(f"Starting tests on {total_samples} samples...")
        if self.samples_limit is not None and len(samples) > self.samples_limit:
            print(f"(Limited from {len(samples)} total samples)")
        print(f"{sep}\n")

        results: List[TestResult] = []
        for idx, sample in enumerate(limited_samples, 1):
            result = self._test_sample(idx, total_samples, sample, dataset_name)
            if result:
                results.append(result)
            time.sleep(API_RATE_LIMIT_DELAY)

        # Display summary for this dataset
        metrics = MetricsCalculator.calculate_metrics(results)
        ResultsPresenter.display_metrics(metrics)
        ResultsPresenter.display_misclassified(results)
        
        self.dataset_metrics[dataset_name] = metrics
        self.all_results.extend(results)

    def _test_sample(self, idx: int, total: int, sample: TestSample, dataset_name: str) -> TestResult:
        """
        Test a single sample.

        Args:
            idx: Current sample index (1-based)
            total: Total number of samples
            sample: Sample to test
            dataset_name: Name of the dataset this sample comes from

        Returns:
            TestResult or None if API error
        """
        message = sample.get("message", "")
        true_toxic = sample.get("toxic", False)
        true_score = sample.get("score", None)

        display_msg = message[:50] + ("..." if len(message) > 50 else "")
        print(f"[{idx}/{total}] Testing: {display_msg}")

        prediction = self.api_client.moderate(message)

        if prediction:
            pred_toxic = prediction.get("toxic", False)
            is_correct = pred_toxic == true_toxic

            result: TestResult = {
                "message": message,
                "true_toxic": true_toxic,
                "true_score": true_score,
                "predicted_toxic": pred_toxic,
                "correct": is_correct,
                "dataset": dataset_name,
            }

            status = "✓ CORRECT" if is_correct else "✗ WRONG"
            print(f"  True: {true_toxic} | Predicted: {pred_toxic} | {status}")
            return result
        else:
            print(f"  Skipping due to API error")
            return None


