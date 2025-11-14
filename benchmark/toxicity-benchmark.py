#!/usr/bin/env python
"""
Toxicity Model Benchmark - Entry point script

This script runs a comprehensive benchmark of the toxicity model,
testing it against datasets and generating detailed metrics and reports.
"""

from toxicity_benchmark import ToxicityModelTester
from toxicity_benchmark.config import DEFAULT_API_URL, DEFAULT_DATA_FOLDER, SEPARATOR_WIDTH


def main():
    """Entry point for the application"""
    print("=" * SEPARATOR_WIDTH)
    print("TOXICITY MODEL TESTER")
    print("=" * SEPARATOR_WIDTH)
    print(f"\nAPI Endpoint: {DEFAULT_API_URL}")
    print(f"Data Folder: {DEFAULT_DATA_FOLDER}\n")

    tester = ToxicityModelTester(api_url=DEFAULT_API_URL, data_folder=DEFAULT_DATA_FOLDER)
    tester.run_tests()

    print(f"\n{'='*SEPARATOR_WIDTH}")
    print("Testing complete!")
    print(f"{'='*SEPARATOR_WIDTH}")


if __name__ == "__main__":
    main()
