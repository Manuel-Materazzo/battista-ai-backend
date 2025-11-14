# Toxicity Model Benchmark

## Quick Start

1. **Prepare datasets** in the `datasets/` folder
   - Each JSON file should contain an array of objects with: `message`, `score`, `toxic` fields
   - The script automatically loads all .json files from this folder

2. **Start the API** on http://localhost:8000/v1/moderate

3. **Run the benchmark:**
   ```bash
   python toxicity-benchmark.py
   ```

## Configuration

Edit `toxicity_benchmark/config.py` to customize:

**API Settings:**
- `DEFAULT_API_URL`: API endpoint (default: "http://localhost:8000/v1/moderate")
- `API_TIMEOUT`: Request timeout in seconds (default: 10)
- `API_RATE_LIMIT_DELAY`: Delay between requests in seconds (default: 0.1)

**Data & Output:**
- `DEFAULT_DATA_FOLDER`: Folder with JSON datasets (default: "./datasets")
- `OUTPUT_FOLDER`: Output directory (default: "./output")
- `TIMESTAMP_FORMAT`: Output file timestamp format (default: "%Y%m%d_%H%M%S")

**Output Format:**
- `OUTPUT_ENABLE_JSON`: Save JSON results (default: True)
- `OUTPUT_ENABLE_CSV`: Save CSV results (default: True)

**Display:**
- `MISCLASSIFIED_DISPLAY_LIMIT`: Show first N misclassified examples (default: 10)
- `METRIC_DECIMAL_PLACES`: Decimal precision for metrics (default: 2)

## Output Files

Generated in the `output/` folder with timestamps:
1. **test_results_merged_{timestamp}.json** - All results combined
2. **test_results_merged_{timestamp}.csv** - Merged CSV results
3. **test_summary_{timestamp}.md** - Summary metrics in markdown format

## Metrics

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of predicted toxic messages that are actually toxic
- **Recall**: Proportion of actual toxic messages that were detected
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Breakdown of TP, FP, TN, FN
