"""Configuration for the toxicity benchmark"""

# API Configuration
DEFAULT_API_URL = "http://localhost:8000/v1/moderate"
API_TIMEOUT = 10
API_RATE_LIMIT_DELAY = 0.1  # seconds between requests

# Data Configuration
DEFAULT_DATA_FOLDER = "./datasets"

# Output Configuration
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
MISCLASSIFIED_DISPLAY_LIMIT = 10
SEPARATOR_WIDTH = 60
OUTPUT_FOLDER = "./output"

# Output Format Toggles
OUTPUT_ENABLE_JSON = True
OUTPUT_ENABLE_CSV = True

# Metric Precision
METRIC_DECIMAL_PLACES = 2
