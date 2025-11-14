"""Dataset loading utilities"""

import json
from pathlib import Path
from typing import List, Dict

from .models import TestSample


class DatasetLoader:
    """Handles loading datasets from JSON files"""

    @staticmethod
    def load_datasets(data_folder: str) -> Dict[str, List[TestSample]]:
        """
        Load all JSON datasets from the specified folder.

        Args:
            data_folder: Path to folder containing JSON dataset files

        Returns:
            Dict mapping filename to list of test samples
        """
        datasets = {}
        data_path = Path(data_folder)

        if not data_path.exists():
            print(f"Warning: Folder '{data_folder}' does not exist!")
            return datasets

        json_files = list(data_path.glob("*.json"))
        print(f"Found {len(json_files)} JSON file(s) in '{data_folder}'")

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        datasets[json_file.stem] = data
                    else:
                        datasets[json_file.stem] = [data]
                    count = len(data) if isinstance(data, list) else 1
                    print(f"  ✓ Loaded {json_file.name}: {count} samples")
            except Exception as e:
                print(f"  ✗ Error loading {json_file.name}: {e}")

        return datasets
