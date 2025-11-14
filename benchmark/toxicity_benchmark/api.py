"""API client for communicating with the toxicity moderation service"""

import requests
from typing import Optional, Dict

from .config import API_TIMEOUT


class APIClient:
    """Handles API communication with the toxicity moderation service"""

    def __init__(self, api_url: str, timeout: int = API_TIMEOUT):
        """
        Initialize API client.

        Args:
            api_url: Base URL of the moderation API
            timeout: Request timeout in seconds
        """
        self.api_url = api_url
        self.timeout = timeout

    def moderate(self, message: str) -> Optional[Dict]:
        """
        Send a message to the moderation API.

        Args:
            message: Text to moderate

        Returns:
            API response dict or None if error occurred
        """
        try:
            response = requests.post(
                self.api_url, json={"prompt": message}, timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"  ✗ API Error: {e}")
            return None
