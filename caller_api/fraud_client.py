# the client code that will call the fraud detection API endpoints, it can be used for testing the API and also for making actual predictions on
#   new transactions.

# well add the ui later, for now we can use this client code to test the API endpoints and see how the predictions are working.
# connect it to the  a model performance dashboard later to visualize the results and monitor the model's performance over time.


import httpx
import json
from typing import List, Dict, Union # for type hints


API_URL = "http://localhost:8000/predict"

# ── Core function ──────────────────────────────────────────────

def score_transactions(
    transactions: Union[dict, list[dict]]
) -> list[dict]:
    """
    The ONLY job of this function:
    - accept transaction(s)
    - send to API
    - return results
    
    It does NOT know where transactions came from.
    It does NOT print anything.
    It does NOT know what you will do with results.
    """
    payload = {"transactions": transactions}

    response = httpx.post(API_URL, json=payload, timeout=10.0)
    response.raise_for_status()
    return response.json()["predictions"]