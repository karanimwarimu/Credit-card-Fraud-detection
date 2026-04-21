
# this module is responsible for receiving transaction data, sending it to the fraud detection API, and returning the predictions to the caller.

# fraud_client/run.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


import json
import argparse
from fraud_client import score_transactions

from _utilities.test_fraud_api import generate_transactions 
from _utilities.logging_setup import setup_logging
from pathlib import Path
from datetime import datetime, timedelta


logging = setup_logging()

# ── Display ────────────────────────────────────────────────────
def display_results(results: list[dict]) -> None:
    print(f"\n{'─'*55}")
    print(f"  {'TXN ID':<15} {'PROB':>7}  {'DECISION':<12} {'FLAG'}")
    print(f"{'─'*55}")
    for r in results:
        txn_id   = r.get("transaction_id") or "N/A"
        prob     = r["fraud_probability"]
        decision = r["decision"]
        flag     = "*** FRAUD ***" if r["is_fraud"] else ""
        print(f"  {txn_id:<15} {prob:>7.4f}  {decision:<12}  {flag}")
    print(f"{'─'*55}\n")



# ── Entry point ────────────────────────────────────────────────

if __name__ == "__main__":
  

    # ── score and display ──
    
    #start time
    start_time = datetime.now()
    transactions = generate_transactions(22) # generate 5 synthetic transactions for testing
    print(f"ALL RANSACTIONS GENERATED INCLUDE  : " , transactions)
    results = score_transactions(transactions)
    
  
    display_results(results)
    end_time = datetime.now()
    logging.info(f"Processed {len(transactions)} transactions in {(end_time - start_time).total_seconds():.2f} seconds.")
    print(f"Processing time: {end_time - start_time}")