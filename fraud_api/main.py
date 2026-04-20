# holds the api logic for the fraud detection model, including loading the model, making predictions, and handling batch predictions.


from pydantic import BaseModel
import joblib
from fastapi import FastAPI , HTTPException
from schemas import Transaction, PredictionResponse, BatchRequest, BatchPredictionResponse
from fraud_detection_apk import  predict_batch_fast
from pathlib import Path

#import logging
from _utilities.logging_setup import setup_logging

"""

#This module defines the FastAPI application for the Fraud Detection API. It includes endpoints for health checks

logger = logging.getLogger("uvicorn.error")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) # we can use this logger to log important information and debug messages in our API
"""

# using configured logger from logging_setup.py

logger = setup_logging()


app = FastAPI(title="Fraud Detection API", description="API for predicting fraudulent transactions using a trained model.", version="1.0")


# route to check if the API is running and healthy, we can use this endpoint to perform a simple health check.

@app.get("/health")

def health_check():
    return {"status": "ok", "message": "Fraud Detection API is healthy and ready to receive requests."}

# route to receive the transaction(s) data and return the prediction results, 
    # this endpoint can handle both single transactions and batches of transactions.
    
    
@app.post("/predict", response_model=list[BatchPredictionResponse]) # we expect a list of predictions in the response, even if it's just one transaction

async def predict(request: BatchRequest):
    try :
        logger.info(f"Received prediction request with {len(request.transactions) if isinstance(request.transactions, list) else 1} transaction(s).")
        txns =  request.transactions
        if isinstance (txns, Transaction):
            txns = [txns.dict()] # convert single transaction to list of one transaction
        
        raw = [txn.dict() for txn in txns] # convert list of Transaction objects to list of dicts
        predictions = predict_batch_fast(raw) # get predictions for the batch of transactions
        logger.info(f"Generated predictions for {len(predictions)} transaction(s).")
        responses = [
            PredictionResponse(
                transaction_id = r.get("transaction_id"),
                fraud_probability = r["fraud_probability"],
                prediction = r["prediction"],
                decision = "Fraud" if r["is_fraud"] else "Not Fraud",
                is_fraud = r["is_fraud"],
            )
            for r in predictions
          ]

        return BatchPredictionResponse(predictions=responses)
        logger.info(f"Returning prediction response for {len(responses)} transaction(s).")
        
    except ValueError as ve:
        logger.error(f"Value error processing prediction request: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))    
    except Exception as e:
        logger.error(f"Error processing prediction request: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the prediction request.")