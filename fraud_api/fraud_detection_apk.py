""" 
FiLe: fraud_detection_apk.py
Author:  Ian karani
Purpose : Load the model and make predictions on new data.
 
 """  
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from pathlib import Path
from logging import INFO
from test_fraud_api import generate_transactions


# Get the path of the current file
current_dir = Path(__file__).resolve().parent

# Go up one level, then into 'models', then find the file
model_path = current_dir.parent / "models" / "bestlogreg_fraud_model.pkl"

# Load the trained model and its artifacts (like feature columns and threshold)
_artifacts = joblib.load(model_path)

Model = _artifacts["model"]
threshold = _artifacts["threshold"]
feature_columns = _artifacts["feature_columns"]


# Define a function to mAKE THE DECISION BASED ON THE MODEL'S PREDICTION

def _decide(pred_prob: float) -> (int, str, bool): # type: ignore # returns the predicted class, decision string, and boolean flag for fraud
    if pred_prob >= 0.9:  # if the predicted probability is very high, we can be more confident in labeling it as fraud
        return 1, "Fraud", True
    elif pred_prob >= threshold:
        return 1, "Rview", True
    else:
        return 0, "Not Fraud", False

# used for one transaction prediction endpoint, it takes a single transaction as input and returns the prediction result.

def predict_fraud(transaction_data: dict) -> dict:
    # Convert the input data to a DataFrame
    df = pd.DataFrame([transaction_data], columns=feature_columns)  # ensure correct column order
    
    # Make prediction using the loaded model
    pred_prob = Model.predict_proba(df)[0][1]  # get the probability of the positive class (fraud)
    
    # Get the decision based on the predicted probability
    pred_class, decision, is_fraud = _decide(pred_prob)
    
    # Return the prediction results in a structured format
    return {
        "fraud_probability": pred_prob,
        "prediction": pred_class,
        "decision": decision,
        "is_fraud": is_fraud
    }

# we use this instead for the batch prediction endpoint, it takes a list of transactions and returns a list of predictions for each transaction.

def predict_batch(transactions: list[dict]) -> list[dict]:
    # Convert list of transactions into DataFrame
    df = pd.DataFrame(transactions)

    # Enforce column order (critical in production)
    df = df[feature_columns]

    # Predict probabilities for all rows at once
    probs = Model.predict_proba(df)[:, 1]  # vector of fraud probabilities

    results = []

    for prob in probs:
        pred_class, decision, is_fraud = _decide(prob)

        results.append({
            "fraud_probability": float(prob),
            "prediction": pred_class,
            "decision": decision,
            "is_fraud": is_fraud
        })

    return results

# This is the optimized version of the prediction function that can handle both single transactions and batches of transactions.

def predict_batch_fast(transactions: list[dict]) -> list[dict]:
    df = pd.DataFrame(transactions)[feature_columns]
    probs = Model.predict_proba(df)[:, 1]

    # print the probabilities to see the distribution
    print(probs.min(), probs.max(), probs.mean())
    
    predictions = np.where(probs >= threshold, 1, 0)
    decisions = np.where(
        probs >= 0.9, "Fraud",
        np.where(probs >= threshold, "Review", "Not Fraud")
    )
    is_fraud_flags = probs >= threshold

    return [
        {
            "fraud_probability": float(p),
            "prediction": int(c),
            "decision": str(d),
            "is_fraud": bool(f)
        }
        for p, c, d, f in zip(probs, predictions, decisions, is_fraud_flags)
    ]
   
   
    
# what is happening here is that we are loading the model and its artifacts, then we define a function to make predictions on new transaction data. 
  # The function takes a dictionary of transaction data, converts it to a DataFrame, and uses the model to predict the probability of fraud. 
  # Based on the predicted probability, it makes a decision and returns the results in a structured format. 
  # The decision logic is designed to be more conservative, labeling transactions as "Review" if the probability is above the threshold but not high enough to be labeled as "Fraud".
  
# example usage of the predict_fraud function   :

#prediction_result = predict_fraud(new_transaction)

#decison = _decide(prediction_result["fraud_probability"])

#print('prediction result' , prediction_result)
#print('decision' , decison)



#use the predict_batch_fast function to get predictions for all transactions above and print the results


#print("Generated Transactions:" , transactions)

#print("\n\n")

#transactions_list = list(transactions.values())


"""
# generate the synthetic transactions and get predictions for the batch of transactions
transactions = generate_transactions(50)

# create a cv file for every list of transactions generated,
# to see the distribution of the transactions and the predicted probabilities, we can use this for further analysis and visualization.   

import pandas as pd
df = pd.DataFrame(transactions)
i = 0 

dataset_path = current_dir.parent / "Data_set" / f"synthetic_transactions{i}.csv"
df.to_csv(dataset_path, index=False)
print("Synthetic transactions saved to CSV file.")
i += 1
"""

#transactions = pd.read_csv(current_dir.parent / "Data_set" / "synthetic_transactions0.csv").to_dict(orient='records')

new_transaction = { "Time": 123456.0 , "V1": -1.0, "V2": 0.5, "V3": 0.3, "V4": -0.2, "V5": 0.1, "V6": -0.1, "V7": 0.2, "V8": -0.3, "V9": 0.4, "V10": -0.4, "V11": 0.5, "V12": -0.5, "V13": 0.6, "V14": -0.6, "V15": 0.7, "V16": -0.7, "V17": 0.8, "V18": -0.8, "V19": 0.9, "V20": -0.9, "V21": 1.0, "V22": -1.0, "V23": 1.1, "V24": -1.1, "V25": 1.2, "V26": -1.2, "V27": 1.3, "V28": -1.3, "Amount": 100.0 }

transactions = [new_transaction]

results = predict_batch_fast(transactions)

for i, res in enumerate(results, 1):
    print(f"Transaction {i}: {res}")

