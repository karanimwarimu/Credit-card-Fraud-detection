import numpy as np


# function to generate synthetic transactions with a mix of normal, suspicious, and fraud-like patterns. 
# This will help us test our batch prediction function.  

def generate_transactions(n=50, seed=42):
    np.random.seed(seed)
    transactions = []

    for i in range(n):
        if i < 20:
            # 🟢 Normal
            scale = np.random.uniform(0.01, 0.3)
            amount = np.random.uniform(1, 50)

        elif i < 35:
            # 🟡 Suspicious
            scale = np.random.uniform(0.3, 1.5)
            amount = np.random.uniform(50, 500)

        else:
            # 🔴 Fraud-like
            scale = np.random.uniform(1.5, 6)
            amount = np.random.uniform(500, 5000)

        tx = {
            "Time": float(np.random.uniform(100000, 200000)),
            "Amount": float(amount)
        }

        # Generate V1–V28 with randomness + asymmetry
        for j in range(1, 29):
            val = np.random.normal(0, scale)

            # inject occasional extreme spikes (fraud signal)
            if i >= 35 and np.random.rand() > 0.7:
                val *= np.random.uniform(2, 5)

            tx[f"V{j}"] = float(val)

        transactions.append(tx)

    return transactions




# smooth test transactions used earlier for second test 

#In this example lets use several transactions with varying probabilities to see how the decision logic works.

transactions = {
    "transaction_1": { "Time": 123456.0 , "V1": -1.0, "V2": 0.5, "V3": 0.3, "V4": -0.2, "V5": 0.1, "V6": -0.1, "V7": 0.2, "V8": -0.3, "V9": 0.4, "V10": -0.4, "V11": 0.5, "V12": -0.5, "V13": 0.6, "V14": -0.6, "V15": 0.7, "V16": -0.7, "V17": 0.8, "V18": -0.8, "V19": 0.9, "V20": -0.9, "V21": 1.0, "V22": -1.0, "V23": 1.1, "V24": -1.1, "V25": 1.2, "V26": -1.2, "V27": 1.3, "V28": -1.3, "Amount": 100.0 },
    # Add more transactions with varying values to test the decision logic
    "transaction_2": { "Time": 123456.0 , "V1": -0.5, "V2": 0.3, "V3": 0.1, "V4": -0.1, "V5": 0.05, "V6": -0.05, "V7": 0.1, "V8": -0.15, "V9": 0.2, "V10": -0.2, "V11": 0.25, "V12": -0.25, "V13": 0.3, "V14": -0.3, "V15": 0.35, "V16": -0.35, "V17": 0.4, "V18": -0.4, "V19": 0.45, "V20": -0.45, "V21": 0.5, "V22": -0.5, "V23": 0.55, "V24": -0.55, "V25": 0.6, "V26": -0.6, "V27": 0.65, "V28": -0.65, "Amount": 50.0 },
    "transaction_3": { "Time": 123456.0 , "V1": -0.2, "V2": 0.1, "V3": 0.05, "V4": -0.05, "V5": 0.02, "V6": -0.02, "V7": 0.05, "V8": -0.1, "V9": 0.15, "V10": -0.15, "V11": 0.2, "V12": -0.2, "V13": 0.25, "V14": -0.25, "V15": 0.3, "V16": -0.3, "V17": 0.35, "V18": -0.35, "V19": 0.4, "V20": -0.4, "V21": 0.45, "V22": -0.45, "V23": 0.5, "V24": -0.5, "V25": 0.55, "V26": -0.55, "V27": 0.6, "V28": -0.6, "Amount": 20.0 },
    "transaction_4": { "Time": 123456.0 , "V1": -0.1, "V2": 0.05, "V3": 0.02, "V4": -0.02, "V5": 0.01, "V6": -0.01, "V7": 0.02, "V8": -0.04, "V9": 0.08, "V10": -0.08, "V11": 0.1, "V12": -0.1, "V13": 0.15, "V14": -0.15, "V15": 0.2, "V16": -0.2, "V17": 0.25, "V18": -0.25, "V19": 0.3, "V20": -0.3, "V21": 0.35, "V22": -0.35, "V23": 0.4, "V24": -0.4, "V25": 0.45, "V26": -0.45, "V27": 0.5, "V28": -0.5, "Amount": 10.0 }
}


# used for first test, to check if the batch prediction function can handle ONE TRANSACTION and return results in the expected format.  

new_transaction = { "Time": 123456.0 , "V1": -1.0, "V2": 0.5, "V3": 0.3, "V4": -0.2, "V5": 0.1, "V6": -0.1, "V7": 0.2, "V8": -0.3, "V9": 0.4, "V10": -0.4, "V11": 0.5, "V12": -0.5, "V13": 0.6, "V14": -0.6, "V15": 0.7, "V16": -0.7, "V17": 0.8, "V18": -0.8, "V19": 0.9, "V20": -0.9, "V21": 1.0, "V22": -1.0, "V23": 1.1, "V24": -1.1, "V25": 1.2, "V26": -1.2, "V27": 1.3, "V28": -1.3, "Amount": 100.0 }
  