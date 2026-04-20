#pydantic validation for the input data to the API

from pydantic import BaseModel #for data validation
from typing import Optional, Union #for optional fields


''' 
Index(['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
       'Class'],
      dtype='object')# these are the columns in the dataset, we will use them as fields in the Transaction model
'''

class Transaction(BaseModel):
    Time: Optional[float] = None #optional field, not all transactions may have this field
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    
    
class PredictionResponse(BaseModel):
    fraud_probability: float
    prediction : int # 0 for non-fraud, 1 for fraud
    decision: str # "Fraud" or "Not Fraud"
    is_fraud: bool # True if fraud, False if not fraud
    
    
class BatchRequest(BaseModel):
     # Union means: accept ONE Transaction object OR a list of them
    transactions: Union[Transaction, list[Transaction]]
    
# we can also define a response model for batch predictions if needed
class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    
    
# WHAT IS HAPPENNING HERE IS THAT WE ARE DEFINING THE SCHEMAS FOR THE API, THIS WILL HELP US TO VALIDATE THE INPUT DATA AND ALSO TO STRUCTURE THE OUTPUT DATA IN A CONSISTENT WAY.  

    