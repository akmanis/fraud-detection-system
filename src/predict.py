import numpy as np
import pickle

model = pickle.load(open("models/fraud_model.pkl", "rb"))

def predict_transaction(amount):

    features = np.random.normal(0,1,(1,30))
    features[0][-1] = amount

    probability = model.predict_proba(features)[0][1]

    return probability
