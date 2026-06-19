import joblib
import numpy as np
import pandas as pd
from pathlib import Path


MODEL_PATH = Path("data/models/ConstantMedian.pkl")
model = joblib.load(MODEL_PATH)

# TODO 
# from pricing_amazon_products.inference import predict_price

def predict_price(catalog_content: str):
    if not catalog_content or not catalog_content.strip():
        return "Please enter a product description."

    features = [[0]]

    pred_log_price = model.predict(features)[0]

    pred_price = float(np.exp(pred_log_price))

    return round(pred_price, 2)