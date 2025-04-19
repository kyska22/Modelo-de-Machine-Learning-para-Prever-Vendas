import json
import pandas as pd
import numpy as np
import mlflow.sklearn
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('livraria-vendas-model')
    model = mlflow.sklearn.load_model(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        input_data = pd.DataFrame(data)
        predictions = model.predict(input_data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        error = str(e)
        return {"error": error}
