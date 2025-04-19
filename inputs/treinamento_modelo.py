import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from azureml.core import Workspace, Experiment

# Configurar MLflow
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
experiment = Experiment(workspace=ws, name="livraria-vendas-prediction")
mlflow.start_run()

# Carregar dados
dataset = Dataset.get_by_name(ws, name='vendas_livros')
df = dataset.to_pandas_dataframe()

# Pré-processamento
X = df[['Mes', 'Titulo', 'Assunto']]
y = df['Vendas']

# Transformação de dados
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Titulo', 'Assunto'])
    ],
    remainder='passthrough'
)

# Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
model.fit(X_train, y_train)

# Avaliar
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')

# Log métricas e modelo
mlflow.log_metric("MAE", mae)
mlflow.log_metric("MSE", mse)
mlflow.sklearn.log_model(model, "model")
mlflow.end_run()
