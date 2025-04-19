# Projeto de Previsão de Vendas para Livraria Online - Azure Machine Learning

## Visão Geral do Projeto

Vou criar um projeto completo no Azure Machine Learning para prever vendas semanais de livros em uma livraria online, com foco em otimizar a produção e estoque durante a temporada de alta (Dezembro a Janeiro).

## 1. Configuração do Ambiente Azure

### 1.1 Grupo de Recursos
```bash
az group create --name RG-LivrariaML --location eastus
```

### 1.2 Workspace do Azure Machine Learning
```bash
az ml workspace create --name LivrariaML-Workspace --resource-group RG-LivrariaML --location eastus
```

## 2. Arquivos do Projeto

### 2.1 Arquivo de Dados (`vendas_livros.csv`)

```csv
Data,Mes,Titulo,Assunto,Vendas
2023-01-01,1,"Hábitos Atômicos","autoajuda",120
2023-01-01,1,"Azure Fundamentals","cloud",85
2023-01-01,1,"Python para Iniciantes","programação",150
2023-02-01,2,"Pai Rico, Pai Pobre","finanças",200
2023-02-01,2,"Introdução à IA","IA",90
2023-03-01,3,"O Poder do Agora","autoajuda",110
2023-03-01,3,"Investimentos Inteligentes","finanças",180
2023-04-01,4,"Aprendendo Django","programação",95
2023-04-01,4,"Machine Learning Básico","IA",120
2023-05-01,5,"O Milionário Automático","finanças",160
2023-05-01,5,"DevOps com Azure","cloud",75
2023-06-01,6,"O Poder do Hábito","autoajuda",130
2023-06-01,6,"Data Science com Python","programação",110
2023-07-01,7,"Mindset","autoajuda",140
2023-07-01,7,"Deep Learning","IA",85
2023-08-01,8,"Os Segredos da Mente Milionária","finanças",190
2023-08-01,8,"Azure AI Services","cloud",65
2023-09-01,9,"Python Avançado","programação",100
2023-09-01,9,"O Homem Mais Rico da Babilônia","finanças",210
2023-10-01,10,"Como Fazer Amigos e Influenciar Pessoas","autoajuda",180
2023-10-01,10,"Inteligência Artificial Aplicada","IA",95
2023-11-01,11,"Cloud Computing","cloud",80
2023-11-01,11,"Pandas para Análise de Dados","programação",120
2023-12-01,12,"O Alquimista","autoajuda",300
2023-12-01,12,"Presentes Financeiros para seu Filho","finanças",350
2023-12-01,12,"Azure Machine Learning","cloud",120
2023-12-01,12,"Aprendizado Profundo","IA",180
2023-12-01,12,"Flask Web Development","programação",150
2024-01-01,1,"O Poder da Ação","autoajuda",280
2024-01-01,1,"Os Axiomas de Zurique","finanças",320
2024-01-01,1,"TensorFlow para Iniciantes","IA",150
2024-01-01,1,"Docker para Desenvolvedores","programação",130
2024-02-01,2,"Rápido e Devagar","autoajuda",140
2024-02-01,2,"Os Segredos do ChatGPT","IA",110
```

### 2.2 Notebook de Análise Exploratória (`exploracao_dados.ipynb`)

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from azureml.core import Workspace, Dataset

# Conectar ao workspace
ws = Workspace.from_config()

# Carregar dados
dataset = Dataset.get_by_name(ws, name='vendas_livros')
df = dataset.to_pandas_dataframe()

# Análise exploratória
print(df.head())
print(df.info())
print(df.describe())

# Gráfico de vendas por mês
plt.figure(figsize=(12,6))
sns.lineplot(x='Mes', y='Vendas', hue='Assunto', data=df, ci=None)
plt.title('Vendas Mensais por Assunto')
plt.xlabel('Mês')
plt.ylabel('Vendas')
plt.grid(True)
plt.show()

# Gráfico de vendas por assunto
plt.figure(figsize=(10,6))
sns.boxplot(x='Assunto', y='Vendas', data=df)
plt.title('Distribuição de Vendas por Assunto')
plt.xticks(rotation=45)
plt.show()

# Vendas médias por mês
vendas_mes = df.groupby('Mes')['Vendas'].mean().reset_index()
plt.figure(figsize=(12,6))
sns.barplot(x='Mes', y='Vendas', data=vendas_mes)
plt.title('Vendas Médias por Mês')
plt.xlabel('Mês')
plt.ylabel('Vendas Médias')
plt.show()
```

### 2.3 Script de Treinamento (`treinamento_modelo.py`)

```python
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
```

### 2.4 Script de Inferência (`score.py`)

```python
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
```

## 3. Implementação no Azure ML

### 3.1 Criar Data Asset
1. No Azure ML Studio, vá para "Data"
2. Criar novo Data Asset com o arquivo `vendas_livros.csv`
3. Nomear como "vendas_livros"

### 3.2 Criar Compute Cluster
1. Vá para "Compute"
2. Criar novo Compute Cluster:
   - Nome: "cpu-cluster"
   - Tipo de máquina: Standard_DS3_v2
   - Número mínimo de nós: 0
   - Número máximo de nós: 4

### 3.3 Executar Automated ML
1. Criar novo experimento Automated ML
2. Selecionar dataset "vendas_livros"
3. Configurar:
   - Nome do experimento: "automl-vendas-livros"
   - Coluna alvo: "Vendas"
   - Tipo de tarefa: Regressão
   - Selecionar compute cluster criado
4. Definir configurações adicionais:
   - Métrica primária: Normalized root mean squared error
   - Tempo máximo: 60 minutos
   - Validação cruzada: 5 folds

### 3.4 Pipeline no Designer

1. Criar novo pipeline no Designer
2. Adicionar os seguintes módulos:

```
Dataset -> Split Data -> Normalize Data -> Train Model -> Score Model -> Evaluate Model
```

3. Configurar cada módulo:
   - Dataset: Selecionar "vendas_livros"
   - Split Data: 70% treino, 30% teste
   - Normalize Data: Método ZScore
   - Train Model: Algoritmo de Regressão (Boosted Decision Tree)
   - Score Model: Usar dados de teste
   - Evaluate Model: Métricas de regressão

### 3.5 Implantação do Modelo

1. Registrar o melhor modelo encontrado (Automated ML ou Designer)
2. Criar um endpoint em tempo real:
   - Selecionar "Deploy"
   - Nome: "livraria-vendas-endpoint"
   - Tipo de computação: Azure Container Instance
   - Usar o script `score.py` para inferência

## 4. Monitoramento e Consumo

### 4.1 Testar o Endpoint
```python
import requests
import json

url = "SEU_ENDPOINT_URL"
key = "SUA_CHAVE_API"

data = {
    "data": [
        {"Mes": 12, "Titulo": "Azure Machine Learning", "Assunto": "cloud"},
        {"Mes": 1, "Titulo": "O Poder da Ação", "Assunto": "autoajuda"}
    ]
}

headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'}
response = requests.post(url, json=data, headers=headers)
print(response.json())
```

### 4.2 Monitoramento no Azure ML
1. Acessar o endpoint no Azure ML Studio
2. Verificar métricas de desempenho
3. Configurar alertas para degradação de desempenho

## Conclusão

Este projeto completo no Azure ML permite:
- Prever vendas semanais de livros por assunto/título
- Otimizar a produção e estoque
- Reduzir desperdícios
- Garantir disponibilidade durante a temporada de alta

O modelo pode ser refinado com mais dados históricos e características adicionais como promoções, eventos sazonais, etc.
