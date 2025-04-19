# README - Projeto de Previsão de Vendas para Livraria Online

![Azure Machine Learning](https://img.shields.io/badge/Azure-Machine%20Learning-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0.2-orange)
![MLflow](https://img.shields.io/badge/MLflow-1.26.1-lightgrey)

## 📌 Visão Geral

Este projeto utiliza Azure Machine Learning para prever vendas semanais de livros em uma livraria online, com foco em otimizar a produção e estoque durante a temporada de alta (Dezembro a Janeiro).

**Objetivos Principais:**
- Prever demanda semanal por título/assunto
- Otimizar processos de impressão e estoque
- Reduzir desperdícios e custos operacionais
- Garantir disponibilidade dos títulos mais vendidos

## 🛠️ Tecnologias Utilizadas

- **Azure Machine Learning**
- **Python 3.8+**
- **Scikit-learn**
- **MLflow**
- **Pandas/Numpy**
- **Matplotlib/Seaborn**

## 📂 Estrutura do Projeto

```
livraria-ml-project/
│
├── data/
│   └── vendas_livros.csv          # Dataset histórico de vendas
│
├── notebooks/
│   └── exploracao_dados.ipynb     # Análise exploratória dos dados
│
├── scripts/
│   ├── treinamento_modelo.py      # Script de treinamento do modelo
│   └── score.py                   # Script para inferência do modelo
│
├── azure/
│   ├── create_resources.sh        # Script para criar recursos Azure
│   └── pipeline_designer.json     # Export do pipeline do Designer
│
└── README.md                      # Este arquivo
```

## 🚀 Como Executar o Projeto

### Pré-requisitos

- Conta Azure com acesso ao Machine Learning
- Azure CLI instalado e configurado
- Python 3.8 ou superior

### Configuração Inicial

1. Criar recursos no Azure:
```bash
az group create --name RG-LivrariaML --location eastus
az ml workspace create --name LivrariaML-Workspace --resource-group RG-LivrariaML --location eastus
```

2. Configurar ambiente Python:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac)
venv\Scripts\activate     # Windows)
pip install -r requirements.txt
```

### Executando o Projeto

1. Carregue o dataset no Azure ML Studio
2. Execute o notebook `exploracao_dados.ipynb` para análise inicial
3. Treine o modelo com:
```bash
python scripts/treinamento_modelo.py
```
4. Siga os passos no Azure ML Studio para implantar o modelo

## 📊 Resultados Esperados

O modelo deve prever com precisão as vendas semanais por:
- Título do livro
- Assunto (autoajuda, tecnologia, finanças)
- Período do ano (especialmente Dez-Jan)

**Métricas de Desempenho:**
- MAE (Mean Absolute Error): < 20 unidades
- MSE (Mean Squared Error): < 500

## 📈 Gráficos de Exemplo

![Vendas Mensais por Assunto](https://via.placeholder.com/600x400?text=Vendas+Mensais+por+Assunto)
![Distribuição de Vendas por Assunto](https://via.placeholder.com/600x400?text=Distribuição+de+Vendas+por+Assunto)

## 🤝 Contribuição

Contribuições são bem-vindas! Siga os passos:

1. Faça um fork do projeto
2. Crie uma branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

## ✉️ Contato

Equipe de Data Science - contato@livrariaonline.com.br

---

**Nota:** Este projeto foi desenvolvido como parte do laboratório da DIO (Digital Innovation One) sobre Azure Machine Learning.
