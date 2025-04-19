## 📝 Exemplo de Código (Snippet)

```python
# Configuração do AutoML
automl_settings = {
    "task": 'regression',
    "primary_metric": 'normalized_root_mean_squared_error',
    "training_data": train_data,
    "label_column_name": 'Vendas',
    "n_cross_validations": 5,
    "compute_target": compute_target,
    "experiment_timeout_hours": 0.5
}
```
