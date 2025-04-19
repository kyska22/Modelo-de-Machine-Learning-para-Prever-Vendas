## 📝 Checklist de Implementação

### Pré-processamento:
- [ ] Codificar variáveis categóricas (Título, Assunto)
- [ ] Normalizar dados numéricos
- [ ] Dividir dados (70/30)

### Modelagem:
- [ ] Testar algoritmos:
  - Random Forest
  - Gradient Boosting
  - Regressão Linear
- [ ] Configurar MLflow para tracking

### Implantação:
- [ ] Criar endpoint ACI (Azure Container Instance)
- [ ] Testar API com payload de exemplo:
```json
{
  "data": [
    {"Mes": 12, "Titulo": "Azure ML", "Assunto": "cloud"},
    {"Mes": 1, "Titulo": "Hábitos Atômicos", "Assunto": "autoajuda"}
  ]
}
```
