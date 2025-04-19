## 📝 Guia de Configuração

### Pré-requisitos Azure:
1. **Acesso**: Conta Azure com permissões para:
   - Criar Machine Learning Workspace
   - Provisionar recursos de computação

2. **Configuração Inicial**:
```bash
# Criar grupo de recursos
az group create --name DIO-LivrariaML --location eastus

# Criar workspace
az ml workspace create --name DIO-LivrariaWS --resource-group DIO-LivrariaML
```

3. **Níveis de Acesso**:
   - Colaborador no Resource Group
   - Acesso ao Key Vault associado
