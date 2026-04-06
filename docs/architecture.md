# Arquitetura do Sistema - Detecção de Doenças Cardiovasculares

**Projeto:** Cardiovascular AI Detection
**Autora:** Nathalia Adriele
**Data:** 2 de abril de 2026

---

## 1. Visão Arquitetural

O sistema segue uma arquitetura modular em camadas, seguindo princípios de engenharia de software e MLOps.

```
┌─────────────────────────────────────────────────────────────┐
│                    CAMADA DE APRESENTAÇÃO                    │
│                      (Streamlit App)                         │
├─────────────────────────────────────────────────────────────┤
│  Home  │  Sobre  │  Dados  │  Predição  │  Métricas  │  ... │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      CAMADA DE SERVIÇO                       │
│                    (Business Logic)                         │
├─────────────────────────────────────────────────────────────┤
│  Prediction Service  │  Evaluation Service  │  Interpreter  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    CAMADA DE MODELO                          │
│                   (ML Models Layer)                         │
├─────────────────────────────────────────────────────────────┤
│  XGBoost  │  RF  │  LR  │  GBM  │  SVM  │  KNN  │  DT     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 CAMADA DE PRÉ-PROCESSAMENTO                  │
│                (Data Processing Layer)                      │
├─────────────────────────────────────────────────────────────┤
│  Validation  │  Preprocessing  │  Feature Engineering        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    CAMADA DE DADOS                           │
│                    (Data Layer)                             │
├─────────────────────────────────────────────────────────────┤
│  Raw Data  │  Processed Data  │  Model Artifacts            │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Componentes da Arquitetura

### 2.1 Camada de Apresentação (App Layer)

**Localização:** `app/`

**Responsabilidades:**
- Interface web interativa via Streamlit
- Visualização de dados e resultados
- Entrada de dados do usuário
- Exibição de métricas e explicações

**Componentes:**
- `Home.py`: Aplicação principal
- `pages/`: Páginas individuais do sistema
- `components/`: Componentes reutilizáveis
- `utils/`: Funções auxiliares da aplicação

### 2.2 Camada de Serviço (Service Layer)

**Localização:** `src/models/`

**Responsabilidades:**
- Lógica de negócio de ML
- Treinamento de modelos
- Predição e inferência
- Avaliação de performance
- Interpretabilidade

**Componentes:**
- `train.py`: Treinamento de modelos
- `predict.py`: Predição em novos dados
- `evaluate.py`: Avaliação de performance
- `interpret.py`: Interpretação com SHAP
- `registry.py`: Registro de modelos

### 2.3 Camada de Processamento de Dados (Data Processing Layer)

**Localização:** `src/data/` e `src/features/`

**Responsabilidades:**
- Ingestão de dados
- Validação de qualidade
- Pré-processamento
- Feature engineering
- Transformação de dados

**Componentes:**
- `ingestion.py`: Leitura de dados
- `validation.py`: Validação de qualidade
- `preprocessing.py`: Pré-processamento
- `feature_engineering.py`: Engenharia de features

### 2.4 Camada de Dados (Data Layer)

**Localização:** `data/`

**Responsabilidades:**
- Armazenamento de dados brutos
- Armazenamento de dados processados
- Armazenamento de artefatos de modelo

**Estrutura:**
- `raw/`: Dados originais
- `processed/`: Dados tratados
- `external/`: Dados externos

### 2.5 Camada de Artefatos (Artifacts Layer)

**Localização:** `models/`

**Responsabilidades:**
- Armazenamento de modelos treinados
- Armazenamento de pré-processadores
- Armazenamento de metadados

**Estrutura:**
- `trained/`: Modelos serializados
- `artifacts/`: Pré-processadores e configurações

---

## 3. Fluxo de Dados

### 3.1 Fluxo de Treinamento

```
Dados Brutos → Validação → Pré-processamento → Feature Engineering →
Divisão Treino/Val/Teste → Treinamento → Avaliação →
Serialização → Deploy
```

### 3.2 Fluxo de Predição

```
Dados do Paciente → Validação → Pré-processamento → Feature Engineering →
Modelo Carregado → Predição → Interpretação → Resultado
```

---

## 4. Design Patterns Utilizados

### 4.1 Pipeline Pattern

**Onde:** `src/pipeline/training_pipeline.py`

**Objetivo:** Orquestrar o fluxo completo de treino

**Benefícios:**
- Reprodutibilidade
- Modularidade
- Fácil manutenção

### 4.2 Strategy Pattern

**Onde:** `src/models/train.py`

**Objetivo:** Permitir troca de algoritmos facilmente

**Benefícios:**
- Flexibilidade
- Comparação justa entre modelos
- Extensibilidade

### 4.3 Factory Pattern

**Onde:** `src/data/ingestion.py`

**Objetivo:** Criar objetos de ingestão de dados

**Benefícios:**
- Abstração da fonte de dados
- Suporte a múltiplos formatos

### 4.4 Singleton Pattern

**Onde:** Configurações globais

**Objetivo:** Garantir única instância de configurações

**Benefícios:**
- Consistência
- Economia de memória

---

## 5. Tecnologias e Ferramentas

### 5.1 Core
- **Python 3.9+**: Linguagem principal
- **pandas**: Manipulação de dados
- **numpy**: Computação científica

### 5.2 Machine Learning
- **scikit-learn**: Framework de ML
- **xgboost**: Gradient boosting
- **imbalanced-learn**: Balanceamento de classes

### 5.3 Interpretabilidade
- **shap**: Interpretação de modelos

### 5.4 Visualização
- **matplotlib**: Gráficos estáticos
- **plotly**: Gráficos interativos
- **seaborn**: Visualização estatística

### 5.5 Aplicação Web
- **streamlit**: Interface web

### 5.6 DevOps
- **pytest**: Testes automatizados
- **GitHub Actions**: CI/CD
- **Docker**: Containerização

---

## 6. Escalabilidade e Performance

### 6.1 Otimizações Implementadas

**Treinamento:**
- Paralelização com `n_jobs=-1`
- Validação cruzada estratificada
- Otimização de hiperparâmetros eficiente

**Predição:**
- Batch processing
- Caching de modelos carregados
- Pré-processamento otimizado

**Aplicação:**
- Lazy loading de recursos
- Caching de resultados
- Streaming de dados grandes

### 6.2 Estratégias de Escala

**Horizontal Scaling:**
- API REST para múltiplas instâncias
- Load balancer
- Kubernetes deployment

**Vertical Scaling:**
- Otimização de memória
- GPU para treinamento
- SSD para I/O rápido

---

## 7. Segurança e Privacidade

### 7.1 Proteção de Dados

**Em Repouso:**
- Criptografia de arquivos sensíveis
- Controle de acesso RBAC
- Anonimização de dados

**Em Trânsito:**
- HTTPS/TLS
- API authentication
- Rate limiting

### 7.2 Conformidade

- **LGPD**: Lei Geral de Proteção de Dados (Brasil)
- **GDPR**: General Data Protection Regulation (Europa)
- **HIPAA**: Health Insurance Portability and Accountability Act (EUA)

---

## 8. Monitoramento e Observabilidade

### 8.1 Métricas Monitoradas

**Modelo:**
- Drift de dados
- Drift de predições
- Performance ao longo do tempo
- Taxa de erros

**Aplicação:**
- Tempo de resposta
- Taxa de sucesso
- Uso de recursos
- Erros e exceções

### 8.2 Alertas

- Performance abaixo de threshold
- Drift detectado
- Erros em produção
- Falhas de saúde

---

## 9. Deploy e Infraestrutura

### 9.1 Ambientes

**Desenvolvimento:**
- Local com ambiente virtual
- Jupyter notebooks
- Testes automatizados

**Staging:**
- Cloud staging
- Testes de integração
- Validação de features

**Produção:**
- Cloud production
- Alta disponibilidade
- Backup e recovery

### 9.2 Estratégia de Deploy

**Continuous Deployment:**
- CI/CD automatizado
- Testes automáticos
- Deploy por stages

**Rollback:**
- Versionamento de modelos
- Reversão automática em falhas
- Backup de versões anteriores

---

## 10. Manutenção e Evolução

### 10.1 Manutenção

**Rotina:**
- Atualização de dependências
- Retreinamento periódico
- Monitoramento contínuo
- Correção de bugs

**Preventiva:**
- Testes de carga
- Auditoria de segurança
- Análise de viés
- Validação clínica

### 10.2 Evolução

**Curto Prazo:**
- Validação externa
- Otimização de features
- Melhoria de UX

**Médio Prazo:**
- Integração com sistemas clínicos
- Adicionar exames de imagem
- Multi-task learning

**Longo Prazo:**
- Expansão para outras doenças
- Validação clínica formal
- Aprovação regulatória

---

**Documento Versão:** 1.0
**Última Atualização:** 2 de abril de 2026
**Responsável:** Nathalia Adriele
