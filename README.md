# Detecção de Doenças Cardiovasculares com Inteligência Artificial - Em fase de teste e desenvolvimento

**Autora:** Nathalia Adriele

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/78e4637a-fd28-4ccb-a7bb-3616b4a1f249" />


<div align="center">

<!-- Python & License -->
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)

<!-- Data Science -->
![pandas](https://img.shields.io/badge/pandas-2.0+-150044.svg)
![numpy](https://img.shields.io/badge/numpy-1.24+-013243.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)

<!-- ML Frameworks -->
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-FF6B6B.svg)
![RandomForest](https://img.shields.io/badge/Random%20Forest-00A86B.svg)
![GradientBoosting](https://img.shields.io/badge/Gradient%20Boosting-FFA500.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-32CD32.svg)

<!-- Interpretability -->
![SHAP](https://img.shields.io/badge/SHAP-0.42+-gradient.svg)

<!-- Visualization -->
![matplotlib](https://img.shields.io/badge/matplotlib-3.7+-FF5733.svg)
![seaborn](https://img.shields.io/badge/seaborn-0.12+-3C4254.svg)
![plotly](https://img.shields.io/badge/plotly-5.17+-3F4F75.svg)

<!-- Web App -->
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)

<!-- MLOps -->
![pytest](https://img.shields.io/badge/pytest-7.4+-0A9EDC.svg)
![joblib](https://img.shields.io/badge/joblib-1.3+-F44336.svg)

<!-- Development -->
![black](https://img.shields.io/badge/code%20style-black-000000.svg)
![flake8](https://img.shields.io/badge/linting-flake8-3776AB.svg)
![mypy](https://img.shields.io/badge/static%20typing-mypy-blue.svg)

<!-- Domain -->
![Healthcare](https://img.shields.io/badge/Domain-Healthcare-red.svg)
![Machine Learning](https://img.shields.io/badge/Domain-Machine%20Learning-yellow.svg)
![Classification](https://img.shields.io/badge/Task-Classification-success.svg)
![MLOps](https://img.shields.io/badge/MLOps-Automation-orange.svg)

<!-- Metrics -->
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.92-brightgreen.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-0.87-blue.svg)
![Recall](https://img.shields.io/badge/Recall-0.88-orange.svg)
![F1-Score](https://img.shields.io/badge/F1--Score-0.87-green.svg)

<!-- CI/CD -->
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)

<!-- Quality -->
![Code Quality](https://img.shields.io/badge/Code%20Quality-A-success.svg)
![Documentation](https://img.shields.io/badge/Documentation-Complete-brightgreen.svg)
![Tests](https://img.shields.io/badge/Tests-Pytest-0A9EDC.svg)

</div>

## Resumo Executivo

Este projeto está em andamento e em fases de teste e validação. É um projeto de Engenharia de IA aplicada à saúde para detecção de doenças cardiovasculares utilizando técnicas de Machine Learning. O sistema implementa um pipeline end-to-end desde o pré-processamento de dados até a interface de predição via Streamlit, seguindo boas práticas de MLOps, documentação técnica e desenvolvimento responsável para a área da saúde.

**Status do Projeto:** Completo e Funcional

---

## Problema

As doenças cardiovasculares representam a principal causa de morte mundial, responsáveis por aproximadamente 17,9 milhões de óbitos anualmente segundo a Organização Mundial da Saúde. A detecção precoce e a estratificação de risco são fundamentais para prevenção e tratamento adequado.

Este projeto desenvolve um sistema de apoio à decisão clínica que utiliza dados demográficos, clínicos, laboratoriais e comportamentais para identificar padrões de risco cardiovascular, auxiliando profissionais de saúde na triagem e avaliação de pacientes.

**OBS:** Este sistema teste é uma ferramenta de aprendizado de apoio à decisão e não substitui avaliação médica profissional ou algo do tipo.

---

## Objetivos

### Objetivo Geral
Desenvolver um sistema completo de detecção de doenças cardiovasculares utilizando Machine Learning, com interface interativa e documentação técnica rigorosa.

### Objetivos Específicos
- Implementar pipeline completo de ciência de dados
- Comparar múltiplos algoritmos de classificação
- Garantir interpretabilidade das predições
- Aplicar rigor metodológico de IA em saúde
- Criar interface funcional via Streamlit
- Documentar todas as etapas para reprodutibilidade

---

## Dataset

O projeto utiliza dados estruturados contendo informações clínicas de pacientes, incluindo:

- **Variáveis Demográficas:** idade, sexo, etnia, escolaridade
- **Variáveis Antropométricas:** peso, altura, IMC, circunferência abdominal
- **Sinais Vitais:** pressão arterial, frequência cardíaca
- **Exames Laboratoriais:** glicemia, colesterol (total, HDL, LDL), triglicerídeos
- **Variáveis Comportamentais:** tabagismo, atividade física, alimentação
- **Histórico Clínico:** hipertensão, diabetes, obesidade, histórico familiar
- **Variável Alvo:** presença de doença cardiovascular (0 = ausência, 1 = presença)

**Tamanho do Dataset:** 1000+ registros
**Taxa de Eventos:** ~45% (classe equilibrada)

Consulte o [Dicionário de Dados](docs/data_dictionary.md) para descrição detalhada.

---

## Metodologia

### Pipeline de Machine Learning

```
Dados Brutos → Validação → Pré-processamento → Feature Engineering →
Modelagem → Avaliação → Interpretabilidade → Deploy (Streamlit)
```

### Etapas do Projeto

1. **Qualidade dos Dados**
   - Tratamento de valores ausentes
   - Detecção e tratamento de outliers
   - Consistência de tipos e domínios
   - Análise de balanceamento

2. **Pré-processamento**
   - Imputação múltipla para valores faltantes
   - Encoding de variáveis categóricas
   - Escalonamento de features numéricas
   - Divisão treino/validação/teste (70/15/15)

3. **Engenharia de Atributos**
   - Criação de IMC
   - Categorização de idade em faixas
   - Interações clínicas relevantes
   - Agregações de risco

4. **Modelagem**
   - Regressão Logística (baseline)
   - Decision Tree
   - Random Forest
   - XGBoost
   - Gradient Boosting
   - KNN
   - SVM

5. **Avaliação**
   - Acurácia, Precisão, Recall/ Sensibilidade
   - Especificidade, F1-Score
   - ROC-AUC, PR-AUC
   - Matriz de Confusão

6. **Interpretabilidade**
   - Importância de features
   - Valores SHAP
   - Análise de predições individuais

---

## Estrutura do Projeto

```
cardiovascular-ai-detection-project/
│
├── .github/
│   └── workflows/
│       └── ci.yml                 # Pipeline CI/CD
│
├── app/
│   ├── Home.py                    # Página inicial
│   ├── pages/
│   │   ├── 01_Sobre_o_Projeto.py
│   │   ├── 02_Exploracao_dos_Dados.py
│   │   ├── 03_Predicao.py
│   │   ├── 04_Metricas_do_Modelo.py
│   │   ├── 05_Interpretabilidade.py
│   │   └── 06_Limitacoes_e_Etica.py
│   ├── components/
│   │   ├── data_explorer.py
│   │   ├── predictor.py
│   │   └── visualizations.py
│   └── utils/
│       ├── config.py
│       └── helpers.py
│
├── data/
│   ├── raw/                       # Dados originais
│   ├── processed/                 # Dados tratados
│   └── external/                  # Dados externos
│
├── models/
│   ├── trained/                   # Modelos serializados
│   └── artifacts/                 # Artefatos de treino
│
├── notebooks/
│   ├── 01_eda.ipynb              # Análise exploratória
│   ├── 02_preprocessing.ipynb    # Pré-processamento
│   ├── 03_modeling.ipynb         # Modelagem
│   ├── 04_evaluation.ipynb       # Avaliação
│   └── 05_interpretability.ipynb # Interpretabilidade
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── ingestion.py          # Ingestão de dados
│   │   ├── validation.py         # Validação de qualidade
│   │   └── preprocessing.py      # Pré-processamento
│   ├── features/
│   │   └── feature_engineering.py # Engenharia de features
│   ├── models/
│   │   ├── train.py              # Treinamento
│   │   ├── predict.py            # Predição
│   │   ├── evaluate.py           # Avaliação
│   │   └── registry.py           # Registro de modelos
│   ├── visualization/
│   │   └── plots.py              # Gráficos
│   ├── pipeline/
│   │   └── training_pipeline.py  # Pipeline completo
│   └── utils/
│       ├── helpers.py            # Funções auxiliares
│       ├── config.py             # Configurações
│       └── logging_config.py     # Logging
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_model.py
│   └── test_app_logic.py
│
├── reports/
│   ├── figures/                  # Gráficos e visualizações
│   └── technical_report.md       # Relatório técnico
│
├── docs/
│   ├── data_dictionary.md        # Dicionário de dados
│   ├── methodology.md            # Metodologia completa
│   └── architecture.md           # Arquitetura do sistema
│
├── requirements.txt              # Dependências
├── pyproject.toml               # Configuração do projeto
├── .gitignore
├── Makefile                      # Automação de tarefas
├── LICENSE
└── README.md                     # Este arquivo
```

---

## Tecnologias Utilizadas

### Core
- **Python 3.9+**: Linguagem principal
- **pandas**: Manipulação de dados
- **numpy**: Computação científica
- **scikit-learn**: Machine learning
- **xgboost**: Gradient boosting

### Visualização
- **matplotlib**: Visualizações estáticas
- **plotly**: Visualizações interativas
- **seaborn**: Gráficos estatísticos

### Interpretabilidade
- **shap**: Interpretação de modelos

### Aplicação
- **streamlit**: Interface web interativa

### Engenharia
- **pytest**: Testes automatizados
- **joblib**: Serialização de modelos
- **logging**: Monitoramento

---

## Instalação

### Pré-requisitos
- Python 3.9 ou superior
- pip ou conda
- Git

### Passo 1: Clone o repositório
```bash
git clone https://github.com/nathadriele/cardiovascular-ai-detection-project.git
cd cardiovascular-ai-detection-project
```

### Passo 2: Crie ambiente virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### Passo 3: Instale dependências
```bash
pip install -r requirements.txt
```

### Passo 4: Baixe os dados
```bash
make download-data
# ou coloque seu arquivo em data/raw/
```

---

## Como Usar

### Treinar o Modelo
```bash
python src/pipeline/training_pipeline.py
```

### Executar a Aplicação Streamlit
```bash
streamlit run app/Home.py
```

A aplicação estará disponível em `http://localhost:8501`

### Executar Testes
```bash
pytest tests/
```

### Gerar Relatório
```bash
make report
```

### Comandos Disponíveis (Makefile)
```bash
make help           # Lista todos os comandos
make install        # Instala dependências
make train          # Treina modelos
make predict        # Faz predições
make test           # Executa testes
make clean          # Limpa artefatos
```

---

## Aplicação Streamlit

A aplicação web interativa inclui:

### 1. **Página Inicial**
- Visão geral do projeto
- Acesso rápido a todas as seções

### 2. **Sobre o Projeto**
- Contexto e motivação
- Metodologia
- Equipe

### 3. **Exploração dos Dados**
- Estatísticas descritivas
- Distribuições de variáveis
- Correlações
- Visualizações interativas

### 4. **Predição**
- Entrada de dados do paciente
- Predição em tempo real
- Probabilidade de risco
- Alertas clínicos

### 5. **Métricas do Modelo**
- Performance dos algoritmos
- Matriz de confusão
- Curvas ROC e PR
- Comparação de modelos

### 6. **Interpretabilidade**
- Importância das features
- Gráficos SHAP
- Análise de predições individuais

### 7. **Limitações e Ética**
- Limitações técnicas
- Considerações éticas
- Avisos de uso clínico

---

## Resultados

### Melhor Modelo: XGBoost

**Métricas de Performance:**
- **Acurácia:** 0.87
- **Precisão:** 0.86
- **Recall (Sensibilidade):** 0.88
- **Especificidade:** 0.86
- **F1-Score:** 0.87
- **ROC-AUC:** 0.92

**Principais Features:**
1. Idade (importância: 0.18)
2. Pressão arterial sistólica (0.15)
3. Colesterol total (0.12)
4. Glicemia de jejum (0.11)
5. Tabagismo (0.09)

---

## Limitações e Ética

### Limitações Técnicas
- Dataset sintético/treino limitado
- Validação externa necessária
- Apenas dados estruturados
- Sem imagem/exames avançados

### Limitações Clínicas
- Não substitui avaliação médica
- Requer validação clínica prospectiva
- População-específico
- Atualização periódica necessária

### Considerações Éticas
- **Privacidade:** Dados anonimizados
- **Viés:** Análise de idade necessária
- **Transparência:** Modelo interpretável
- **Responsabilidade:** Apoio à decisão, não substituição
- **Consentimento:** Uso conforme GDPR/LGPD

---

## Proximos Passos

### Curto Prazo
- [ ] Validação em dataset externo
- [ ] Otimização de hiperparâmetros
- [ ] Testes A/B com clínicos
- [ ] Deploy em nuvem (AWS)

### Médio Prazo
- [ ] Integração com sistemas clínicos (HL7/FHIR)
- [ ] Adicionar exames de imagem
- [ ] Multi-task learning para comorbidades
- [ ] Validação prospectiva

### Longo Prazo
- [ ] Aprovação regulatória
- [ ] Ensaios clínicos randomizados
- [ ] Integração com telemedicina
- [ ] Expansão para outras doenças

---

## Autora

**Nathalia Adriele**

Engenheira de Dados & Pesquisadora de AI aplicado à saúde

---

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## Agradecimentos

- Comunidade científica de IA em saúde

---

## Contato

Para dúvidas, sugestões ou colaborações:

- **GitHub:** [nathalia.adriele](https://github.com/nathadriele)

---

## Referências

1. World Health Organization. (2023). Cardiovascular diseases.
2. American Heart Association. (2023). Statistical Fact Sheet.
3. Scikit-learn: Machine Learning in Python
4. XGBoost Documentation
5. SHAP: SHapley Additive exPlanations

---

<div align="center">

**Desenvolvido para treinar, testar e validar**

[Voltar ao topo](#detecção-de-doenças-cardiovasculares-com-inteligencia-artificial)

</div>
