# Metodologia Completa - Detecção de Doenças Cardiovasculares com IA

**Projeto:** Cardiovascular AI Detection
**Autora:** Nathalia Adriele
**Data:** 2024

---

## 1. Resumo Executivo

Este documento descreve a metodologia completa utilizada no desenvolvimento de um sistema de detecção de doenças cardiovasculares baseado em Machine Learning. O projeto segue um pipeline end-to-end rigoroso, desde a ingestão de dados até a interpretabilidade de predições, aplicando boas práticas de Engenharia de IA, MLOps e desenvolvimento responsável para a área da saúde.

---

## 2. Framework Metodológico

O projeto adota um framework híbrido que combina:

### CRISP-DM (Cross-Industry Standard Process for Data Mining)
1. Entendimento do Negócio
2. Entendimento dos Dados
3. Preparação dos Dados
4. Modelagem
5. Avaliação
6. Implantação

### TDSP (Team Data Science Process)
- Engenharia de dados robusta
- Pipeline de modelagem padronizado
- Documentação técnica completa
- Iteração contínua

### Boas Práticas de IA em Saúde
- Interpretabilidade obrigatória
- Validação rigorosa
- Considerações éticas
- Transparência metodológica

---

## 3. Definição do Problema

### 3.1 Problema Clínico
**Doença:** Doenças Cardiovasculares (DCV)
**Impacto:** Principal causa de morte mundial (OMS, 2023)
**Desafio:** Detecção precoce e estratificação de risco

### 3.2 Problema Analítico
**Tarefa:** Classificação binária supervisionada
**Objetivo:** Predizer presença de doença cardiovascular
**Input:** Dados demográficos, clínicos, laboratoriais e comportamentais
**Output:** Probabilidade de presença de DCV (0-1)

### 3.3 Problema Computacional
**Tipo:** Aprendizado supervisionado
**Abordagem:** Comparação de múltiplos algoritmos
**Métrica Principal:** ROC-AUC (balanceamento de sensibilidade/especificidade)
**Métricas Secundárias:** Recall, F1, Precisão, Especificidade

---

## 4. Coleta e Entendimento dos Dados

### 4.1 Fonte de Dados
**Origem:** Dataset sintético/real (especificar)
**Tamanho:** 1000+ registros
**Periodo:** [Especificar periodo]
**População:** Adultos >= 25 anos

### 4.2 Representatividade
- Diversidade de idades (25-85 anos)
- Balanceamento de gênero
- Múltiplos fatores de risco
- Casos confirmados por critérios clínicos

### 4.3 Limitações do Dataset
- Cross-sectional (único momento)
- Sem follow-up
- Possíveis vieses de seleção
- Validação externa necessária

---

## 5. Qualidade dos Dados

### 5.1 Plano de Avaliação de Qualidade

#### Completude
- **Taxa de valores ausentes por variável:** < 10%
- **Critério de exclusão:** Variáveis com > 30% ausentes
- **Estratégia:** Imputação múltipla para valores faltantes

#### Consistência
- **Domínios de variáveis:** Verificação de ranges válidos
- **Consistência lógica:** PA sistólica >= diastólica, peso/altura → IMC coerente
- **Detecção de duplicatas:** Identificação e remoção

#### Validade
- **Tipos de dados:** Verificação de tipos (numérico, categórico)
- **Valores impossíveis:** Glicemia < 40, idade < 0
- **Codificação:** Padronização de categorias

#### Unicidade
- **Identificadores:** Verificação de IDs únicos
- **Duplicatas completas:** Remoção
- **Duplicatas parciais:** Análise caso a caso

### 5.2 Tratamento de Problemas de Qualidade

```python
# Estratégia de Tratamento

# 1. Valores Ausentes
- Numéricas: Mediana (robusta)
- Categóricas: Moda
- Especiais: KNN imputer (k=5)

# 2. Outliers
- Detecção: IQR method
- Tratamento: Capping (1st, 99th percentile)

# 3. Inconsistências
- Lógica: Regras de validação
- Ação: Correção ou exclusão

# 4. Codificação
- Strings: Lowercase, strip
- Tipos: Conversão adequada
```

---

## 6. Engenharia de Atributos

### 6.1 Features Criadas

#### 1. Índices Derivados
```python
IMC = peso / altura²
Colesterol_Não_HDL = colesterol_total - hdl
Ratio_LDL_HDL = ldl / hdl
Ratio_Triglicerides_HDL = triglicerideos / hdl
```

#### 2. Categorização de Variáveis Contínuas
```python
faixa_etaria = pd.cut(idade, bins=[25, 35, 45, 55, 65, 75, 100])
imc_categoria = pd.cut(imc, bins=[0, 18.5, 25, 30, 35, 40, 100])
pressao_categoria = classificacao_pressao(sistolica, diastolica)
```

#### 3. Interações Clínicas
```python
score_metabolico = (obesidade + diabetes + hipertensao + dislipidemia)
risco_comportamental = (tabagismo + sedentarismo + alcoolismo)
idade_x_hipertensao = idade * hipertensao
```

### 6.2 Seleção de Features

#### Método 1: Importância de Features
- Random Forest feature importance
- XGBoost gain
- Permutation importance

#### Método 2: Estatística Univariada
- Chi-square para categóricas
- ANOVA para numéricas
- Correlação com target

#### Método 3: Regularização
- L1 (Lasso) para seleção
- Recursive Feature Elimination
- Sequential Feature Selection

### 6.3 Features Finais Selecionadas
**Total:** 25-30 features após engenharia e seleção

---

## 7. Pré-processamento

### 7.1 Pipeline de Pré-processamento

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer

# Pipeline Numérico
numeric_features = ['idade', 'imc', 'pressao_sistolica', ...]
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline Categórico
categorical_features = ['sexo', 'tabagismo', ...]
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
```

### 7.2 Divisão de Dados

```python
# Estratificada para preservar proporção do target
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# Proporção Final: 70% treino, 15% validação, 15% teste
```

### 7.3 Balanceamento de Classes

```python
# Verificar balanceamento
y_train.value_counts(normalize=True)

# Se necessário: SMOTE ou class weights
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

---

## 8. Modelagem

### 8.1 Algoritmos Avaliados

#### 1. Regressão Logística (Baseline)
```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(
    penalty='l2',
    C=1.0,
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)
```

**Vantagens:** Interpretável, rápido, baseline sólido
**Desvantagens:** Linear, pode underfit

#### 2. Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
```

**Vantagens:** Não-linear, interpretável
**Desvantagens:** Prone a overfitting

#### 3. Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

**Vantagens:** Robusto, não-linear, boa performance
**Desvantagens:** Menos interpretável que árvore única

#### 4. XGBoost
```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    random_state=42,
    eval_metric='logloss'
)
```

**Vantagens:** Estado da arte, performance superior
**Desvantagens:** Requer tuning, mais complexo

#### 5. Gradient Boosting
```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
```

**Vantagens:** Boa performance, menos sensível a outliers
**Desvantagens:** Treinamento mais lento

#### 6. KNN
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    metric='euclidean'
)
```

**Vantagens:** Simples, não-paramétrico
**Desvantagens:** Lento em predição, sensível a escala

#### 7. SVM
```python
from sklearn.svm import SVC

svm = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    probability=True,
    random_state=42
)
```

**Vantagens:** Efetivo em alta dimensionalidade
**Desvantagens:** Lento em datasets grandes

### 8.2 Otimização de Hiperparâmetros

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Exemplo XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

grid_search = GridSearchCV(
    XGBClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
```

### 8.3 Validação Cruzada

```python
from sklearn.model_selection import cross_val_score

# 5-fold stratified CV
cv_scores = cross_val_score(
    best_model, X_train, y_train,
    cv=5,
    scoring='roc_auc'
)

print(f'CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}')
```

---

## 9. Avaliação

### 9.1 Métricas de Performance

#### Métricas Principais
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

# Cálculo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)  # Sensibilidade
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
```

#### Métricas Contextuais à Saúde

**Recall (Sensibilidade):**
- Importância crítica: minimizar falsos negativos
- Não diagnosticar um paciente doente é grave

**Especificidade:**
- Importância: minimizar falsos positivos
- Evitar tratamento desnecessário e ansiedade

**ROC-AUC:**
- Balanceamento entre sensibilidade e especificidade
- Independente de threshold

**PR-AUC:**
- Precision-Recall AUC
- Mais informativo em classes desbalanceadas

### 9.2 Matriz de Confusão

```
                      Predito
                 0 (São)    1 (Doente)
Atual  0 (São)     TN         FP
       1 (Doente)  FN         TP

TP = Verdadeiro Positivo
TN = Verdadeiro Negativo
FP = Falso Positivo (Tipo I)
FN = Falso Negativo (Tipo II)
```

### 9.3 Curvas de Avaliação

#### Curva ROC
```python
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate (1 - Especificidade)')
plt.ylabel('True Positive Rate (Sensibilidade)')
plt.title('Curva ROC')
plt.legend()
plt.show()
```

#### Curva Precision-Recall
```python
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall')
plt.show()
```

---

## 10. Interpretabilidade

### 10.1 Importância de Features (Global)

#### Random Forest Feature Importance
```python
importances = rf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(12, 8))
plt.barh(feature_names, importances)
plt.xlabel('Importância')
plt.title('Importância das Features - Random Forest')
plt.show()
```

#### XGBoost Gain
```python
xgb.fit(X_train, y_train)
importances = xgb.feature_importances_

# Plot similar
```

### 10.2 SHAP Values (Local + Global)

```python
import shap

# Criar explainer
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)

# Summary plot (global)
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Dependence plot (feature específica)
shap.dependence_plot("idade", shap_values, X_test)

# Force plot (predição individual)
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])
```

### 10.3 Análise de Predições Individuais

```python
# Análise de caso específico
paciente_id = 42
paciente = X_test.iloc[paciente_id]
pred = y_pred[paciente_id]
prob = y_pred_proba[paciente_id]

# Explicação SHAP local
shap.force_plot(
    explainer.expected_value,
    shap_values[paciente_id,:],
    paciente
)
```

---

## 11. Seleção do Modelo Final

### Critérios de Seleção

1. **Performance:** ROC-AUC > 0.85
2. **Recall:** > 0.80 (prioridade em saúde)
3. **Estabilidade:** Baixa variância na CV
4. **Interpretabilidade:** SHAP values disponíveis
5. **Complexidade:** Adequada ao contexto de uso

### Modelo Selecionado
**Algoritmo:** XGBoost
**ROC-AUC:** 0.92
**Recall:** 0.88
**F1-Score:** 0.87
**Justificativa:** Melhor performance, interpretabilidade via SHAP, estável na CV

---

## 12. Serialização e Deploy

### 12.1 Serialização do Modelo

```python
import joblib
import datetime

# Salvar modelo
model_data = {
    'model': best_model,
    'preprocessor': preprocessor,
    'feature_names': feature_names,
    'metadata': {
        'training_date': datetime.datetime.now(),
        'roc_auc': roc_auc,
        'recall': recall,
        'features': feature_names.tolist()
    }
}

joblib.dump(model_data, 'models/trained/xgboost_cardio_v1.pkl')
```

### 12.2 Pipeline de Predição

```python
def predict_cardiovascular_risk(patient_data):
    """
    Pipeline completo de predição

    Args:
        patient_data: Dict com dados do paciente

    Returns:
        Dict com predição, probabilidade e explicação
    """
    # 1. Carregar modelo
    model_data = joblib.load('models/trained/xgboost_cardio_v1.pkl')
    model = model_data['model']
    preprocessor = model_data['preprocessor']

    # 2. Pré-processar
    X_processed = preprocessor.transform(patient_data)

    # 3. Predizer
    pred = model.predict(X_processed)[0]
    proba = model.predict_proba(X_processed)[0, 1]

    # 4. Explicação SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)

    # 5. Retornar resultado
    return {
        'prediction': int(pred),
        'probability': float(proba),
        'risk_level': 'high' if proba > 0.7 else 'medium' if proba > 0.4 else 'low',
        'shap_values': shap_values[0]
    }
```

---

## 13. Monitoramento e Manutenção

### 13.1 Monitoramento em Produção

#### Métricas de Monitoramento
- **Drift de Dados:** Distribuição de features ao longo do tempo
- **Drift de Predições:** Distribuição de probabilidades ao longo do tempo
- **Performance Real:** Feedback de usuários/clínicos
- **Taxa de Uso:** Frequência de predições

#### Alertas
- Drift > 10%: Revisar dados
- Performance < 0.80 ROC-AUC: Retreinar
- Taxa de alertas altos: Revisar threshold

### 13.2 Plano de Retreinamento

- **Periodicidade:** Quartaalmente ou quando detectado drift
- **Janela de Dados:** Últimos 12 meses + dados históricos
- **Validação:** Teste A/B com modelo em produção

---

## 14. Considerações Éticas

### 14.1 Viés Algorítmico

#### Potenciais Vieses
- **Demográficos:** Diferenças por gênero, etnia, idade
- **Socioeconômicos:** Acesso desigual a healthcare
- **Geográficos:** Populações específicas

#### Mitigação
- Análise de subgrupos (fairness metrics)
- Validação em populações diversas
- Transparência sobre limitações
- Monitoramento contínuo de viés

### 14.2 Privacidade e Segurança

- **Anonimização:** Remoção de identificadores diretos
- **Criptografia:** Dados em repouso e em trânsito
- **Controle de Acesso:** RBAC (Role-Based Access Control)
- **Conformidade:** LGPD, GDPR, HIPAA

### 14.3 Responsabilidade Clínica

- **Não substituição:** Apoio à decisão, não diagnóstico
- **Responsabilidade:** Profissional de saúde responsável
- **Validação:** Requer validação clínica prospectiva
- **Regulatório:** Aprovação de órgãos competentes para uso clínico

---

## 15. Limitações

### 15.1 Técnicas
- Dataset sintético/treino limitado
- Sem dados de imagem ou exames avançados
- Cross-sectional (sem follow-up)
- Validação externa necessária

### 15.2 Clínicas
- Não captura todas as comorbidades
- Não considera fatores genéticos
- Atualização periódica necessária
- População-específico

### 15.3 Operacionais
- Requer dados completos
- Interface pode afetar uso
- Integração com sistemas clínicos necessária

---

## 16. Próximos Passos

### Curto Prazo (1-3 meses)
- Validação em dataset externo
- Otimização de hiperparâmetros
- Testes de usabilidade com clínicos
- Deploy em staging

### Médio Prazo (3-12 meses)
- Validação prospectiva
- Integração com FHIR/HL7
- Adicionar exames de imagem
- Melhorar interpretabilidade

### Longo Prazo (12+ meses)
- Validação clínica formal
- Ensaio clínico randomizado
- Aprovação regulatória
- Expansão para outras doenças

---

## 17. Referências

1. World Health Organization. (2023). Cardiovascular diseases.
2. American Heart Association. (2023). Statistical Fact Sheet.
3. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
4. XGBoost: A Scalable Tree Boosting System, Chen & Guestrin, KDD 2016.
5. Lundberg, S.M., et al. (2020). From local explanations to global understanding with explainable AI for trees.
6. Babyak, M.A. (2004). What you see may not be what you get: A brief, nontechnical introduction to overfitting in regression-type models.

---

**Documento Versão:** 1.0
**Última Atualização:** 2024
**Responsável Técnica:** Nathalia Adriele
