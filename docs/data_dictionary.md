# Dicionário de Dados - Detecção de Doenças Cardiovasculares

**Projeto:** Cardiovascular AI Detection
**Autora:** Nathalia Adriele
**Data:** 2024

---

## 1. Visão Geral

Este documento descreve todas as variáveis utilizadas no projeto de detecção de doenças cardiovasculares, incluindo tipos de dados, descrições clínicas, domínios e tratamentos aplicados.

**Total de Variáveis:** 38
**Variável Alvo:** presenca_doenca_cardiovascular
**Tamanho do Dataset:** 1000+ registros

---

## 2. Variáveis Demográficas

| Variável | Tipo | Descrição | Domínio | Tratamento |
|----------|------|-----------|---------|------------|
| id_paciente | int | Identificador único do paciente | 1-99999 | Removido antes da modelagem |
| idade | int | Idade em anos completos | 25-85 | Numérica contínua, sem tratamento |
| sexo | str | Sexo biológico | 'M', 'F' | One-hot encoding |
| etnia | str | Autodeclaração racial/étnica | 'branca', 'negra', 'parda', 'amarela', 'indigena' | One-hot encoding |
| escolaridade | str | Nível de escolaridade | 'fundamental', 'medio', 'superior', 'pos' | Ordinal encoding |
| estado_civil | str | Estado civil | 'solteiro', 'casado', 'divorciado', 'viuvo' | One-hot encoding |
| renda | float | Renda familiar mensal (R$) | 0-50000 | Escalonamento StandardScaler |

---

## 3. Variáveis Antropométricas

| Variável | Tipo | Descrição | Domínio | Unidade | Tratamento |
|----------|------|-----------|---------|---------|------------|
| peso | float | Peso corporal | 35-150 | kg | Numérica contínua |
| altura | float | Altura | 1.40-2.10 | m | Numérica contínua |
| imc | float | Indice de massa corporal (calculado) | 12-50 | kg/m² | Derivado de peso/altura² |
| circunferencia_abdominal | float | Circunferência abdominal | 50-150 | cm | Numérica contínua |

**IMC Classificação:**
- Abaixo do peso: < 18.5
- Normal: 18.5-24.9
- Sobrepeso: 25-29.9
- Obesidade I: 30-34.9
- Obesidade II: 35-39.9
- Obesidade III: >= 40

---

## 4. Variáveis Vitais

| Variável | Tipo | Descrição | Domínio | Unidade | Tratamento |
|----------|------|-----------|---------|---------|------------|
| pressao_arterial_sistolica | int | Pressão arterial sistólica | 80-220 | mmHg | Numérica contínua |
| pressao_arterial_diastolica | int | Pressão arterial diastólica | 40-140 | mmHg | Numérica contínua |
| frequencia_cardiaca | int | Frequência cardíaca em repouso | 40-140 | bpm | Numérica contínua |

**Classificação PA:**
- Normal: < 120/80 mmHg
- Pré-hipertensão: 120-139/80-89 mmHg
- Hipertensão Estágio 1: 140-159/90-99 mmHg
- Hipertensão Estágio 2: >= 160/100 mmHg

---

## 5. Variáveis Laboratoriais

| Variável | Tipo | Descrição | Domínio | Unidade | Tratamento |
|----------|------|-----------|---------|---------|------------|
| glicemia_jejum | float | Glicose em jejum | 60-400 | mg/dL | Numérica contínua |
| hemoglobina_glicada | float | Hemoglobina glicada (HbA1c) | 4-15 | % | Numérica contínua |
| colesterol_total | float | Colesterol total | 100-450 | mg/dL | Numérica contínua |
| hdl | float | Lipoproteína de alta densidade | 20-120 | mg/dL | Numérica contínua |
| ldl | float | Lipoproteína de baixa densidade | 30-300 | mg/dL | Numérica contínua |
| triglicerideos | float | Triglicerídeos | 30-1000 | mg/dL | Numérica contínua |
| creatinina | float | Creatinina sérica | 0.3-8.0 | mg/dL | Numérica contínua |
| ureia | float | Ureia sérica | 10-150 | mg/dL | Numérica contínua |
| acido_urico | float | Ácido úrico | 2.0-15.0 | mg/dL | Numérica contínua |
| sodio | float | Sódio sérico | 120-160 | mEq/L | Numérica contínua |
| potassio | float | Potássio sérico | 2.5-7.0 | mEq/L | Numérica contínua |

**Valores de Referência:**
- Glicemia jejum normal: 70-99 mg/dL
- HbA1c normal: < 5.7%
- Colesterol total desejável: < 200 mg/dL
- HDL proteção: > 40 mg/dL (homens), > 50 mg/dL (mulheres)
- LDL desejável: < 100 mg/dL
- Triglicerídeos normal: < 150 mg/dL

---

## 6. Variáveis Comportamentais

| Variável | Tipo | Descrição | Domínio | Tratamento |
|----------|------|-----------|---------|------------|
| tabagismo | str | Status de tabagismo | 'nunca', 'ex_fumante', 'fumante_atual' | Ordinal encoding |
| alcoolismo | str | Consumo de álcool | 'nunca', 'social', 'moderado', 'pesado' | Ordinal encoding |
| atividade_fisica | str | Frequência de atividade física | 'nenhuma', '1-2x_semana', '3-4x_semana', '5+_semana' | Ordinal encoding |
| sedentarismo | bin | Estilo de vida sedentário | 0, 1 | Binária (1=sedentário) |
| padrao_alimentar | str | Qualidade da dieta | 'ruim', 'regular', 'bom', 'otimo' | Ordinal encoding |

---

## 7. Variáveis Clínicas

| Variável | Tipo | Descrição | Domínio | Tratamento |
|----------|------|-----------|---------|------------|
| hipertensao | bin | Diagnóstico de hipertensão | 0, 1 | Binária |
| diabetes | bin | Diagnóstico de diabetes | 0, 1 | Binária |
| obesidade | bin | Diagnóstico de obesidade | 0, 1 | Binária |
| dislipidemia | bin | Diagnóstico de dislipidemia | 0, 1 | Binária |
| historico_familiar_doenca_cardiovascular | bin | Histórico familiar de DCV | 0, 1 | Binária |
| insuficiencia_cardiaca | bin | Insuficiência cardíaca | 0, 1 | Binária |
| doenca_coronariana | bin | Doença coronariana | 0, 1 | Binária |
| arritmia | bin | Arritmia cardíaca | 0, 1 | Binária |
| angina | bin | Angina pectoris | 0, 1 | Binária |
| infarto_previo | bin | Infarto do miocárdio prévio | 0, 1 | Binária |
| avc_previo | bin | Acidente vascular cerebral prévio | 0, 1 | Binária |

---

## 8. Variáveis Medicamentosas

| Variável | Tipo | Descrição | Domínio | Tratamento |
|----------|------|-----------|---------|------------|
| uso_anti_hipertensivo | bin | Uso de anti-hipertensivos | 0, 1 | Binária |
| uso_estatina | bin | Uso de estatinas | 0, 1 | Binária |
| uso_hipoglicemiante | bin | Uso de hipoglicemiantes | 0, 1 | Binária |
| uso_anticoagulante | bin | Uso de anticoagulantes | 0, 1 | Binária |

---

## 9. Variáveis de Exames Cardiológicos

| Variável | Tipo | Descrição | Domínio | Tratamento |
|----------|------|-----------|---------|------------|
| resultado_eletrocardiograma | str | Resultado do ECG | 'normal', 'alteracoes_minimas', 'alteracoes_significativas' | Ordinal encoding |
| dor_toracica | bin | Presença de dor torácica | 0, 1 | Binária |
| angina_induzida_exercicio | bin | Angina induzida por exercício | 0, 1 | Binária |
| capacidade_funcional | int | Capacidade funcional (METs) | 1-15 | Numérica discreta |
| frequencia_cardiaca_maxima | int | Frequência cardíaca máxima atingida | 80-220 | Numérica contínua |
| depressao_st | bin | Depressão do segmento ST | 0, 1 | Binária |
| resultado_teste_ergometrico | str | Resultado do teste ergométrico | 'normal', 'inconclusivo', 'positivo' | One-hot encoding |

---

## 10. Variável Alvo

| Variável | Tipo | Descrição | Domínio | Tratamento |
|----------|------|-----------|---------|------------|
| presenca_doenca_cardiovascular | bin | Presença de doença cardiovascular | 0, 1 | Variável alvo (binária) |

**Definição:**
- 0: Ausência de doença cardiovascular diagnosticada
- 1: Presença de doença cardiovascular diagnosticada

**Doenças Incluídas:**
- Doença coronariana
- Insuficiência cardíaca
- Arritmias complexas
- Histórico de IAM
- Histórico de AVC

---

## 11. Classificação de Tipos de Dados

### Numéricas Contínuas
idade, peso, altura, imc, circunferencia_abdominal, pressao_arterial_sistolica, pressao_arterial_diastolica, frequencia_cardiaca, glicemia_jejum, hemoglobina_glicada, colesterol_total, hdl, ldl, triglicerideos, creatinina, ureia, acido_urico, sodio, potassio, frequencia_cardiaca_maxima

**Impacto:** Requer escalonamento, análise de distribuição, detecção de outliers.

### Numéricas Discretas
capacidade_funcional

**Impacto:** Pode ser tratada como contínua ou categórica ordinal.

### Categóricas Nominais
sexo, etnia, estado_civil, resultado_teste_ergometrico

**Impacto:** Requer one-hot encoding, não há ordem natural.

### Categóricas Ordinais
escolaridade, tabagismo, alcoolismo, atividade_fisica, padrao_alimentar, resultado_eletrocardiograma

**Impacto:** Requer ordinal encoding (preserva ordem).

### Binárias
sedentarismo, hipertensao, diabetes, obesidade, dislipidemia, historico_familiar_doenca_cardiovascular, insuficiencia_cardiaca, doenca_coronariana, arritmia, angina, infarto_previo, avc_previo, uso_anti_hipertensivo, uso_estatina, uso_hipoglicemiante, uso_anticoagulante, dor_toracica, angina_induzida_exercicio, depressao_st

**Impacto:** Não requer encoding adicional, prontas para modelagem.

---

## 12. Variáveis Derivadas (Feature Engineering)

### Criadas Durante o Pré-processamento

1. **faixa_etaria**: Categorização de idade
   - '25-34', '35-44', '45-54', '55-64', '65-74', '75+'

2. **imc_categoria**: Categorização de IMC
   - 'abaixo_peso', 'normal', 'sobrepeso', 'obesidade_i', 'obesidade_ii', 'obesidade_iii'

3. **pressao_arterial_categoria**: Classificação da PA
   - 'normal', 'pre_hipertensao', 'hipertensao_i', 'hipertensao_ii'

4. **colesterol_nao_hdl**: Calculado como colesterol_total - hdl

5. **ratio_ldl_hdl**: Proporção ldl/hdl

6. **ratio_triglicerides_hdl**: Proporção triglicerídeos/hdl

7. **score_risco_framingham_simplificado**: Score de risco simplificado

---

## 13. Plano de Tratamento de Qualidade

### Valores Ausentes
- **Demográficas:** Imputação por moda
- **Laboratoriais:** Imputação por mediana (robusta a outliers)
- **Vitais:** Imputação por KNN (k=5)
- **Comportamentais:** Imputação por moda

### Outliers
- **Detecção:** IQR (1.5 * IQR) ou Z-score (> 3)
- **Tratamento:** Capping em percentis 1 e 99

### Inconsistências
- **PA sistólica < diastólica:** Troca de valores ou exclusão
- **IMC < 12 ou > 60:** Revisão ou exclusão
- **Glicemia < 40 ou > 600:** Revisão ou exclusão

### Codificação
- Padronização de strings (lowercase, strip)
- Conversão de tipos (object → category)
- Tratamento de valores '9', '999', 'NA' como ausentes

---

## 14. Prevenção de Data Leakage

### Regras Aplicadas

1. **Imputação:** Fit apenas em treino, transform em validação/teste
2. **Escalonamento:** Fit apenas em treino, transform em validação/teste
3. **Encoding:** Fit apenas em treino, transform em validação/teste
4. **Feature Engineering:** Usar apenas features disponíveis no momento da predição

### Pipeline de Separação

```
Dados Originais → Split (70/15/15) → Pré-processamento (fit treino) →
Transform (treino/val/teste) → Modelagem
```

---

## 15. Referências Clínicas

- Diretrizes Brasileiras de Hipertensão, 2023
- American Heart Association. Guidelines for Cardiovascular Disease Prevention
- Sociedade Brasileira de Cardiologia. Diretriz de Dislipidemias
- Organização Mundial da Saúde. Classificação de IMC

---

**Documento Versão:** 1.0
**Última Atualização:** 2024
**Responsável:** Nathalia Adriele
