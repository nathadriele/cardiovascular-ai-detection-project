"""
Sample Data Generator for Cardiovascular AI Detection
Author: Nathalia Adriele

Gera dataset sintético para demonstração do projeto.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_cardiovascular_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Gera dados sintéticos de doença cardiovascular.

    Args:
        n_samples: Número de amostras
        random_state: Semente aleatória

    Returns:
        DataFrame com dados gerados
    """
    logger.info(f"Gerando {n_samples} amostras de dados cardiovascular")

    np.random.seed(random_state)

    # Variáveis demográficas
    id_paciente = np.arange(1, n_samples + 1)
    idade = np.random.randint(25, 85, n_samples)
    sexo = np.random.choice(['M', 'F'], n_samples)
    etnia = np.random.choice(['branca', 'negra', 'parda', 'amarela', 'indigena'], n_samples, p=[0.5, 0.2, 0.2, 0.08, 0.02])

    # Variáveis antropométricas
    peso = np.random.normal(75, 15, n_samples)
    altura = np.random.normal(1.68, 0.10, n_samples)
    imc = peso / (altura ** 2)
    circunferencia_abdominal = np.random.normal(90, 15, n_samples)

    # Sinais vitais
    pressao_arterial_sistolica = np.random.normal(130, 20, n_samples)
    pressao_arterial_diastolica = np.random.normal(85, 12, n_samples)
    frequencia_cardiaca = np.random.normal(75, 10, n_samples)

    # Exames laboratoriais
    glicemia_jejum = np.random.normal(100, 30, n_samples)
    hemoglobina_glicada = np.random.normal(5.7, 0.8, n_samples)
    colesterol_total = np.random.normal(200, 40, n_samples)
    hdl = np.random.normal(50, 15, n_samples)
    ldl = np.random.normal(120, 35, n_samples)
    triglicerideos = np.random.normal(150, 80, n_samples)

    # Variáveis comportamentais
    tabagismo = np.random.choice(['nunca', 'ex_fumante', 'fumante_atual'], n_samples, p=[0.5, 0.3, 0.2])
    alcoolismo = np.random.choice(['nunca', 'social', 'moderado', 'pesado'], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    atividade_fisica = np.random.choice(['nenhuma', '1-2x_semana', '3-4x_semana', '5+_semana'], n_samples, p=[0.3, 0.3, 0.25, 0.15])
    sedentarismo = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])

    # Variáveis clínicas (correlacionadas com idade e outros fatores)
    # Probabilidade aumenta com idade, IMC, etc.
    risk_score = (idade - 25) / 60 + (imc - 20) / 30 + (pressao_arterial_sistolica - 120) / 50

    hipertensao = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    diabetes = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    obesidade = (imc > 30).astype(int)
    dislipidemia = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    historico_familiar_doenca_cardiovascular = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

    # Variáveis medicamentosas
    uso_anti_hipertensivo = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    uso_estatina = np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
    uso_hipoglicemiante = np.random.choice([0, 1], n_samples, p=[0.92, 0.08])
    uso_anticoagulante = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])

    # Variável alvo (baseada em score de risco + ruído)
    cardiovascular_prob = 0.3 + 0.5 * risk_score - 0.2
    cardiovascular_prob = np.clip(cardiovascular_prob, 0, 1)
    presenca_doenca_cardiovascular = np.random.binomial(1, cardiovascular_prob)

    # Criar DataFrame
    data = pd.DataFrame({
        'id_paciente': id_paciente,
        'idade': idade,
        'sexo': sexo,
        'etnia': etnia,
        'peso': peso,
        'altura': altura,
        'imc': imc,
        'circunferencia_abdominal': circunferencia_abdominal,
        'pressao_arterial_sistolica': pressao_arterial_sistolica,
        'pressao_arterial_diastolica': pressao_arterial_diastolica,
        'frequencia_cardiaca': frequencia_cardiaca,
        'glicemia_jejum': glicemia_jejum,
        'hemoglobina_glicada': hemoglobina_glicada,
        'colesterol_total': colesterol_total,
        'hdl': hdl,
        'ldl': ldl,
        'triglicerideos': triglicerideos,
        'tabagismo': tabagismo,
        'alcoolismo': alcoolismo,
        'atividade_fisica': atividade_fisica,
        'sedentarismo': sedentarismo,
        'hipertensao': hipertensao,
        'diabetes': diabetes,
        'obesidade': obesidade,
        'dislipidemia': dislipidemia,
        'historico_familiar_doenca_cardiovascular': historico_familiar_doenca_cardiovascular,
        'uso_anti_hipertensivo': uso_anti_hipertensivo,
        'uso_estatina': uso_estatina,
        'uso_hipoglicemiante': uso_hipoglicemiante,
        'uso_anticoagulante': uso_anticoagulante,
        'presenca_doenca_cardiovascular': presenca_doenca_cardiovascular
    })

    # Adicionar alguns valores ausentes realisticamente
    n_missing = int(0.05 * n_samples)
    for col in ['peso', 'altura', 'circunferencia_abdominal', 'triglicerideos']:
        missing_indices = np.random.choice(n_samples, n_missing, replace=False)
        data.loc[missing_indices, col] = np.nan

    logger.info(f"Dados gerados: {data.shape}")
    logger.info(f"Distribuição do target: {data['presenca_doenca_cardiovascular'].value_counts(normalize=True).to_dict()}")

    return data


def save_data(df: pd.DataFrame, output_dir: str = "data/raw") -> None:
    """
    Salva dataset gerado.

    Args:
        df: DataFrame a ser salvo
        output_dir: Diretório de saída
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / "cardiovascular_data.csv"

    df.to_csv(filepath, index=False)
    logger.info(f"Dataset salvo em {filepath}")


if __name__ == "__main__":
    # Gerar dataset
    df = generate_cardiovascular_data(n_samples=1000, random_state=42)

    # Salvar
    save_data(df)

    # Estatísticas básicas
    print("\n" + "=" * 50)
    print("DATASET CARDIOVASCULAR - DADOS GERADOS")
    print("=" * 50)
    print(f"\nTotal de Registros: {len(df)}")
    print(f"Total de Variáveis: {df.shape[1]}")
    print(f"\nDistribuição do Target:")
    print(df['presenca_doenca_cardiovascular'].value_counts())
    print(f"\nProporção:")
    print(df['presenca_doenca_cardiovascular'].value_counts(normalize=True))
    print(f"\nValores Ausentes por Variável:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print("=" * 50)
