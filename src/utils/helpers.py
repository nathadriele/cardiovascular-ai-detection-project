"""
Helper Functions Module
Author: Nathalia Adriele

Funções auxiliares para o projeto.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def validate_input(data: Dict[str, Any], feature_ranges: Dict[str, tuple]) -> tuple[bool, List[str]]:
    """
    Valida entrada de dados do paciente.

    Args:
        data: Dicionário com dados do paciente
        feature_ranges: Ranges válidos por feature

    Returns:
        Tuple (válido, lista_de_erros)
    """
    errors = []

    for feature, value in data.items():
        if feature in feature_ranges:
            min_val, max_val = feature_ranges[feature]
            if not (min_val <= value <= max_val):
                errors.append(f"{feature} fora do range [{min_val}, {max_val}]")

    is_valid = len(errors) == 0

    return is_valid, errors


def calculate_risk_level(probability: float, thresholds: Dict[str, float]) -> str:
    """
    Calcula nível de risco baseado na probabilidade.

    Args:
        probability: Probabilidade de doença
        thresholds: Thresholds para classificação

    Returns:
        Nível de risco ('low', 'medium', 'high')
    """
    if probability < thresholds['low']:
        return 'Baixo'
    elif probability < thresholds['medium']:
        return 'Médio'
    else:
        return 'Alto'


def format_prediction_result(probability: float, prediction: int) -> Dict[str, Any]:
    """
    Formata resultado da predição.

    Args:
        probability: Probabilidade prevista
        prediction: Predição binária

    Returns:
        Dicionário formatado
    """
    result = {
        'probability': probability,
        'prediction': 'Doença Cardiovascular Detectada' if prediction == 1 else 'Sem Doença Cardiovascular',
        'risk_level': calculate_risk_level(probability, {'low': 0.4, 'medium': 0.7}),
        'confidence': 'Alta' if probability > 0.8 or probability < 0.2 else 'Moderada'
    }

    return result


def load_model_and_preprocessor(model_path: str, preprocessor_path: str) -> tuple:
    """
    Carrega modelo e pré-processador.

    Args:
        model_path: Caminho do modelo
        preprocessor_path: Caminho do pré-processador

    Returns:
        Tuple (modelo, preprocessor)
    """
    import joblib

    model_data = joblib.load(model_path)
    preprocessor_data = joblib.load(preprocessor_path)

    model = model_data['model']
    preprocessor = preprocessor_data['preprocessor']

    logger.info("Modelo e pré-processador carregados")

    return model, preprocessor


def create_patient_data_dict(form_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Cria DataFrame a partir de dados do formulário.

    Args:
        form_data: Dados do formulário

    Returns:
        DataFrame com dados do paciente
    """
    df = pd.DataFrame([form_data])

    logger.info(f"Dados do paciente criados: {df.shape}")

    return df


def get_feature_importance(model: Any, feature_names: List[str], top_n: int = 10) -> pd.DataFrame:
    """
    Obtém importância das features.

    Args:
        model: Modelo treinado
        feature_names: Lista de nomes de features
        top_n: Número de top features

    Returns:
        DataFrame com importâncias
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        logger.warning("Modelo não suporta feature_importances_")
        return pd.DataFrame()

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    return importance_df


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Formata valor como porcentagem.

    Args:
        value: Valor decimal (0-1)
        decimals: Número de casas decimais

    Returns:
        String formatada
    """
    return f"{value * 100:.{decimals_f}f}%".format(decimals=decimals)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Divisão segura com tratamento de divisão por zero.

    Args:
        numerator: Numerador
        denominator: Denominador
        default: Valor padrão se divisão por zero

    Returns:
        Resultado da divisão
    """
    if denominator == 0:
        return default
    return numerator / denominator


def log_model_metrics(metrics: Dict[str, float], model_name: str) -> None:
    """
    Registra métricas do modelo no log.

    Args:
        metrics: Dicionário de métricas
        model_name: Nome do modelo
    """
    logger.info("=" * 50)
    logger.info(f"Métricas do Modelo: {model_name}")
    logger.info("=" * 50)

    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.3f}")

    logger.info("=" * 50)


def create_download_link(df: pd.DataFrame, filename: str = "predictions.csv") -> str:
    """
    Cria link de download para DataFrame.

    Args:
        df: DataFrame para download
        filename: Nome do arquivo

    Returns:
        Link HTML para download
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'

    return href


if __name__ == "__main__":
    # Testar funções
    test_data = {'idade': 45, 'peso': 70, 'altura': 1.75}

    is_valid, errors = validate_input(test_data, {
        'idade': (25, 85),
        'peso': (35, 150),
        'altura': (1.40, 2.10)
    })

    print(f"Válido: {is_valid}, Erros: {errors}")

    risk = calculate_risk_level(0.65, {'low': 0.4, 'medium': 0.7})
    print(f"Risco: {risk}")
