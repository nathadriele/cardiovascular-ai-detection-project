"""
Model Interpretability Module
Author: Nathalia Adriele

Este módulo é responsável pela interpretabilidade de modelos usando SHAP.
"""

import pandas as pd
import numpy as np
import shap
from typing import Dict, List, Any
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInterpreter:
    """
    Classe para interpretação de modelos.
    """

    def __init__(self, model: Any):
        """
        Inicializa o ModelInterpreter.

        Args:
            model: Modelo treinado
        """
        self.model = model
        self.explainer = None
        self.shap_values = None

    def create_explainer(self, X_train: np.ndarray, model_type: str = "tree") -> None:
        """
        Cria explainer SHAP.

        Args:
            X_train: Dados de treino
            model_type: Tipo de modelo ('tree', 'linear', 'deep')
        """
        logger.info(f"Criando explainer SHAP (tipo: {model_type})")

        if model_type == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif model_type == "linear":
            self.explainer = shap.LinearExplainer(self.model, X_train)
        else:
            self.explainer = shap.Explainer(self.model, X_train)

        logger.info("Explainer criado com sucesso")

    def calculate_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula valores SHAP.

        Args:
            X: Features para calcular SHAP

        Returns:
            Array com valores SHAP
        """
        logger.info("Calculando valores SHAP")

        if self.explainer is None:
            raise ValueError("É necessário criar o explainer primeiro")

        self.shap_values = self.explainer.shap_values(X)

        logger.info(f"Valores SHAP calculados: {self.shap_values.shape}")

        return self.shap_values

    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Obtém importância global das features.

        Args:
            feature_names: Lista de nomes de features

        Returns:
            DataFrame com importâncias
        """
        logger.info("Calculando importância global das features")

        if self.shap_values is None:
            raise ValueError("É necessário calcular SHAP values primeiro")

        # Importância média absoluta
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)

        logger.info(f"Top 5 features: {importance_df.head()['feature'].tolist()}")

        return importance_df

    def plot_summary(self, feature_names: List[str], save_path: str = None) -> plt.Figure:
        """
        Plota gráfico summary do SHAP.

        Args:
            feature_names: Lista de nomes de features
            save_path: Caminho para salvar

        Returns:
            Figure do matplotlib
        """
        logger.info("Plotando SHAP summary")

        if self.shap_values is None:
            raise ValueError("É necessário calcular SHAP values primeiro")

        fig, ax = plt.subplots(figsize=(10, 8))

        shap.summary_plot(
            self.shap_values,
            feature_names=feature_names,
            plot_type="bar",
            show=False
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico salvo em {save_path}")

        return fig

    def plot_dependence(self,
                       feature_idx: int,
                       feature_names: List[str],
                       save_path: str = None) -> plt.Figure:
        """
        Plota gráfico de dependência para uma feature.

        Args:
            feature_idx: Índice da feature
            feature_names: Lista de nomes de features
            save_path: Caminho para salvar

        Returns:
            Figure do matplotlib
        """
        logger.info(f"Plotando dependência: {feature_names[feature_idx]}")

        if self.shap_values is None:
            raise ValueError("É necessário calcular SHAP values primeiro")

        fig, ax = plt.subplots(figsize=(10, 6))

        shap.dependence_plot(
            feature_idx,
            self.shap_values,
            feature_names=feature_names,
            show=False
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico salvo em {save_path}")

        return fig

    def explain_prediction(self,
                          instance: np.ndarray,
                          feature_names: List[str]) -> Dict[str, Any]:
        """
        Explica uma predição individual.

        Args:
            instance: Instância a ser explicada
            feature_names: Lista de nomes de features

        Returns:
            Dicionário com explicação
        """
        logger.info("Explicando predição individual")

        if self.explainer is None:
            raise ValueError("É necessário criar o explainer primeiro")

        # Calcular SHAP values para a instância
        shap_values_instance = self.explainer.shap_values(instance)

        # Criar explicação
        explanation = {
            'base_value': float(self.explainer.expected_value),
            'shap_values': shap_values_instance.tolist(),
            'feature_names': feature_names,
            'prediction': float(self.model.predict_proba(instance.reshape(1, -1))[0, 1])
        }

        # Identificar features mais importantes para esta predição
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values_instance
        }).sort_values('shap_value', key=abs, ascending=False)

        explanation['top_features'] = feature_importance.head(5).to_dict('records')

        return explanation

    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        """
        Obtém top N features mais importantes.

        Args:
            n: Número de features

        Returns:
            DataFrame com top features
        """
        if self.shap_values is None:
            raise ValueError("É necessário calcular SHAP values primeiro")

        importance = np.abs(self.shap_values).mean(axis=0)

        top_idx = np.argsort(importance)[-n:][::-1]

        top_features = pd.DataFrame({
            'feature_idx': top_idx,
            'mean_abs_shap': importance[top_idx]
        })

        return top_features


def interpret_model(model: Any,
                   X_train: np.ndarray,
                   X_test: np.ndarray,
                   feature_names: List[str],
                   model_type: str = "tree") -> Dict[str, Any]:
    """
    Função auxiliar para interpretação completa do modelo.

    Args:
        model: Modelo treinado
        X_train: Dados de treino
        X_test: Dados de teste
        feature_names: Lista de nomes de features
        model_type: Tipo de modelo

    Returns:
        Dicionário com resultados de interpretação
    """
    logger.info("Iniciando interpretação do modelo")

    interpreter = ModelInterpreter(model)

    # Criar explainer
    interpreter.create_explainer(X_train, model_type)

    # Calcular SHAP values
    shap_values = interpreter.calculate_shap_values(X_test)

    # Obter importância das features
    feature_importance = interpreter.get_feature_importance(feature_names)

    results = {
        'interpreter': interpreter,
        'shap_values': shap_values,
        'feature_importance': feature_importance,
        'top_features': feature_importance.head(10)
    }

    logger.info("Interpretação concluída")

    return results


if __name__ == "__main__":
    # Exemplo de uso
    # model = load_model()
    # X_train, X_test, feature_names = load_data()
    # results = interpret_model(model, X_train, X_test, feature_names, "tree")
    # print(results['top_features'])
    pass
