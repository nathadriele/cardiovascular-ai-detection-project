"""
Model Evaluation Module
Author: Nathalia Adriele

Este módulo é responsável pela avaliação de modelos de machine learning.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Classe para avaliação de modelos.
    """

    def __init__(self):
        """Inicializa o ModelEvaluator."""
        self.evaluation_results = {}

    def calculate_metrics(self,
                         y_true: pd.Series,
                         y_pred: np.ndarray,
                         y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calcula todas as métricas de avaliação.

        Args:
            y_true: Valores reais
            y_pred: Predições
            y_pred_proba: Probabilidades preditas

        Returns:
            Dicionário com métricas calculadas
        """
        logger.info("Calculando métricas de avaliação")

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }

        # Especificidade
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp)

        # Taxas de erro
        metrics['false_positive_rate'] = fp / (fp + tn)
        metrics['false_negative_rate'] = fn / (fn + tp)

        logger.info(f"Métricas calculadas. ROC-AUC: {metrics['roc_auc']:.3f}, "
                   f"Recall: {metrics['recall']:.3f}, F1: {metrics['f1_score']:.3f}")

        return metrics

    def confusion_matrix_analysis(self,
                                 y_true: pd.Series,
                                 y_pred: np.ndarray) -> Dict[str, int]:
        """
        Analisa a matriz de confusão.

        Args:
            y_true: Valores reais
            y_pred: Predições

        Returns:
            Dicionário com valores da matriz de confusão
        """
        logger.info("Analisando matriz de confusão")

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        cm_analysis = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }

        logger.info(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

        return cm_analysis

    def plot_confusion_matrix(self,
                             y_true: pd.Series,
                             y_pred: np.ndarray,
                             save_path: str = None) -> plt.Figure:
        """
        Plota matriz de confusão.

        Args:
            y_true: Valores reais
            y_pred: Predições
            save_path: Caminho para salvar o gráfico

        Returns:
        Figure do matplotlib
        """
        logger.info("Plotando matriz de confusão")

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predito')
        ax.set_ylabel('Real')
        ax.set_title('Matriz de Confusão')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Matriz de confusão salva em {save_path}")

        return fig

    def plot_roc_curve(self,
                      y_true: pd.Series,
                      y_pred_proba: np.ndarray,
                      save_path: str = None) -> plt.Figure:
        """
        Plota curva ROC.

        Args:
            y_true: Valores reais
            y_pred_proba: Probabilidades preditas
            save_path: Caminho para salvar

        Returns:
            Figure do matplotlib
        """
        logger.info("Plotando curva ROC")

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'ROC-AUC = {auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate (1 - Especificidade)')
        ax.set_ylabel('True Positive Rate (Sensibilidade)')
        ax.set_title('Curva ROC')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Curva ROC salva em {save_path}")

        return fig

    def plot_precision_recall_curve(self,
                                    y_true: pd.Series,
                                    y_pred_proba: np.ndarray,
                                    save_path: str = None) -> plt.Figure:
        """
        Plota curva Precision-Recall.

        Args:
            y_true: Valores reais
            y_pred_proba: Probabilidades preditas
            save_path: Caminho para salvar

        Returns:
            Figure do matplotlib
        """
        logger.info("Plotando curva Precision-Recall")

        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision)
        ax.set_xlabel('Recall (Sensibilidade)')
        ax.set_ylabel('Precision')
        ax.set_title('Curva Precision-Recall')
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Curva PR salva em {save_path}")

        return fig

    def generate_classification_report(self,
                                      y_true: pd.Series,
                                      y_pred: np.ndarray) -> str:
        """
        Gera relatório de classificação.

        Args:
            y_true: Valores reais
            y_pred: Predições

        Returns:
            String com relatório
        """
        logger.info("Gerando relatório de classificação")

        report = classification_report(y_true, y_pred,
                                      target_names=['Sem Doença', 'Com Doença'],
                                      digits=3)

        logger.info("Relatório gerado")

        return report

    def compare_models(self,
                      models_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compara resultados de múltiplos modelos.

        Args:
            models_results: Dicionário com resultados por modelo

        Returns:
            DataFrame com comparação
        """
        logger.info("Comparando modelos")

        comparison_df = pd.DataFrame(models_results).T

        # Ordenar por ROC-AUC
        comparison_df = comparison_df.sort_values('roc_auc', ascending=False)

        logger.info("Comparação concluída")

        return comparison_df

    def evaluate_model(self,
                      model: Any,
                      X_test: np.ndarray,
                      y_test: pd.Series,
                      model_name: str = "Model") -> Dict:
        """
        Avaliação completa de um modelo.

        Args:
            model: Modelo treinado
            X_test: Features de teste
            y_test: Target de teste
            model_name: Nome do modelo

        Returns:
            Dicionário com resultados completos
        """
        logger.info(f"Avaliando modelo: {model_name}")

        # Predições
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Métricas
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)

        # Matriz de confusão
        cm_analysis = self.confusion_matrix_analysis(y_test, y_pred)

        # Relatório de classificação
        class_report = self.generate_classification_report(y_test, y_pred)

        results = {
            'model_name': model_name,
            'metrics': metrics,
            'confusion_matrix': cm_analysis,
            'classification_report': class_report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        self.evaluation_results[model_name] = results

        return results


def evaluate_best_model(model: Any,
                       X_test: np.ndarray,
                       y_test: pd.Series,
                       model_name: str = "Best Model") -> Dict:
    """
    Função auxiliar para avaliar o melhor modelo.

    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test: Target de teste
        model_name: Nome do modelo

    Returns:
        Dicionário com resultados de avaliação
    """
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(model, X_test, y_test, model_name)

    logger.info("=" * 50)
    logger.info(f"Resultados Finais - {model_name}")
    logger.info("=" * 50)
    logger.info(f"Acurácia: {results['metrics']['accuracy']:.3f}")
    logger.info(f"Precisão: {results['metrics']['precision']:.3f}")
    logger.info(f"Recall: {results['metrics']['recall']:.3f}")
    logger.info(f"Especificidade: {results['metrics']['specificity']:.3f}")
    logger.info(f"F1-Score: {results['metrics']['f1_score']:.3f}")
    logger.info(f"ROC-AUC: {results['metrics']['roc_auc']:.3f}")
    logger.info("=" * 50)

    return results


if __name__ == "__main__":
    # Exemplo de uso
    # model = load_model()
    # X_test, y_test = load_test_data()
    # results = evaluate_best_model(model, X_test, y_test, "XGBoost")
    pass
