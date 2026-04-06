"""
Model Training Module
Author: Nathalia Adriele

Este módulo é responsável pelo treinamento de modelos de machine learning.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib
import logging
from typing import Dict, Any, Tuple
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Classe para treinar e comparar múltiplos modelos.
    """

    def __init__(self, random_state: int = 42):
        """
        Inicializa o ModelTrainer.

        Args:
            random_state: Semente aleatória
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.training_results = {}

    def create_models(self) -> Dict[str, Any]:
        """
        Cria dicionário com todos os modelos a serem testados.

        Returns:
            Dicionário com modelos configurados
        """
        logger.info("Criando modelos de machine learning")

        models = {
            'logistic_regression': LogisticRegression(
                penalty='l2',
                C=1.0,
                class_weight='balanced',
                random_state=self.random_state,
                max_iter=1000
            ),
            'decision_tree': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced',
                random_state=self.random_state
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.random_state
            ),
            'xgboost': XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,
                random_state=self.random_state,
                eval_metric='logloss',
                n_jobs=-1
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='euclidean',
                n_jobs=-1
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=self.random_state
            )
        }

        self.models = models
        logger.info(f"{len(models)} modelos criados")

        return models

    def train_model(self,
                    model_name: str,
                    model: Any,
                    X_train: np.ndarray,
                    y_train: pd.Series) -> Any:
        """
        Treina um modelo específico.

        Args:
            model_name: Nome do modelo
            model: Objeto do modelo
            X_train: Features de treino
            y_train: Target de treino

        Returns:
            Modelo treinado
        """
        logger.info(f"Treinando modelo: {model_name}")

        try:
            model.fit(X_train, y_train)
            logger.info(f"{model_name} treinado com sucesso")
            return model
        except Exception as e:
            logger.error(f"Erro ao treinar {model_name}: {str(e)}")
            raise

    def train_all_models(self,
                         X_train: np.ndarray,
                         y_train: pd.Series) -> Dict[str, Any]:
        """
        Treina todos os modelos.

        Args:
            X_train: Features de treino
            y_train: Target de treino

        Returns:
            Dicionário com modelos treinados
        """
        logger.info("Treinando todos os modelos")

        trained_models = {}

        for name, model in self.models.items():
            try:
                trained_model = self.train_model(name, model, X_train, y_train)
                trained_models[name] = trained_model
            except Exception as e:
                logger.error(f"Falha ao treinar {name}: {str(e)}")

        logger.info(f"{len(trained_models)} modelos treinados com sucesso")

        return trained_models

    def cross_validate_model(self,
                            model: Any,
                            X_train: np.ndarray,
                            y_train: pd.Series,
                            cv: int = 5) -> Dict[str, float]:
        """
        Realiza validação cruzada em um modelo.

        Args:
            model: Modelo a ser validado
            X_train: Features de treino
            y_train: Target de treino
            cv: Número de folds

        Returns:
            Dicionário com métricas de CV
        """
        logger.info(f"Realizando validação cruzada ({cv}-fold)")

        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )

        cv_results = {
            'mean_roc_auc': cv_scores.mean(),
            'std_roc_auc': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }

        logger.info(f"CV ROC-AUC: {cv_results['mean_roc_auc']:.3f} ± {cv_results['std_roc_auc']:.3f}")

        return cv_results

    def hyperparameter_tuning(self,
                            model_name: str,
                            model: Any,
                            X_train: np.ndarray,
                            y_train: pd.Series) -> Any:
        """
        Realiza otimização de hiperparâmetros.

        Args:
            model_name: Nome do modelo
            model: Modelo base
            X_train: Features de treino
            y_train: Target de treino

        Returns:
            Melhor modelo encontrado
        """
        logger.info(f"Otimizando hiperparâmetros: {model_name}")

        # Definir grids de parâmetros
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }

        if model_name not in param_grids:
            logger.warning(f"Não há grid de parâmetros definido para {model_name}")
            return model

        grid_search = GridSearchCV(
            model,
            param_grids[model_name],
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Melhores parâmetros para {model_name}: {grid_search.best_params_}")
        logger.info(f"Melhor score: {grid_search.best_score_:.3f}")

        return grid_search.best_estimator_

    def save_model(self,
                  model: Any,
                  model_name: str,
                  filepath: str,
                  metadata: Dict[str, Any] = None) -> None:
        """
        Salva modelo treinado em disco.

        Args:
            model: Modelo treinado
            model_name: Nome do modelo
            filepath: Caminho para salvar
            metadata: Metadados adicionais
        """
        logger.info(f"Salvando modelo: {model_name}")

        model_data = {
            'model': model,
            'model_name': model_name,
            'training_date': datetime.datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Modelo salvo em {filepath}")


def train_best_model(X_train: np.ndarray,
                     y_train: pd.Series,
                     X_val: np.ndarray,
                     y_val: pd.Series) -> Tuple[Any, Dict]:
    """
    Função auxiliar para treinar o melhor modelo.

    Args:
        X_train: Features de treino
        y_train: Target de treino
        X_val: Features de validação
        y_val: Target de validação

    Returns:
        Tuple (melhor_modelo, resultados)
    """
    logger.info("Iniciando treinamento do melhor modelo")

    trainer = ModelTrainer()
    trainer.create_models()

    # Treinar todos os modelos
    trained_models = trainer.train_all_models(X_train, y_train)

    # Avaliar todos os modelos
    results = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        results[name] = {
            'accuracy': accuracy_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba)
        }

    # Selecionar melhor modelo
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = trained_models[best_model_name]

    logger.info(f"Melhor modelo: {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']:.3f})")

    # Otimizar hiperparâmetros do melhor modelo
    if best_model_name in ['random_forest', 'xgboost', 'gradient_boosting']:
        best_model = trainer.hyperparameter_tuning(best_model_name, best_model, X_train, y_train)

    return best_model, {**results, 'best_model': best_model_name}


if __name__ == "__main__":
    # Exemplo de uso
    # X_train, X_val, y_train, y_val = load_data()
    # best_model, results = train_best_model(X_train, y_train, X_val, y_val)
    # print(f"Melhor modelo: {results['best_model']}")
    pass
