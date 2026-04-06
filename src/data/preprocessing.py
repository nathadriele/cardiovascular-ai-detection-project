"""
Data Preprocessing Module
Author: Nathalia Adriele

Este módulo é responsável pelo pré-processamento de dados para machine learning.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Dict, List, Tuple, Optional
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Classe para pré-processamento de dados.
    """

    def __init__(self, target_col: str = 'presenca_doenca_cardiovascular'):
        """
        Inicializa o DataPreprocessor.

        Args:
            target_col: Nome da coluna alvo
        """
        self.target_col = target_col
        self.preprocessor = None
        self.feature_names = None
        self.preprocessing_info = {}

    def separate_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separa features e target.

        Args:
            df: DataFrame completo

        Returns:
            Tuple (X, y)
        """
        logger.info("Separando features e target")

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        logger.info(f"Features: {X.shape[1]}, Target: {y.name}")

        return X, y

    def identify_column_types(self, X: pd.DataFrame) -> Dict[str, List]:
        """
        Identifica tipos de colunas (numéricas, categóricas, binárias).

        Args:
            X: DataFrame de features

        Returns:
            Dicionário com listas de colunas por tipo
        """
        logger.info("Identificando tipos de colunas")

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Identificar colunas binárias (0/1)
        binary_cols = []
        for col in numeric_cols:
            if X[col].nunique() == 2 and set(X[col].dropna().unique()) <= {0, 1}:
                binary_cols.append(col)

        # Remover binárias de numéricas
        numeric_cols = [col for col in numeric_cols if col not in binary_cols]

        # Remover ID se existir
        if 'id_paciente' in numeric_cols:
            numeric_cols.remove('id_paciente')

        column_types = {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'binary': binary_cols
        }

        self.preprocessing_info['column_types'] = column_types

        logger.info(f"Numéricas: {len(numeric_cols)}, Categóricas: {len(categorical_cols)}, Binárias: {len(binary_cols)}")

        return column_types

    def create_preprocessor(self, column_types: Dict[str, List]) -> ColumnTransformer:
        """
        Cria o pipeline de pré-processamento.

        Args:
            column_types: Dicionário com tipos de colunas

        Returns:
            ColumnTransformer configurado
        """
        logger.info("Criando pipeline de pré-processamento")

        # Pipeline para numéricas
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Pipeline para categóricas nominais
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])

        # Combinar transformadores
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, column_types['numeric']),
                ('cat', categorical_transformer, column_types['categorical'])
            ],
            remainder='passthrough'  # Manter binárias inalteradas
        )

        self.preprocessor = preprocessor

        logger.info("Pipeline de pré-processamento criado")

        return preprocessor

    def split_data(self,
                   X: pd.DataFrame,
                   y: pd.Series,
                   test_size: float = 0.15,
                   val_size: float = 0.15,
                   random_state: int = 42) -> Tuple:
        """
        Divide os dados em treino, validação e teste (estratificado).

        Args:
            X: Features
            y: Target
            test_size: Proporção para teste
            val_size: Proporção para validação
            random_state: Semente aleatória

        Returns:
            Tuple (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Dividindo dados em treino/validação/teste")

        # Primeira divisão: treino e temporário
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(test_size + val_size),
            stratify=y,
            random_state=random_state
        )

        # Segunda divisão: validação e teste
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=test_size / (test_size + val_size),
            stratify=y_temp,
            random_state=random_state
        )

        logger.info(f"Treino: {X_train.shape[0]}, Val: {X_val.shape[0]}, Teste: {X_test.shape[0]}")

        self.preprocessing_info['split'] = {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'train_ratio': len(X_train) / len(X),
            'val_ratio': len(X_val) / len(X),
            'test_ratio': len(X_test) / len(X)
        }

        return X_train, X_val, X_test, y_train, y_val, y_test

    def fit_transform(self,
                      X_train: pd.DataFrame,
                      X_val: pd.DataFrame,
                      X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit no treino e transform em treino/val/teste.

        Args:
            X_train: Features de treino
            X_val: Features de validação
            X_test: Features de teste

        Returns:
            Tuple (X_train_proc, X_val_proc, X_test_proc)
        """
        logger.info("Fit e transform dos dados")

        # Fit no treino
        X_train_processed = self.preprocessor.fit_transform(X_train)

        # Transform em val e teste
        X_val_processed = self.preprocessor.transform(X_val)
        X_test_processed = self.preprocessor.transform(X_test)

        # Salvar nomes das features
        self.feature_names = self._get_feature_names()

        logger.info(f"Features processadas: {X_train_processed.shape[1]}")

        return X_train_processed, X_val_processed, X_test_processed

    def _get_feature_names(self) -> List[str]:
        """
        Obtém nomes das features após o pré-processamento.

        Returns:
            Lista de nomes de features
        """
        feature_names = []

        # Features numéricas (nomes originais)
        numeric_features = self.preprocessor.named_transformers_['num'].get_feature_names_out()
        feature_names.extend(numeric_features)

        # Features categóricas (one-hot encoded)
        cat_features = self.preprocessor.named_transformers_['cat'].get_feature_names_out()
        feature_names.extend(cat_features)

        # Features binárias (passthrough)
        binary_features = self.preprocessing_info['column_types']['binary']
        feature_names.extend(binary_features)

        return feature_names

    def save_preprocessor(self, filepath: str) -> None:
        """
        Salva o pré-processador em disco.

        Args:
            filepath: Caminho para salvar
        """
        logger.info(f"Salvando pré-processador em {filepath}")

        joblib.dump({
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'preprocessing_info': self.preprocessing_info
        }, filepath)

    def load_preprocessor(self, filepath: str) -> None:
        """
        Carrega o pré-processador do disco.

        Args:
            filepath: Caminho do arquivo
        """
        logger.info(f"Carregando pré-processador de {filepath}")

        data = joblib.load(filepath)
        self.preprocessor = data['preprocessor']
        self.feature_names = data['feature_names']
        self.preprocessing_info = data['preprocessing_info']


def preprocess_cardiovascular_data(df: pd.DataFrame,
                                   target_col: str = 'presenca_doenca_cardiovascular',
                                   save_preprocessor: bool = True) -> Tuple:
    """
    Função auxiliar para pré-processar dados cardiovascular completos.

    Args:
        df: DataFrame completo
        target_col: Nome da coluna alvo
        save_preprocessor: Se deve salvar o pré-processador

    Returns:
        Tuple (X_train, X_val, X_test, y_train, y_val, y_test, preprocessor)
    """
    logger.info("Iniciando pré-processamento cardiovascular")

    # Inicializar preprocessor
    preprocessor_obj = DataPreprocessor(target_col=target_col)

    # Separar features e target
    X, y = preprocessor_obj.separate_features_target(df)

    # Identificar tipos de colunas
    column_types = preprocessor_obj.identify_column_types(X)

    # Criar pipeline de pré-processamento
    preprocessor_obj.create_preprocessor(column_types)

    # Dividir dados
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor_obj.split_data(X, y)

    # Fit e transform
    X_train_proc, X_val_proc, X_test_proc = preprocessor_obj.fit_transform(X_train, X_val, X_test)

    # Salvar preprocessor se solicitado
    if save_preprocessor:
        preprocessor_obj.save_preprocessor('models/artifacts/preprocessor.pkl')

    return X_train_proc, X_val_proc, X_test_proc, y_train, y_val, y_test, preprocessor_obj


if __name__ == "__main__":
    # Exemplo de uso
    # df = pd.read_csv("data/raw/cardiovascular_data.csv")
    # X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preprocess_cardiovascular_data(df)
    # print(f"Treino: {X_train.shape}")
    # print(f"Features: {preprocessor.feature_names[:10]}")
    pass
