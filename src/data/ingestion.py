"""
Data Ingestion Module
Author: Nathalia Adriele

Este módulo é responsável pela ingestão e carregamento de dados de diversas fontes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Classe para gerenciar a ingestão de dados de diferentes fontes.
    """

    def __init__(self, data_dir: str = "data/raw"):
        """
        Inicializa o DataIngestion.

        Args:
            data_dir: Diretório onde os dados brutos estão armazenados
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_csv(self,
                 filename: str,
                 **kwargs) -> pd.DataFrame:
        """
        Carrega dados de um arquivo CSV.

        Args:
            filename: Nome do arquivo CSV
            **kwargs: Argumentos adicionais para pd.read_csv

        Returns:
            DataFrame com os dados carregados
        """
        filepath = self.data_dir / filename

        try:
            logger.info(f"Carregando dados de {filepath}")
            df = pd.read_csv(filepath, **kwargs)
            logger.info(f"Dados carregados com sucesso: {df.shape[0]} linhas, {df.shape[1]} colunas")
            return df
        except FileNotFoundError:
            logger.error(f"Arquivo não encontrado: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Erro ao carregar arquivo: {str(e)}")
            raise

    def load_excel(self,
                   filename: str,
                   sheet_name: Union[str, int] = 0,
                   **kwargs) -> pd.DataFrame:
        """
        Carrega dados de um arquivo Excel.

        Args:
            filename: Nome do arquivo Excel
            sheet_name: Nome ou índice da planilha
            **kwargs: Argumentos adicionais para pd.read_excel

        Returns:
            DataFrame com os dados carregados
        """
        filepath = self.data_dir / filename

        try:
            logger.info(f"Carregando dados de {filepath}, sheet: {sheet_name}")
            df = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
            logger.info(f"Dados carregados com sucesso: {df.shape[0]} linhas, {df.shape[1]} colunas")
            return df
        except FileNotFoundError:
            logger.error(f"Arquivo não encontrado: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Erro ao carregar arquivo: {str(e)}")
            raise

    def load_parquet(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Carrega dados de um arquivo Parquet.

        Args:
            filename: Nome do arquivo Parquet
            **kwargs: Argumentos adicionais para pd.read_parquet

        Returns:
            DataFrame com os dados carregados
        """
        filepath = self.data_dir / filename

        try:
            logger.info(f"Carregando dados de {filepath}")
            df = pd.read_parquet(filepath, **kwargs)
            logger.info(f"Dados carregados com sucesso: {df.shape[0]} linhas, {df.shape[1]} colunas")
            return df
        except FileNotFoundError:
            logger.error(f"Arquivo não encontrado: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Erro ao carregar arquivo: {str(e)}")
            raise

    def save_data(self,
                  df: pd.DataFrame,
                  filename: str,
                  format: str = 'csv') -> None:
        """
        Salva DataFrame em arquivo.

        Args:
            df: DataFrame a ser salvo
            filename: Nome do arquivo de saída
            format: Formato do arquivo ('csv', 'parquet', 'excel')
        """
        output_path = self.data_dir / filename

        try:
            if format == 'csv':
                df.to_csv(output_path, index=False)
            elif format == 'parquet':
                df.to_parquet(output_path, index=False)
            elif format == 'excel':
                df.to_excel(output_path, index=False)
            else:
                raise ValueError(f"Formato não suportado: {format}")

            logger.info(f"Dados salvos com sucesso em {output_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar dados: {str(e)}")
            raise


def load_cardiovascular_data(filepath: str) -> pd.DataFrame:
    """
    Função auxiliar para carregar o dataset cardiovascular.

    Args:
        filepath: Caminho para o arquivo de dados

    Returns:
        DataFrame com os dados cardiovascular
    """
    logger.info(f"Carregando dataset cardiovascular de {filepath}")

    # Tentar diferentes formatos
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Formato de arquivo não suportado: {filepath}")

    logger.info(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")

    return df


if __name__ == "__main__":
    # Exemplo de uso
    ingestion = DataIngestion()

    # Carregar dados (exemplo)
    # df = ingestion.load_csv("cardiovascular_data.csv")
    # print(df.head())
    # print(df.info())
