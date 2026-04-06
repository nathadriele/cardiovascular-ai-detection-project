"""
Data Validation Module
Author: Nathalia Adriele

Este módulo é responsável pela validação da qualidade dos dados.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """
    Classe para validar a qualidade dos dados.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Inicializa o DataValidator.

        Args:
            df: DataFrame a ser validado
        """
        self.df = df.copy()
        self.validation_results = {}

    def check_completeness(self, threshold: float = 0.30) -> Dict[str, float]:
        """
        Verifica a completude dos dados (valores ausentes).

        Args:
            threshold: Limiar para considerar variável com muitos ausentes

        Returns:
            Dicionário com taxa de ausentes por variável
        """
        logger.info("Verificando completude dos dados")

        missing_rates = (self.df.isnull().sum() / len(self.df)).to_dict()

        # Identificar colunas com muitos ausentes
        high_missing = {col: rate for col, rate in missing_rates.items()
                       if rate > threshold}

        self.validation_results['completeness'] = {
            'missing_rates': missing_rates,
            'high_missing_columns': high_missing,
            'total_missing_cells': self.df.isnull().sum().sum(),
            'overall_completeness': (1 - self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1]))
        }

        logger.info(f"Taxa geral de completude: {self.validation_results['completeness']['overall_completeness']:.2%}")

        return missing_rates

    def check_consistency(self) -> Dict[str, List]:
        """
        Verifica consistências lógicas nos dados.

        Returns:
            Dicionário com inconsistências encontradas
        """
        logger.info("Verificando consistência lógica dos dados")

        inconsistencies = {
            'pa_sistolica_menor_diastolica': [],
            'imc_fora_faixa': [],
            'idade_fora_faixa': [],
            'valores_impossiveis': []
        }

        # Verificar PA sistólica >= diastólica
        if 'pressao_arterial_sistolica' in self.df.columns and 'pressao_arterial_diastolica' in self.df.columns:
            mask = self.df['pressao_arterial_sistolica'] < self.df['pressao_arterial_diastolica']
            inconsistencies['pa_sistolica_menor_diastolica'] = self.df[mask].index.tolist()

        # Verificar IMC
        if 'imc' in self.df.columns:
            mask = (self.df['imc'] < 12) | (self.df['imc'] > 60)
            inconsistencies['imc_fora_faixa'] = self.df[mask].index.tolist()

        # Verificar idade
        if 'idade' in self.df.columns:
            mask = (self.df['idade'] < 0) | (self.df['idade'] > 120)
            inconsistencies['idade_fora_faixa'] = self.df[mask].index.tolist()

        self.validation_results['consistency'] = inconsistencies

        total_inconsistencies = sum(len(v) for v in inconsistencies.values())
        logger.info(f"Total de inconsistências encontradas: {total_inconsistencies}")

        return inconsistencies

    def check_duplicates(self) -> Dict[str, int]:
        """
        Verifica duplicatas nos dados.

        Returns:
            Dicionário com informações sobre duplicatas
        """
        logger.info("Verificando duplicatas")

        duplicate_info = {
            'n_duplicate_rows': self.df.duplicated().sum(),
            'n_duplicate_ids': 0,
            'duplicate_percentage': (self.df.duplicated().sum() / len(self.df)) * 100
        }

        # Verificar duplicatas por ID se existir
        if 'id_paciente' in self.df.columns:
            duplicate_info['n_duplicate_ids'] = self.df['id_paciente'].duplicated().sum()

        self.validation_results['duplicates'] = duplicate_info

        logger.info(f"Linhas duplicadas: {duplicate_info['n_duplicate_rows']} ({duplicate_info['duplicate_percentage']:.2f}%)")

        return duplicate_info

    def check_ranges(self, ranges: Dict[str, Tuple[float, float]]) -> Dict[str, List]:
        """
        Verifica se os valores estão dentro dos ranges esperados.

        Args:
            ranges: Dicionário com {coluna: (min, max)}

        Returns:
            Dicionário com índices fora do range por coluna
        """
        logger.info("Verificando ranges de valores")

        out_of_range = {}

        for col, (min_val, max_val) in ranges.items():
            if col in self.df.columns:
                mask = (self.df[col] < min_val) | (self.df[col] > max_val)
                out_of_range[col] = self.df[mask].index.tolist()
                logger.info(f"{col}: {len(out_of_range[col])} valores fora do range [{min_val}, {max_val}]")

        self.validation_results['ranges'] = out_of_range

        return out_of_range

    def check_data_types(self, expected_types: Dict[str, str]) -> Dict[str, bool]:
        """
        Verifica se os tipos de dados estão conforme esperado.

        Args:
            expected_types: Dicionário com {coluna: tipo_esperado}

        Returns:
            Dicionário indicando se cada tipo está correto
        """
        logger.info("Verificando tipos de dados")

        type_check = {}

        for col, expected_type in expected_types.items():
            if col in self.df.columns:
                actual_type = str(self.df[col].dtype)
                type_check[col] = expected_type in actual_type

                if not type_check[col]:
                    logger.warning(f"{col}: esperado {expected_type}, encontrado {actual_type}")

        self.validation_results['data_types'] = type_check

        return type_check

    def generate_validation_report(self) -> pd.DataFrame:
        """
        Gera um DataFrame resumo com todos os problemas de validação.

        Returns:
            DataFrame com o relatório de validação
        """
        logger.info("Gerando relatório de validação")

        report_data = []

        # Completude
        if 'completeness' in self.validation_results:
            for col, rate in self.validation_results['completeness']['missing_rates'].items():
                report_data.append({
                    'variavel': col,
                    'tipo_problema': 'valor_ausente',
                    'severidade': 'alta' if rate > 0.2 else 'media' if rate > 0.1 else 'baixa',
                    'detalhes': f'{rate:.1%} ausentes'
                })

        # Consistência
        if 'consistency' in self.validation_results:
            for problem_type, indices in self.validation_results['consistency'].items():
                if len(indices) > 0:
                    report_data.append({
                        'variavel': 'multiple',
                        'tipo_problema': problem_type,
                        'severidade': 'alta',
                        'detalhes': f'{len(indices)} registros'
                    })

        report_df = pd.DataFrame(report_data)

        logger.info(f"Relatório gerado: {len(report_df)} problemas identificados")

        return report_df


def validate_cardiovascular_data(df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    Função auxiliar para validar dados cardiovascular.

    Args:
        df: DataFrame a ser validado

    Returns:
        Tuple (dados_validos, relatorio_validacao)
    """
    validator = DataValidator(df)

    # Definir ranges esperados
    ranges = {
        'idade': (25, 85),
        'imc': (15, 50),
        'pressao_arterial_sistolica': (80, 220),
        'pressao_arterial_diastolica': (40, 140),
        'frequencia_cardiaca': (40, 140),
        'glicemia_jejum': (60, 400),
        'colesterol_total': (100, 450),
        'hdl': (20, 120),
        'ldl': (30, 300)
    }

    # Executar validações
    validator.check_completeness(threshold=0.30)
    validator.check_consistency()
    validator.check_duplicates()
    validator.check_ranges(ranges)

    # Gerar relatório
    report = validator.generate_validation_report()

    # Dados são válidos se não houver problemas de alta severidade
    is_valid = len(report[report['severidade'] == 'alta']) == 0

    return is_valid, validator.validation_results


if __name__ == "__main__":
    # Exemplo de uso
    # df = pd.read_csv("data/raw/cardiovascular_data.csv")
    # validator = DataValidator(df)
    # validator.check_completeness()
    # validator.check_consistency()
    # validator.check_duplicates()
    # report = validator.generate_validation_report()
    # print(report)
    pass
