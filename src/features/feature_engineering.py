"""
Feature Engineering Module
Author: Nathalia Adriele

Este módulo é responsável pela engenharia de atributos para o modelo cardiovascular.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Classe para engenharia de features.
    """

    def __init__(self):
        """Inicializa o FeatureEngineer."""
        self.features_created = []
        self.feature_descriptions = {}

    def calculate_bmi(self, df: pd.DataFrame, peso_col: str = 'peso', altura_col: str = 'altura') -> pd.Series:
        """
        Calcula IMC (Índice de Massa Corporal).

        Args:
            df: DataFrame
            peso_col: Nome da coluna de peso
            altura_col: Nome da coluna de altura

        Returns:
            Series com IMC calculado
        """
        logger.info("Calculando IMC")

        imc = df[peso_col] / (df[altura_col] ** 2)

        self.features_created.append('imc')
        self.feature_descriptions['imc'] = 'Índice de Massa Corporal (kg/m²)'

        return imc

    def categorize_bmi(self, imc: pd.Series) -> pd.Series:
        """
        Categoriza IMC em faixas.

        Args:
            imc: Series com valores de IMC

        Returns:
            Series com categorias de IMC
        """
        logger.info("Categorizando IMC")

        bins = [0, 18.5, 25, 30, 35, 40, np.inf]
        labels = ['abaixo_peso', 'normal', 'sobrepeso', 'obesidade_i', 'obesidade_ii', 'obesidade_iii']

        imc_cat = pd.cut(imc, bins=bins, labels=labels, right=False)

        self.features_created.append('imc_categoria')
        self.feature_descriptions['imc_categoria'] = 'Categoria do IMC'

        return imc_cat

    def categorize_age(self, idade: pd.Series) -> pd.Series:
        """
        Categoriza idade em faixas etárias.

        Args:
            idade: Series com valores de idade

        Returns:
            Series com faixas etárias
        """
        logger.info("Categorizando idade")

        bins = [0, 35, 45, 55, 65, 75, np.inf]
        labels = ['25-34', '35-44', '45-54', '55-64', '65-74', '75+']

        idade_cat = pd.cut(idade, bins=bins, labels=labels, right=False)

        self.features_created.append('faixa_etaria')
        self.feature_descriptions['faixa_etaria'] = 'Faixa etária'

        return idade_cat

    def categorize_blood_pressure(self, df: pd.DataFrame,
                                   sistolica_col: str = 'pressao_arterial_sistolica',
                                   diastolica_col: str = 'pressao_arterial_diastolica') -> pd.Series:
        """
        Categoriza pressão arterial segundo diretrizes.

        Args:
            df: DataFrame
            sistolica_col: Nome da coluna sistólica
            diastolica_col: Nome da coluna diastólica

        Returns:
            Series com categorias de PA
        """
        logger.info("Categorizando pressão arterial")

        sistolica = df[sistolica_col]
        diastolica = df[diastolica_col]

        categorias = []

        for sys, dia in zip(sistolica, diastolica):
            if sys < 120 and dia < 80:
                categorias.append('normal')
            elif (sys >= 120 and sys < 140) or (dia >= 80 and dia < 90):
                categorias.append('pre_hipertensao')
            elif (sys >= 140 and sys < 160) or (dia >= 90 and dia < 100):
                categorias.append('hipertensao_i')
            else:
                categorias.append('hipertensao_ii')

        pa_cat = pd.Series(categorias, index=df.index)

        self.features_created.append('pressao_arterial_categoria')
        self.feature_descriptions['pressao_arterial_categoria'] = 'Categoria da pressão arterial'

        return pa_cat

    def calculate_non_hdl_cholesterol(self, df: pd.DataFrame,
                                       total_col: str = 'colesterol_total',
                                       hdl_col: str = 'hdl') -> pd.Series:
        """
        Calcula colesterol não-HDL.

        Args:
            df: DataFrame
            total_col: Nome da coluna de colesterol total
            hdl_col: Nome da coluna de HDL

        Returns:
            Series com colesterol não-HDL
        """
        logger.info("Calculando colesterol não-HDL")

        non_hdl = df[total_col] - df[hdl_col]

        self.features_created.append('colesterol_nao_hdl')
        self.feature_descriptions['colesterol_nao_hdl'] = 'Colesterol não-HDL (mg/dL)'

        return non_hdl

    def calculate_ratios(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calcula ratios lipídicos importantes.

        Args:
            df: DataFrame

        Returns:
            Dicionário com ratios calculados
        """
        logger.info("Calculando ratios lipídicos")

        ratios = {}

        # Ratio LDL/HDL
        if 'ldl' in df.columns and 'hdl' in df.columns:
            ratios['ratio_ldl_hdl'] = df['ldl'] / df['hdl']
            self.feature_descriptions['ratio_ldl_hdl'] = 'Ratio LDL/HDL'

        # Ratio Triglicerídeos/HDL
        if 'triglicerideos' in df.columns and 'hdl' in df.columns:
            ratios['ratio_triglicerides_hdl'] = df['triglicerideos'] / df['hdl']
            self.feature_descriptions['ratio_triglicerides_hdl'] = 'Ratio Triglicerídeos/HDL'

        # Ratio Colesterol Total/HDL
        if 'colesterol_total' in df.columns and 'hdl' in df.columns:
            ratios['ratio_total_hdl'] = df['colesterol_total'] / df['hdl']
            self.feature_descriptions['ratio_total_hdl'] = 'Ratio Colesterol Total/HDL'

        self.features_created.extend(ratios.keys())

        return ratios

    def create_metabolic_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Cria score de risco metabólico simples.

        Args:
            df: DataFrame

        Returns:
            Series com score metabólico
        """
        logger.info("Criando score metabólico")

        componentes = []

        if 'hipertensao' in df.columns:
            componentes.append(df['hipertensao'])
        if 'diabetes' in df.columns:
            componentes.append(df['diabetes'])
        if 'obesidade' in df.columns:
            componentes.append(df['obesidade'])
        if 'dislipidemia' in df.columns:
            componentes.append(df['dislipidemia'])

        if componentes:
            score = pd.concat(componentes, axis=1).sum(axis=1)
        else:
            score = pd.Series(0, index=df.index)

        self.features_created.append('score_metabolico')
        self.feature_descriptions['score_metabolico'] = 'Score de risco metabólico (0-4)'

        return score

    def create_behavioral_risk_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Cria score de risco comportamental.

        Args:
            df: DataFrame

        Returns:
            Series com score comportamental
        """
        logger.info("Criando score de risco comportamental")

        score = pd.Series(0, index=df.index)

        # Tabagismo
        if 'tabagismo' in df.columns:
            score += df['tabagismo'].map({
                'nunca': 0,
                'ex_fumante': 0.5,
                'fumante_atual': 1
            }).fillna(0)

        # Sedentarismo
        if 'sedentarismo' in df.columns:
            score += df['sedentarismo']

        # Alcoolismo
        if 'alcoolismo' in df.columns:
            score += df['alcoolismo'].map({
                'nunca': 0,
                'social': 0.25,
                'moderado': 0.5,
                'pesado': 1
            }).fillna(0)

        self.features_created.append('score_risco_comportamental')
        self.feature_descriptions['score_risco_comportamental'] = 'Score de risco comportamental (0-3)'

        return score

    def create_interactions(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Cria interações entre features importantes.

        Args:
            df: DataFrame

        Returns:
            Dicionário com interações criadas
        """
        logger.info("Criando interações entre features")

        interactions = {}

        # Idade x Hipertensão
        if 'idade' in df.columns and 'hipertensao' in df.columns:
            interactions['idade_x_hipertensao'] = df['idade'] * df['hipertensao']
            self.feature_descriptions['idade_x_hipertensao'] = 'Interação: idade * hipertensão'

        # Idade x Obesidade
        if 'idade' in df.columns and 'obesidade' in df.columns:
            interactions['idade_x_obesidade'] = df['idade'] * df['obesidade']
            self.feature_descriptions['idade_x_obesidade'] = 'Interação: idade * obesidade'

        # Pressão Sistólica x Idade
        if 'pressao_arterial_sistolica' in df.columns and 'idade' in df.columns:
            interactions['pa_sistolica_x_idade'] = df['pressao_arterial_sistolica'] * df['idade']
            self.feature_descriptions['pa_sistolica_x_idade'] = 'Interação: PA sistólica * idade'

        self.features_created.extend(interactions.keys())

        return interactions

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todas as transformações de feature engineering.

        Args:
            df: DataFrame original

        Returns:
            DataFrame com novas features
        """
        logger.info("Aplicando feature engineering completo")

        df_transformed = df.copy()

        # Calcular IMC se não existir
        if 'imc' not in df_transformed.columns:
            if 'peso' in df_transformed.columns and 'altura' in df_transformed.columns:
                df_transformed['imc'] = self.calculate_bmi(df_transformed)

        # Categorizar IMC
        if 'imc' in df_transformed.columns:
            df_transformed['imc_categoria'] = self.categorize_bmi(df_transformed['imc'])

        # Categorizar idade
        if 'idade' in df_transformed.columns:
            df_transformed['faixa_etaria'] = self.categorize_age(df_transformed['idade'])

        # Categorizar pressão arterial
        if all(col in df_transformed.columns for col in ['pressao_arterial_sistolica', 'pressao_arterial_diastolica']):
            df_transformed['pressao_arterial_categoria'] = self.categorize_blood_pressure(df_transformed)

        # Calcular colesterol não-HDL
        if all(col in df_transformed.columns for col in ['colesterol_total', 'hdl']):
            df_transformed['colesterol_nao_hdl'] = self.calculate_non_hdl_cholesterol(df_transformed)

        # Calcular ratios
        ratios = self.calculate_ratios(df_transformed)
        for name, series in ratios.items():
            df_transformed[name] = series

        # Criar scores
        if any(col in df_transformed.columns for col in ['hipertensao', 'diabetes', 'obesidade', 'dislipidemia']):
            df_transformed['score_metabolico'] = self.create_metabolic_score(df_transformed)

        if any(col in df_transformed.columns for col in ['tabagismo', 'sedentarismo', 'alcoolismo']):
            df_transformed['score_risco_comportamental'] = self.create_behavioral_risk_score(df_transformed)

        # Criar interações
        interactions = self.create_interactions(df_transformed)
        for name, series in interactions.items():
            df_transformed[name] = series

        logger.info(f"Feature engineering concluído. {len(self.features_created)} features criadas.")

        return df_transformed


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função auxiliar para aplicar feature engineering completo.

    Args:
        df: DataFrame original

    Returns:
        DataFrame com features engenheiradas
    """
    engineer = FeatureEngineer()
    df_transformed = engineer.transform(df)

    return df_transformed


if __name__ == "__main__":
    # Exemplo de uso
    # df = pd.read_csv("data/raw/cardiovascular_data.csv")
    # df_engineered = apply_feature_engineering(df)
    # print(f"Features originais: {df.shape[1]}")
    # print(f"Features criadas: {df_engineered.shape[1]}")
    pass
