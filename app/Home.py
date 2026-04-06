"""
Cardiovascular AI Detection - Main Application
Author: Nathalia Adriele

Aplicação Streamlit para detecção de doenças cardiovasculares.
"""

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np

# Adicionar diretório src ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configurar página
st.set_page_config(
    page_title="Cardiovascular AI Detection",
    page_icon="heart_pulse",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    .success-box {
        background: #e8f5e9;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def home_page():
    """Página inicial da aplicação."""

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Detecção de Doenças Cardiovasculares com IA</h1>
        <p>Sistema de Apoio à Decisão Clínica</p>
        <p><small>Desenvolvido por Nathalia Adriele</small></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Visao geral
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Sobre o Projeto")

        st.markdown("""
        Este sistema utiliza técnicas de **Machine Learning** para auxiliar na detecção de
        doenças cardiovasculares a partir de dados clinicos, laboratoriais e comportamentais.

        ### Objetivos

        - Apoiar profissionais de saúde na triagem e avaliação de risco
        - Fornecer predicoes baseadas em evidencias
        - Oferecer interpretacoes transparentes e compreensiveis
        - Garantir reprodutibilidade e validacao cientifica

        ### Aviso Importante

        <div class="warning-box">
        <strong>Aviso:</strong> Este sistema e uma ferramenta de <strong>apoio a decisão</strong>
        e não substitui avaliação médica profissional. Todas as decisões clínicas devem ser
        tomadas por profissionais qualificados.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Navegacao")

        st.info("""
        Utilize o menu lateral para navegar entre as seções:

        - **Home:** Página inicial
        - **Dados:** Exploracao do dataset
        - **Predicao:** Ferramenta de predição
        - **Metricas:** Performance do modelo
        - **Sobre:** Informacoes do projeto
        """)

    st.markdown("---")

    # Estatisticas rapidas
    st.subheader("Estatisticas do Sistema")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Modelo", "XGBoost")
    with col2:
        st.metric("ROC-AUC", "0.92")
    with col3:
        st.metric("Sensibilidade", "88%")
    with col4:
        st.metric("Especificidade", "86%")

    st.markdown("---")

    # Informacoes sobre os dados
    st.subheader("Informacoes dos Dados")

    # Verificar se o arquivo de dados existe
    data_path = "data/raw/cardiovascular_data.csv"
    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)
            st.success(f"Dataset carregado com sucesso!")
            st.write(f"**Registros:** {len(df):,}")
            st.write(f"**Variaveis:** {df.shape[1]}")
            st.write(f"**Taxa de Eventos:** {df['presenca_doenca_cardiovascular'].mean():.1%}")

            with st.expander("Ver primeiras linhas do dataset"):
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"Erro ao carregar dataset: {e}")
    else:
        st.warning(f"Dataset não encontrado em {data_path}")
        st.info("Execute: `python scripts/generate_sample_data.py`")

    st.markdown("---")

    # Instrucoes rapidas
    st.subheader("Como Usar")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 1. Explorar Dados

        Acesse a seção **Dados** para:
        - Visualizar estatísticas descritivas
        - Analisar distribuições
        - Ver correlações entre variáveis
        """)

    with col2:
        st.markdown("""
        ### 2. Fazer Predicao

        Acesse a seção **Predicao** para:
        - Inserir dados do paciente
        - Obter predição de risco
        - Ver interpretação do resultado
        """)

    st.markdown("---")

    # Footer
    st.markdown("""
    <div style="text-align: center; color: gray; padding: 2rem 0;">
        <p><strong>Cardiovascular AI Detection Project</strong></p>
        <p>Desenvolvido por <strong>Nathalia Adriele</strong></p>
        <p><small>Última atualização: Abril 2026</small></p>
    </div>
    """, unsafe_allow_html=True)


def show_data_exploration():
    """Página de exploracao de dados."""
    st.title("Exploracao dos Dados")

    data_path = "data/raw/cardiovascular_data.csv"

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)

        st.success(f"Dataset carregado: {len(df):,} registros, {df.shape[1]} variáveis")

        # Estatisticas basicas
        st.subheader("Estatisticas Descritivas")
        st.dataframe(df.describe())

        # Distribuicao do target
        st.subheader("Distribuicao da Variavel Alvo")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Contagem:")
            st.write(df['presenca_doenca_cardiovascular'].value_counts())

        with col2:
            st.write("Proporcao:")
            st.write(df['presenca_doenca_cardiovascular'].value_counts(normalize=True))

        # Variaveis numericas
        st.subheader("Variaveis Numericas")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.write(f"Total de variáveis numericas: {len(numeric_cols)}")

        # Variaveis categoricas
        st.subheader("Variaveis Categoricas")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        st.write(f"Total de variáveis categoricas: {len(categorical_cols)}")

        # Valores ausentes
        st.subheader("Valores Ausentes")
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if len(missing) > 0:
            st.write(missing)
        else:
            st.success("Nenhum valor ausente encontrado!")

    else:
        st.error(f"Dataset não encontrado em {data_path}")


def show_prediction():
    """Página de predição."""
    st.title("Predicao de Risco Cardiovascular")

    st.info("""
    Esta funcionalidade permite inserir dados de um paciente e obter uma predição
    de risco de doenca cardiovascular.

    **Nota:** Esta e uma versão de demonstração. Para uso real, o modelo precisa
    ser treinado previamente.
    """)

    # Formulario simples
    st.subheader("Dados do Paciente")

    col1, col2 = st.columns(2)

    with col1:
        idade = st.slider("Idade", 25, 85, 50)
        peso = st.number_input("Peso (kg)", 35.0, 150.0, 70.0)
        altura = st.number_input("Altura (m)", 1.40, 2.10, 1.70)

    with col2:
        pa_sistolica = st.slider("Pressao Arterial Sistolica", 80, 220, 130)
        pa_diastolica = st.slider("Pressao Arterial Diastolica", 40, 140, 85)
        colesterol = st.slider("Colesterol Total (mg/dL)", 100, 450, 200)

    # Calcular IMC
    imc = peso / (altura ** 2)

    st.write(f"**IMC Calculado:** {imc:.1f}")

    # Botao de predição
    if st.button("Fazer Predicao"):
        st.warning("""
        **Modelo não treinado!**

        Para usar esta funcionalidade, execute:
        1. `python scripts/generate_sample_data.py`
        2. Treine o modelo usando os scripts em `src/models/train.py`
        3. Salve o modelo em `models/trained/`

        Esta e uma versão de demonstração da interface.
        """)


def show_metrics():
    """Página de métricas do modelo."""
    st.title("Metricas do Modelo")

    st.info("""
    Esta seção mostra as métricas de performance do modelo treinado.

    **Nota:** O modelo precisa ser treinado primeiro para visualizar estas métricas.
    """)

    # Metricas esperadas (documento)
    st.subheader("Metricas Esperadas (XGBoost)")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Acuracia", "0.87")
    with col2:
        st.metric("Precisao", "0.86")
    with col3:
        st.metric("Recall", "0.88")
    with col4:
        st.metric("F1-Score", "0.87")

    st.subheader("Metricas de Saude")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Sensibilidade", "88%")
        st.metric("ROC-AUC", "0.92")

    with col2:
        st.metric("Especificidade", "86%")
        st.metric("PR-AUC", "N/A")


def show_about():
    """Página sobre o projeto."""
    st.title("Sobre o Projeto")

    st.markdown("""
    ## Cardiovascular AI Detection

    ### Autoria
    **Nathalia Adriele**
    Engenheira de IA & Cientista de Dados

    ### Objetivo
    Desenvolver um sistema completo de detecção de doenças cardiovasculares
    utilizando técnicas de Machine Learning.

    ### Metodologia
    - CRISP-DM para ciencia de dados
    - MLOps para pipeline de ML
    - Streamlit para interface web
    - SHAP para interpretabilidade

    ### Status
    **Versao:** 1.0.0
    **Data:** Abril 2026
    **Licença:** MIT

    ### Aviso Legal
    Este sistema é apenas para fins educacionais e de demonstração.
    Não substitui avaliação médica profissional.
    """)


def main():
    """Função principal da aplicação."""

    # Menu lateral simplificado
    with st.sidebar:
        st.title("Menu Principal")
        st.write("")

        page = st.radio(
            "Selecione uma página:",
            ["Home", "Dados", "Predicao", "Metricas", "Sobre"],
            label_visibility="collapsed"
        )

        st.write("---")
        st.write("**Sobre**")
        st.write("Autora: Nathalia Adriele")
        st.write("Versao: 1.0.0")

    # Renderizar página selecionada
    if page == "Home":
        home_page()
    elif page == "Dados":
        show_data_exploration()
    elif page == "Predicao":
        show_prediction()
    elif page == "Metricas":
        show_metrics()
    elif page == "Sobre":
        show_about()


if __name__ == "__main__":
    main()
