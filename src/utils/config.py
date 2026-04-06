"""
Configuration Module
Author: Nathalia Adriele

Configurações e constantes do projeto.
"""

import os
from pathlib import Path

# Diretórios do projeto
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

# Configuração da página Streamlit
PAGE_CONFIG = {
    "page_title": "Cardiovascular AI Detection",
    "page_icon": "heart_pulse",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Configurações de logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
}

# Features do modelo
CARDIOVASCULAR_FEATURES = [
    'idade',
    'sexo',
    'peso',
    'altura',
    'imc',
    'pressao_arterial_sistolica',
    'pressao_arterial_diastolica',
    'frequencia_cardiaca',
    'glicemia_jejum',
    'colesterol_total',
    'hdl',
    'ldl',
    'triglicerideos',
    'tabagismo',
    'sedentarismo',
    'hipertensao',
    'diabetes',
    'obesidade',
    'dislipidemia',
    'historico_familiar_doenca_cardiovascular'
]

# Ranges de validação
FEATURE_RANGES = {
    'idade': (25, 85),
    'peso': (35, 150),
    'altura': (1.40, 2.10),
    'imc': (12, 50),
    'pressao_arterial_sistolica': (80, 220),
    'pressao_arterial_diastolica': (40, 140),
    'frequencia_cardiaca': (40, 140),
    'glicemia_jejum': (60, 400),
    'colesterol_total': (100, 450),
    'hdl': (20, 120),
    'ldl': (30, 300),
    'triglicerideos': (30, 1000)
}

# Thresholds de risco
RISK_THRESHOLDS = {
    'low': 0.4,
    'medium': 0.7
}

# Metadados do projeto
PROJECT_METADATA = {
    'name': 'Cardiovascular AI Detection',
    'version': '1.0.0',
    'author': 'Nathalia Adriele',
    'description': 'Sistema de detecção de doenças cardiovasculares usando IA',
    'created': '2026-04-02',
    'updated': '2026-04-02'
}
