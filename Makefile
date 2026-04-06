.PHONY: help install train test clean predict report download-data setup-environment

help:
	@echo "Comandos Disponíveis:"
	@echo "  make install            - Instala dependências"
	@echo "  make setup-environment  - Configura ambiente virtual"
	@echo "  make download-data      - Baixa dataset de exemplo"
	@echo "  make train              - Treina modelo"
	@echo "  make predict            - Faz predições"
	@echo "  make test               - Executa testes"
	@echo "  make report             - Gera relatório"
	@echo "  make clean              - Limpa artefatos"
	@echo "  make run-app            - Executa aplicação Streamlit"

setup-environment:
	@echo "Criando ambiente virtual..."
	python3 -m venv venv
	@echo "Ativando ambiente virtual..."
	source venv/bin/activate
	@echo "Ambiente configurado. Execute: source venv/bin/activate"

install:
	@echo "Instalando dependências..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "Dependências instaladas"

download-data:
	@echo "Gerando dataset de exemplo..."
	python scripts/generate_sample_data.py
	@echo "Dataset gerado em data/raw/"

train:
	@echo "Treinando modelo..."
	python src/pipeline/training_pipeline.py
	@echo "Modelo treinado"

predict:
	@echo "Fazendo predições..."
	python src/models/predict.py
	@echo "Predições concluídas"

test:
	@echo "Executando testes..."
	pytest tests/ -v --cov=src --cov-report=html
	@echo "Testes concluídos"

report:
	@echo "Gerando relatório..."
	jupyter nbconvert --to html notebooks/04_evaluation.ipynb --output reports/evaluation_report.html
	@echo "Relatório gerado"

clean:
	@echo "Limpando artefatos..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*.log' -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf .mypy_cache
	@echo "Limpeza concluída"

run-app:
	@echo "Iniciando aplicação Streamlit..."
	streamlit run app/Home.py

format:
	@echo "Formatando código..."
	black src/ tests/ app/

lint:
	@echo "Verificando código..."
	flake8 src/ tests/ app/
	mypy src/

type-check:
	@echo "Verificando tipos..."
	mypy src/

all: install train test report
	@echo "Pipeline completo executado"

docker-build:
	@echo "Construindo imagem Docker..."
	docker build -t cardiovascular-ai .

docker-run:
	@echo "Executando container Docker..."
	docker run -p 8501:8501 cardiovascular-ai

docs:
	@echo "Gerando documentação..."
	cd docs && make html

# Comandos para desenvolvimento
dev-install: install
	pip install -r requirements-dev.txt

dev-test: test lint type-check

# Comandos para produção
prod-build:
	@echo "Preparando para produção..."
	python -m pytest tests/ --cov=src --cov-fail-under=80

prod-deploy:
	@echo "Deploy em produção..."
	@echo "Configure seu comando de deploy aqui"
