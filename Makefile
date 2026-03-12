.PHONY: server client clear serverMl experiments
server:
	.venv/Scripts/python src/grpc_server.py
client:
	.venv/Scripts/python -m streamlit run app.py
clear:
	cls
	cls
	cls
serverMl:
	uv run mlflow server --port 5000
experiments:
	python src/mlflow_experiments.py
