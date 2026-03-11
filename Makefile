.PHONY: server client clear
server:
	.venv/Scripts/python src/grpc_server.py
client:
	.venv/Scripts/streamlit run app.py
clear:
	cls
	cls
	cls
serverMl:
	uv run mlflow server --port 5000
experiments:
	python src/mlflow_experiments.py