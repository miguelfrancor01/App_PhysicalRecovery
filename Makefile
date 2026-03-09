.PHONY: server client clear
server:
	uv run python src/grpc_server.py
client:
	uv run streamlit run app.py
clear:
	cls
	cls
	cls