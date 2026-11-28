PYTHON ?= python3
PORT ?= 8000
HOST ?= 127.0.0.1
DATA_DIR ?= ./backend_data
CORS_ORIGINS ?= *

.PHONY: run run-local tunnel named-tunnel test

run:
	$(PYTHON) launch.py --host $(HOST) --port $(PORT) --data-dir $(DATA_DIR) --cors-origins "$(CORS_ORIGINS)"

run-local:
	$(PYTHON) launch.py --host $(HOST) --port $(PORT) --data-dir $(DATA_DIR) --no-tunnel --no-auth

tunnel:
	$(PYTHON) launch.py --host $(HOST) --port $(PORT) --data-dir $(DATA_DIR) --tunnel quick --cors-origins "$(CORS_ORIGINS)"

named-tunnel:
	$(PYTHON) launch.py --host $(HOST) --port $(PORT) --data-dir $(DATA_DIR) --tunnel named --cors-origins "$(CORS_ORIGINS)"

test:
	PYTHONPATH=.. pytest tests -v
