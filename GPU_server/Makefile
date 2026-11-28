# Standalone ExhibitFlow GPU Worker Makefile (Vast.ai / Linux NVIDIA GPU)
#
#   HF_TOKEN=hf_xxx make vast       # Vast.ai one-command: setup + foreground launch + READY banner
#   HF_TOKEN=hf_xxx make            # setup + background worker (banner printed once, token NOT logged)
#   make demo                       # Pure-CPU demo mode (no GPU / weights required)
#
# Options: MODE=full|tracking_seg|tracking  PORT=8001  LOCAL=1  TUNNEL=quick|named|none  NO_TUNNEL=1

SHELL        := /bin/bash
.DEFAULT_GOAL := run

ROOT   := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
ifneq ($(word 2,$(ROOT)),)
$(error Path "$(ROOT)" contains spaces; please move GPU_server to a path without spaces)
endif

EXT    := $(ROOT)/external
WORK   := $(ROOT)/.make
STAMP  := $(WORK)/stamp
PIDF   := $(WORK)/worker.pid
LOG    := $(WORK)/worker.log
DATA   ?= $(ROOT)/gpu_data

PY      ?= python3
PORT    ?= 8001
MODE    ?= full
LOCAL   ?=
TUNNEL  ?= quick
NO_TUNNEL ?=
HF_REPO ?= hungnguyen190204/model_exhibitflow

EFF_TUNNEL := $(if $(NO_TUNNEL),none,$(TUNNEL))

export EF_WEIGHTS_DIR := $(ROOT)/weights
export EXHIBITFLOW_WORK := $(WORK)/work
export PYTORCH_CUDA_ALLOC_CONF := expandable_segments:True

TRACKER_URL := https://github.com/khangkaka066/ByteTrack-DMA-LTC-Motion-Tracker.git
MAP_URL     := https://github.com/potatips/multicam-2d-floorplan-tracker.git

.PHONY: vast run demo setup deps externals weights llama mapmodels check logs status stop restart clean help

help:
	@sed -n "2,8p" Makefile | sed "s/^# \{0,1\}//"

setup: check deps externals weights llama mapmodels

check:
	@$(PY) -c "import torch,sys; ok=torch.cuda.is_available(); print('[check] torch',torch.__version__,'| GPU:',torch.cuda.get_device_name(0) if ok else 'NOT FOUND'); sys.exit(0 if ok else 1)" \
	  || { echo "[check] ERROR: Python with CUDA-enabled PyTorch is required."; exit 1; }

deps: $(STAMP)/deps
$(STAMP)/deps:
	@mkdir -p $(STAMP)
	cd $(ROOT) && $(PY) scripts/install_deps.py
	$(PY) -m pip install -U "brotli>=1.2"
	@touch $@

externals: $(EXT)/ByteTrack-DMA-LTC-Motion-Tracker $(EXT)/multicam-2d-floorplan-tracker
$(EXT)/ByteTrack-DMA-LTC-Motion-Tracker:
	@mkdir -p $(EXT)
	git clone --depth 1 $(TRACKER_URL) $@
$(EXT)/multicam-2d-floorplan-tracker:
	@mkdir -p $(EXT)
	git clone --depth 1 $(MAP_URL) $@

weights: deps
	$(PY) $(ROOT)/scripts/fetch_weights.py $(ROOT)/weights $(HF_REPO)

llama: $(STAMP)/llama
$(STAMP)/llama:
	@mkdir -p $(STAMP)
	@bash $(ROOT)/scripts/install_llama.sh && touch $@ || echo "[llama] skipped - VLM full mode unavailable without llama-server"

mapmodels: $(STAMP)/mapmodels | externals
$(STAMP)/mapmodels:
	@mkdir -p $(STAMP)
	cd $(EXT)/multicam-2d-floorplan-tracker && $(PY) scripts/download_structural_models.py
	@touch $@

vast:
	@PORT=$(PORT) MODE=$(MODE) LOCAL=$(LOCAL) TUNNEL=$(EFF_TUNNEL) DATA_DIR=$(DATA) PY=$(PY) bash $(ROOT)/start_vast.sh

run: setup
	@mkdir -p $(DATA) $(WORK)
	@if [ -f $(PIDF) ] && kill -0 $$(cat $(PIDF)) 2>/dev/null; then \
	  echo "[run] GPU worker already running (pid $$(cat $(PIDF))). Use 'make restart' to restart."; \
	else \
	  fifo="$(WORK)/.banner.fifo"; rm -f "$$fifo"; mkfifo "$$fifo"; \
	  ( cd $(ROOT) && exec $(PY) launch.py --host 0.0.0.0 --port $(PORT) --mode $(MODE) \
	      --tunnel $(EFF_TUNNEL) $(if $(LOCAL),--local) --data-dir $(DATA) --log-file $(LOG) ) > "$$fifo" 2>&1 & \
	  pid=$$!; echo $$pid > $(PIDF); \
	  echo "[run] GPU worker starting (pid $$pid), mode=$(MODE), port=$(PORT). Worker log (token-free): $(LOG)"; \
	  cat "$$fifo" & catpid=$$!; \
	  while kill -0 $$pid 2>/dev/null && ! grep -q "pid=$$pid" /proc/$$pid/stat 2>/dev/null; do sleep 0.5; done; \
	  for _ in $$(seq 1 90); do \
	    if ! kill -0 $$pid 2>/dev/null || ! kill -0 $$catpid 2>/dev/null; then break; fi; \
	    sleep 1; \
	  done; \
	  rm -f "$$fifo"; \
	fi

demo:
	@mkdir -p $(DATA) $(WORK)
	cd $(ROOT) && $(PY) launch.py --demo --host 0.0.0.0 --port $(PORT) --tunnel $(EFF_TUNNEL) --data-dir $(DATA) --log-file $(LOG)

logs:
	@tail -n 80 -f $(LOG)

status:
	@if [ -f $(PIDF) ] && kill -0 $$(cat $(PIDF)) 2>/dev/null; then \
	  echo "running (pid $$(cat $(PIDF)))"; \
	  curl -s -H "X-GPU-Token: $${GPU_TOKEN:-}" http://127.0.0.1:$(PORT)/internal/health || echo "(worker starting)"; \
	else echo "not running"; fi

stop:
	@if [ -f $(PIDF) ] && kill -0 $$(cat $(PIDF)) 2>/dev/null; then kill $$(cat $(PIDF)) && echo "[stop] stopped"; else echo "[stop] not running"; fi; rm -f $(PIDF)

restart: stop run

clean:
	rm -rf $(STAMP)
	@echo "[clean] removed setup stamps"
