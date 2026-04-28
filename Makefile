# Check if GPU is available locally (used to pass --gpus all to docker run).
GPUS := $(shell command -v nvidia-smi > /dev/null && nvidia-smi > /dev/null 2>&1 && echo "--gpus all" || echo "")

BASE_FLAGS = -it --rm
RUN_FLAGS  = $(GPUS) $(BASE_FLAGS)

DOCKER_IMAGE_NAME = genrl-mara-autocurricula
IMAGE             = $(DOCKER_IMAGE_NAME):latest
DOCKER_RUN        = docker run $(RUN_FLAGS) $(IMAGE)

# Cluster build always uses CUDA; local build uses whatever the host has.
build:
	DOCKER_BUILDKIT=1 docker build --tag $(IMAGE) .

run:
	$(DOCKER_RUN) bash run.sh

bash:
	$(DOCKER_RUN) bash

# Quick smoke test: run reacher CPPO for 100k steps inside the image.
smoke:
	$(DOCKER_RUN) python run.py --env reacher --no-log-wandb --total-env-steps 100000 --num-envs 64 --num-evals 2 cppo --rollout-length 64 --unroll-length 64 --batch-size 64 --num-epochs 2 --num-mc-samples 4
