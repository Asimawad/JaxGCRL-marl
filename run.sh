#!/bin/bash
# Cluster entrypoint. Defaults to a single CPPO ant_u_maze run.
#
# Override via env vars (set in manifest.yaml.spec.envs[] or pass to docker run -e):
#   METHOD     = cppo | crl                (default: cppo)
#   ENV        = ant | ant_u_maze | ...    (default: ant_u_maze)
#   STEPS      = total env steps           (default: 25000000)
#   SEED       = int seed                  (default: 1)
#   WANDB_PROJECT, WANDB_ENTITY, WANDB_GROUP, EXP_NAME
#
# All other CPPO/CRL knobs are read from script defaults in scripts/run_cppo.sh
# / scripts/run_crl.sh; override via env vars accepted there.

set -euo pipefail

# Activate uv-managed env if present (in docker we run system-wide python via UV_SYSTEM_PYTHON).
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Wandb setup (optional). If WANDB_API_KEY unset, run with --no-log-wandb forced via scripts.
export WANDB_API_KEY=${WANDB_API_KEY:-}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}

METHOD=${METHOD:-cppo}
ENV=${ENV:-ant_u_maze}
STEPS=${STEPS:-25000000}
SEED=${SEED:-1}

DATE=$(date +%Y-%m-%d)
GROUP=${WANDB_GROUP:-${METHOD}-${ENV}-${DATE}}
EXP_NAME=${EXP_NAME:-${METHOD}-${ENV}-seed${SEED}-${DATE}}

mkdir -p logs/cluster
LOG=logs/cluster/${METHOD}_${ENV}_seed${SEED}_${DATE}.log

echo "=== Cluster run ==="
echo "method=${METHOD} env=${ENV} steps=${STEPS} seed=${SEED}"
echo "exp=${EXP_NAME}"
echo "log=${LOG}"
echo "==================="

case "${METHOD}" in
    cppo)
        ENV=${ENV} STEPS=${STEPS} SEED=${SEED} EXP_NAME=${EXP_NAME} \
        WANDB_PROJECT=${WANDB_PROJECT:-jaxgcrl} \
        LOGFILE=${LOG} \
        bash scripts/run_cppo.sh
        ;;
    crl)
        ENV=${ENV} STEPS=${STEPS} SEED=${SEED} EXP_NAME=${EXP_NAME} \
        WANDB_PROJECT=${WANDB_PROJECT:-jaxgcrl} \
        LOGFILE=${LOG} \
        bash scripts/run_crl.sh
        ;;
    *)
        echo "Unknown METHOD: ${METHOD}. Expected one of: cppo, crl"
        exit 1
        ;;
esac
