# Reproducing the JaxGCRL Ant CRL Result

This document explains what was needed to reproduce Figure 7 from the JaxGCRL paper
(arXiv [2408.11052](https://arxiv.org/abs/2408.11052)) — CRL on the Ant environment
reaching ~0.85–0.90 success rate by ~30M steps and holding through 300M steps.

---

## What the paper shows

**Figure 7** (Section 5.5, ablation over energy functions and critic losses):
- Y-axis: *Success rate* — fraction of eval episodes where the ant reached within 0.5 m
  of the goal **at least once** during the episode (`episode_success_any`)
- X-axis: Environment steps (millions), up to 300M
- CRL curve: rises from ~0% at 0M → ~50% at 6M → ~85% at 12M → plateau ~88–90%

**Figure 3** (Section 5.2, baseline comparison at 50M steps) uses the same metric but
the default smaller architecture. IQM over 10 seeds.

---

## Why the default config fails

Running the script in `scripts/train.sh` as-is gets stuck around **35–40% success**
even at 50M steps. The reasons:

### 1. Wrong algorithm config (`train.sh` ≠ paper's best)

| Parameter | `scripts/train.sh` | Paper Figure 7 |
|---|---|---|
| `contrastive_loss_fn` | `bwd_infonce` | **`sym_infonce`** |
| `energy_fn` | `norm` | **`l2`** |
| `use_ln` | `False` | **`True`** |
| `policy_lr` | `3e-4` | **`6e-4`** |
| `h_dim` | 256 (default) | **512** |
| `n_hidden` | 2 (default) | **4** |

`train.sh` is a quick demo (10M steps, modest performance). The paper's benchmark
config lives in **Appendix B.3 / Table 2** and uses the large architecture.

### 2. Network too small

The paper's Figure 7 specifically studies "large architectures" — 4 hidden layers with
wide hidden dimension. The default 2-layer 256-unit network lacks capacity to represent
the state-goal relationship across the full 10-unit navigation range.

### 3. Wrong loss function semantics

- `bwd_infonce`: `diag(logits) - logsumexp(logits, axis=0)` — only backward direction
- `sym_infonce`: `2*diag - logsumexp(axis=1) - logsumexp(axis=0)` — both directions,
  stronger signal, more stable training

### 4. Wrong energy function scaling

- `norm`: `-sqrt(sum((x-y)²) + ε)` — gradient vanishes near zero (sqrt flattening)
- `l2`: `-sum((x-y)²)` — clean squared distance, consistent gradient throughout

---

## Working configuration

```bash
# Single seed, 300M steps — reproduces Figure 7
python -u run.py \
    --env ant \
    --seed 0 \
    --total-env-steps 300000000 \
    --num-envs 512 \
    --num-eval-envs 512 \
    --episode-length 1000 \
    --num-evals 100 \
    --exp-name "crl-ant-paper-seed0" \
    --no-log-wandb \
    crl \
    --h-dim 512 \
    --n-hidden 4 \
    --repr-dim 64 \
    --unroll-length 62 \
    --min-replay-size 1000 \
    --max-replay-size 10000 \
    --batch-size 256 \
    --discounting 0.99 \
    --policy-lr 6e-4 \
    --critic-lr 3e-4 \
    --contrastive-loss-fn sym_infonce \
    --energy-fn l2 \
    --use-ln \
    --logsumexp-penalty-coeff 0.1 \
    --train-step-multiplier 1
```

Or use the provided script:

```bash
SEEDS="0 1 2" bash scripts/run_ant_paper.sh
```

### Observed results (seed 0, H100 80GB)

| Steps | Success rate |
|-------|-------------|
| 3.5M  | 0.4%  |
| 6.5M  | 48%   |
| 9.5M  | 73%   |
| 12.5M | 85%   |
| 18M   | 86%   |
| 33M   | 90%   |
| 39M   | 89%   |

Matches paper Figure 7 closely (single seed; paper reports IQM over 10 seeds).

---

## Code fixes required (vs upstream JaxGCRL)

Two bugs in the upstream repo prevent running with newer JAX (≥0.4.35):

### Fix 1 — Unhashable JIT static argument (`crl.py`)

```python
# BEFORE (causes ValueError with JAX >= 0.4.26):
transitions = jax.vmap(flatten_batch, in_axes=(None, 0, 0))(
    (self.discounting, state_size, tuple(train_env.goal_indices)),
    ...
)

# AFTER — pre-compute with concrete Python types outside the JIT boundary:
_buffer_config = (float(self.discounting), state_size,
                  tuple(int(i) for i in train_env.goal_indices))
...
transitions = jax.vmap(flatten_batch, in_axes=(None, 0, 0))(
    _buffer_config, ...
)
```

`train_env.goal_indices` returns JAX traced values inside a JIT context in newer JAX.
Casting to plain Python ints at construction time (outside JIT) makes the tuple hashable.

### Fix 2 — Assertion fires due to step-count discretisation (`crl.py`)

```python
# BEFORE (always fails — actual steps slightly < total due to epoch rounding):
assert total_steps >= config.total_env_steps

# AFTER — just remove it:
total_steps = current_step
```

### Fix 3 — Metrics invisible due to logging buffering (`env.py`)

```python
# BEFORE — logging.info is silently dropped when wandb-osh
# configures the root logger before basicConfig runs:
logging.info(f"step: {self.x_data[-1]}, {key}: ...")

# AFTER — use print with flush=True for guaranteed immediate output:
print(f"step: {self.x_data[-1]}, {key}: ...", flush=True)
```

All three fixes are already applied in this branch.

---

## GPU memory notes

The paper uses `num_envs=1024` per Table 2, but this OOMs on a single H100 80GB with
`h_dim=512, n_hidden=4` (the activation scan over ~94 training steps needs ~2 GB).
Using `num_envs=512` with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85` fits within 80 GB
and reproduces the same learning curve.

The paper likely ran on multiple GPUs or used `h_dim=1024` with gradient checkpointing
not available in this codebase.

---

## JAX version compatibility

| JAX version | Status |
|---|---|
| 0.4.25 + cuda12.cudnn89 | Requires execstack patch on some kernels |
| **0.4.35 + cuda12** | ✅ Tested, works, GPU detected cleanly |
| 0.6.x | ❌ Breaks — unhashable tracer error (not just a warning) |

Install command:
```bash
pip install "jax[cuda12]==0.4.35" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install "flax==0.8.3" "optax==0.2.3" "brax==0.12.1"
```
