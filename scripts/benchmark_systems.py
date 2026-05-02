"""Wall-clock + memory benchmark for CPPO (optimized & baseline), ICRL, IPPO.

Spawns each system as a subprocess so JAX compilation, GPU memory, and Hydra
config are fully isolated between runs. Parses SPS / compile_seconds /
wall_clock_seconds from the system's terminal logger and polls nvidia-smi for
peak GPU memory while each system runs.

Example:
    python scripts/benchmark_systems.py \\
        --num_envs 512 --num_updates 80 --num_evaluation 8 \\
        --rollout_length 128 --task smacv2_5_units \\
        --systems cppo_optimized cppo_baseline icrl ippo \\
        --out benchmark_results.json

The first eval of each run is treated as compile-contaminated and excluded
from the steady-state SPS mean/std.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Thread
from typing import Any

import numpy as np


SYSTEMS: dict[str, dict[str, str]] = {
    "cppo_optimized": {
        "module": "mava.systems.icrl.anakin.ff_ppo_crl",
        "config_name": "ppo_crl",
    },
    "cppo_baseline": {
        "module": "mava.systems.icrl.anakin.ff_ppo_crl_baseline",
        "config_name": "ppo_crl",
    },
    "icrl": {
        "module": "mava.systems.icrl.anakin.ff_icrl",
        "config_name": "ff_icrl",
    },
    "ippo": {
        "module": "mava.systems.ppo.anakin.ff_ippo",
        "config_name": "ff_ippo",
    },
}


# MavaLogger's terminal output renders training metrics on lines like:
#   ACTOR - Compile seconds: 81.889 | Steps per second: 200.076 | Wall clock seconds: 81.889 | ...
# We extract metrics only from the ACTOR line (training SPS), not from the
# EVALUATOR line (eval-loop SPS). The optimized CPPO additionally emits
# `Compile seconds` (only on iter 0) and `Wall clock seconds` (every iter);
# the legacy baseline emits only `Steps per second`.
_NUM = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
RE_ACTOR_LINE = re.compile(r"ACTOR\s*-")
RE_SPS = re.compile(rf"Steps per second\s*[:=]\s*{_NUM}")
RE_COMPILE = re.compile(rf"Compile seconds\s*[:=]\s*{_NUM}")
RE_WALL = re.compile(rf"Wall clock seconds\s*[:=]\s*{_NUM}")


def _poll_gpu_memory(stop_event: Event, peak_mb: list[int]) -> None:
    """Track peak nvidia-smi memory.used (sum across visible GPUs) at 0.5 Hz."""
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            used = sum(int(x.strip()) for x in out.strip().splitlines() if x.strip())
            peak_mb[0] = max(peak_mb[0], used)
        except Exception:
            pass
        time.sleep(0.5)


def run_system(
    name: str,
    cfg: dict[str, str],
    common_args: list[str],
    extra_args: list[str],
    *,
    repo_root: Path,
    log_dir: Path,
    mirror_lines_matching: re.Pattern[str] | None = None,
) -> dict[str, Any]:
    """Spawn one system, parse SPS/compile/wall metrics, track peak GPU mem."""
    cmd = [
        sys.executable,
        "-u",
        "-m",
        cfg["module"],
        "--config-name",
        cfg["config_name"],
        *common_args,
        *extra_args,
    ]
    log_path = log_dir / f"{name}.log"
    print(f"\n=== {name} ===")
    print("  cmd:", " ".join(cmd))
    print("  log:", log_path)

    peak_mem: list[int] = [0]
    stop = Event()
    poller = Thread(target=_poll_gpu_memory, args=(stop, peak_mem), daemon=True)
    poller.start()

    sps_values: list[float] = []
    wall_values: list[float] = []
    compile_seconds: float | None = None

    start = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(repo_root),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            # Only extract metrics from training (ACTOR) lines so the eval
            # loop's separate Steps-per-second value does not pollute the
            # training SPS series.
            if RE_ACTOR_LINE.search(line):
                m = RE_SPS.search(line)
                if m:
                    sps_values.append(float(m.group(1)))
                m = RE_COMPILE.search(line)
                if m:
                    compile_seconds = float(m.group(1))
                m = RE_WALL.search(line)
                if m:
                    wall_values.append(float(m.group(1)))
            if mirror_lines_matching and mirror_lines_matching.search(line):
                print("  ", line.rstrip())
        proc.wait()
    total_wall = time.time() - start

    stop.set()
    poller.join()

    # The first eval is JIT-compile-contaminated. Empirically the *second*
    # eval is also slow (additional XLA/PMAP warmup). Drop both unless we
    # only have ≤2 evals, in which case we keep what we have.
    if len(sps_values) >= 4:
        steady_sps = sps_values[2:]
    elif len(sps_values) >= 2:
        steady_sps = sps_values[1:]
    else:
        steady_sps = []
    return {
        "name": name,
        "module": cfg["module"],
        "exit_code": proc.returncode,
        "n_evals_logged": len(sps_values),
        "sps_all_evals": sps_values,
        "sps_first_eval": sps_values[0] if sps_values else None,
        "sps_steady_values": steady_sps,
        "sps_steady_mean": float(np.mean(steady_sps)) if steady_sps else None,
        "sps_steady_std": float(np.std(steady_sps)) if steady_sps else None,
        "compile_seconds": compile_seconds,
        "wall_clock_total_s": total_wall,
        "wall_clock_per_eval_s": wall_values,
        "peak_gpu_memory_mb": peak_mem[0],
        "peak_gpu_memory_gb": peak_mem[0] / 1024.0,
    }


def format_table(results: dict[str, dict[str, Any]]) -> str:
    rows = []
    headers = [
        "system",
        "exit",
        "compile (s)",
        "SPS (mean ± std)",
        "n_steady",
        "peak GPU (GB)",
        "total wall (s)",
    ]
    rows.append(headers)
    for name, r in results.items():
        sps_mean = r.get("sps_steady_mean")
        sps_std = r.get("sps_steady_std")
        sps_str = (
            f"{sps_mean:,.0f} ± {sps_std:,.0f}"
            if sps_mean is not None
            else "—"
        )
        compile_str = (
            f"{r['compile_seconds']:.1f}" if r.get("compile_seconds") is not None else "—"
        )
        rows.append(
            [
                name,
                str(r.get("exit_code")),
                compile_str,
                sps_str,
                str(len(r.get("sps_steady_values", []))),
                f"{r.get('peak_gpu_memory_gb', 0):.2f}",
                f"{r.get('wall_clock_total_s', 0):.1f}",
            ]
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    out = []
    for i, row in enumerate(rows):
        out.append(" | ".join(c.ljust(w) for c, w in zip(row, widths)))
        if i == 0:
            out.append("-+-".join("-" * w for w in widths))
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num_envs", type=int, default=512)
    p.add_argument("--num_updates", type=int, default=80)
    p.add_argument("--num_evaluation", type=int, default=8)
    p.add_argument("--rollout_length", type=int, default=128)
    p.add_argument("--task", type=str, default="smacv2_5_units")
    p.add_argument("--num_eval_episodes", type=int, default=64)
    p.add_argument(
        "--systems",
        nargs="+",
        default=["cppo_optimized", "cppo_baseline", "icrl", "ippo"],
        choices=list(SYSTEMS),
    )
    p.add_argument("--out", type=str, default="benchmark_results.json")
    p.add_argument("--log_dir", type=str, default="benchmark_logs")
    p.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Extra Hydra overrides applied to every system, e.g. system.seed=0",
    )
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    common_args = [
        "env=smax",
        f"env.scenario.task_name={args.task}",
        f"arch.num_envs={args.num_envs}",
        f"system.num_updates={args.num_updates}",
        f"arch.num_evaluation={args.num_evaluation}",
        f"system.rollout_length={args.rollout_length}",
        f"arch.num_eval_episodes={args.num_eval_episodes}",
        "logger.loggers.wandb.enabled=false",
        *args.extra,
    ]
    print("Common Hydra overrides:")
    for a in common_args:
        print(f"  {a}")

    mirror = re.compile(r"steps_per_second|compile_seconds|win_rate|Traceback|Error|FAILED|Killed")

    results: dict[str, dict[str, Any]] = {}
    for name in args.systems:
        results[name] = run_system(
            name,
            SYSTEMS[name],
            common_args,
            [],
            repo_root=repo_root,
            log_dir=log_dir,
            mirror_lines_matching=mirror,
        )

    print("\n" + format_table(results))

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nSaved JSON results to {args.out}")


if __name__ == "__main__":
    main()
