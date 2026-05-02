"""Summarize a long ff_ppo_crl run: SPS, win-rate trajectory, total wall."""
import re, sys

NUM = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
RE_SPS_ACTOR = re.compile(rf"ACTOR.*?Steps per second\s*[:=]\s*{NUM}")
RE_WIN_ACTOR = re.compile(rf"ACTOR.*?Win rate\s*[:=]\s*{NUM}")
RE_WIN_EVAL = re.compile(rf"EVALUATOR.*?Win rate\s*[:=]\s*{NUM}")
RE_COMPILE = re.compile(rf"Compile seconds\s*[:=]\s*{NUM}")
RE_WALL = re.compile(rf"ACTOR.*?Wall clock seconds\s*[:=]\s*{NUM}")

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cppo_sub25_80M.log"
sps, train_win, eval_win, walls = [], [], [], []
compile_s = None
with open(path) as f:
    for line in f:
        m = RE_SPS_ACTOR.search(line)
        if m: sps.append(float(m.group(1)))
        m = RE_WIN_ACTOR.search(line)
        if m: train_win.append(float(m.group(1)))
        m = RE_WIN_EVAL.search(line)
        if m: eval_win.append(float(m.group(1)))
        m = RE_COMPILE.search(line)
        if m: compile_s = float(m.group(1))
        m = RE_WALL.search(line)
        if m: walls.append(float(m.group(1)))

print(f"evals seen           : {len(sps)}")
print(f"compile seconds      : {compile_s}")
if len(sps) > 2:
    steady = sps[2:]
    print(f"steady SPS (mean)    : {sum(steady)/len(steady):,.0f}")
    print(f"steady SPS (min,max) : {min(steady):,.0f} / {max(steady):,.0f}")
print(f"total wall (sum)     : {sum(walls):.1f}s")
print()
print("training-rollout win rate (per eval, all):")
for i, w in enumerate(train_win):
    print(f"  eval {i:3d}: {w:.3f}")
print()
print("eval-policy win rate (per eval, non-zero only):")
for i, w in enumerate(eval_win):
    if w > 0:
        print(f"  eval {i:3d}: {w:.3f}")
print()
if eval_win:
    print(f"final eval win rate  : {eval_win[-1]:.3f}")
    print(f"max eval win rate    : {max(eval_win):.3f} at eval {eval_win.index(max(eval_win))}")
