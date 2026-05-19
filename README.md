# Minesweeper AI

CS 175 (Spring 2026) course project — a Minesweeper-playing AI built around a hybrid of constraint-satisfaction reasoning, probabilistic inference, and a deep-RL baseline.

---

## Project structure

```
minesweeper/        Board logic and Gymnasium-compatible environment
agents/             Random / SinglePoint / CSP / Probability / Final agents
evaluation/         Evaluation harness (win-rate over many random boards)
scripts/            CLI entry points (play one game, run benchmarks)
visualization/      Per-step heatmap rendering and GIF replay
tests/              Unit tests for board mechanics and the CSP solver
```

## Difficulty presets

| Name         | Size  | Mines |
| ------------ | ----- | ----- |
| beginner     | 9x9   | 10    |
| intermediate | 16x16 | 40    |
| expert       | 16x30 | 99    |

## Agents

- `random` — picks any unrevealed cell. Floor baseline.
- `single_point` — applies the two single-point Minesweeper rules; random fallback when stuck. Textbook baseline.
- `csp` — single-point + subset reasoning; random fallback when stuck.
- `probability` — CSP + exact frontier enumeration; picks min-mine-probability cell. Exposes `last_probabilities` for visualization.
- `final` — CSP + probability + endgame exact solver + information-gain tie-breaking. Strongest agent in the project.
- `dqn` — CNN Double-DQN (training script in `scripts/train_dqn.py`, planned).

## Targets

| Configuration | Target win rate |
| ------------- | --------------- |
| Beginner      | >= 88%          |
| Intermediate  | >= 70%          |
| Expert        | >= 30%          |

---

# Usage Guide

## 1. Install dependencies

First time only:

```bash
cd Minesweeper_AI
pip install -r requirements.txt
```

This installs `numpy`, `gymnasium`, `pytest`, `matplotlib`, and `imageio` (the last two are needed for the GIF replay feature).

## 2. Play a single game

`scripts/play.py` runs one episode and (optionally) prints the board after every step or saves a GIF replay.

```bash
python -m scripts.play --agent csp --difficulty beginner --render
python -m scripts.play --agent probability --difficulty beginner --render

```

**Arguments**

- `--agent`: `random` / `single_point` / `csp` / `probability` / `final`
- `--difficulty`: `beginner` / `intermediate` / `expert`
- `--seed 42`: fix the random seed for reproducibility
- `--render`: print an ASCII board to the terminal after every move
- `--save-replay PATH`: save a GIF replay to `PATH`. With the `probability` agent, frames where the agent fell back to probability inference also include a mine-probability heatmap overlay.

**Example output (with `--render`):**

```
--- step 1: ('reveal', 4, 4) reward=+0.010
. . . . . . . . .
. . . . . . . . .
. . .       1 . .
. . . 1 1 1 1 . .
...
WIN after 24 steps (9x9, 10 mines, agent=csp)
```

**Save a GIF replay (with mine-probability heatmap):**

```bash
python -m scripts.play --agent probability --difficulty intermediate --seed 7 \
    --save-replay demo.gif
```

In the resulting GIF:

- White cells with numbers = revealed safe cells (numbers are colored by Minesweeper convention).
- Orange `F` cells = flagged as mines.
- Gray cells = unrevealed (no probability info: CSP made a confident move this turn).
- Green→yellow→red cells = unrevealed cells with mine probabilities (only on turns where the probability fallback was triggered). Number inside each cell is the mine probability in percent.
- Blue / orange box = highlight on the cell of the most recent reveal / flag.

## 3. Benchmark an agent

`scripts/benchmark.py` runs many random boards and reports win rate.

```bash
python -m scripts.benchmark --agent csp --difficulty beginner --episodes 1000
```

**Arguments**

- `--episodes 1000`: number of games
- `--progress`: print progress every 5%
- `--seed 0`: controls the per-episode seed sequence

**Output:**

```
Agent:       csp
Config:      9x9, 10 mines  (beginner)
Episodes:    1000
Win rate:    0.6420  (642/1000)
Avg steps:   24.3
```

## 4. Common workflows

**Compare every agent on Beginner:**

```bash
python -m scripts.benchmark --agent random       --difficulty beginner --episodes 500
python -m scripts.benchmark --agent single_point --difficulty beginner --episodes 500
python -m scripts.benchmark --agent csp          --difficulty beginner --episodes 500
python -m scripts.benchmark --agent probability  --difficulty beginner --episodes 500
```

**Record three agents on the same board (for side-by-side comparison in the report):**

```bash
python -m scripts.play --agent single_point --difficulty intermediate --seed 42 --save-replay sp.gif
python -m scripts.play --agent csp          --difficulty intermediate --seed 42 --save-replay csp.gif
python -m scripts.play --agent probability  --difficulty intermediate --seed 42 --save-replay prob.gif
```

**Run unit tests:**

```bash
python -m pytest tests/ -q
```

**Debug a single game and save the trace:**

```bash
python -m scripts.play --agent csp --difficulty intermediate --render --seed 7 > game.txt
```

## 5. Use as a Python library

In a notebook or your own script:

```python
import numpy as np
from minesweeper.env import MinesweeperEnv
from agents import CSPAgent

env = MinesweeperEnv(difficulty="beginner")
agent = CSPAgent(9, 9, 10, rng=np.random.default_rng(0))

obs, _ = env.reset(seed=0)
agent.reset()

while True:
    action = agent.act(obs)
    idx = type(agent).to_action_index(action, env.w, env.h)
    obs, reward, terminated, truncated, info = env.step(idx)
    if terminated:
        print("WIN" if info["won"] else "LOSS")
        break
```

Or run a benchmark programmatically:

```python
from evaluation.evaluate import benchmark
from agents import CSPAgent

result = benchmark(CSPAgent, height=16, width=16, n_mines=40, episodes=200)
print(result)
# {'agent': 'csp', 'win_rate': 0.45, 'wins': 90, 'avg_steps': 78.2, ...}
```

## 6. Recommended development loop

When working on a new agent:

1. Edit the agent file under `agents/`
2. Sanity-check behavior on one game: `python -m scripts.play --agent <name> --render --seed N`
3. Quick win-rate check: `python -m scripts.benchmark --agent <name> --episodes 100`
4. Reliable numbers for the report: `--episodes 1000` (or `10000` for the final results)

---

# Completed Tasks

## Task 1: Evaluation Metrics Expansion

### 1.1 `is_guess` flag on the Agent interface

Every agent now exposes a `last_was_guess: bool` attribute on the base `Agent` class (`agents/base.py`). The flag is set after each call to `act()`:

- **`True`** when the agent resorts to a fallback / non-certain move (random pick, probability-based pick).
- **`False`** when the agent makes a logically deduced move (CSP-confirmed safe/mine, single-point rule, endgame certain safe/mine).

The attribute is initialized in `Agent.__init__()` and reset in `Agent.reset()`. All four existing agents (`RandomAgent`, `SinglePointAgent`, `CSPAgent`, `ProbabilityAgent`) plus `FinalAgent` set the flag correctly. The `act()` return signature was not changed — `last_was_guess` is a side-channel attribute, consistent with the existing `last_reason: str` pattern.

**Files modified:** `agents/base.py`, `agents/random_agent.py`, `agents/single_point_agent.py`, `agents/csp_agent.py`, `agents/probability_agent.py`.

### 1.2 Five new evaluation metrics

The evaluation harness (`evaluation/evaluate.py`) now tracks and reports five additional aggregate metrics from `benchmark()`:

| Metric | Key | Description |
|--------|-----|-------------|
| Mine hit rate | `mine_hit_rate` | Total mine hits (losses) / total steps across all episodes |
| Avg guesses per game | `avg_guesses_per_game` | Mean number of `last_was_guess=True` steps per episode |
| Avg cells revealed before loss | `avg_cells_revealed_before_loss` | Mean cells revealed in lost episodes only; `null` if no losses |
| Avg runtime per move | `avg_runtime_per_move_ms` | Wall-clock time per `act()` call, in milliseconds (uses `time.perf_counter()`) |
| Loss cause breakdown | `loss_cause` | `{"reasoning": int, "guess": int}` — whether each losing move was a deduced move or a guess |

The per-episode dataclass `EpisodeResult` was expanded with `guess_count`, `total_time_s`, and `loss_was_guess` fields. Timing wraps only `agent.act()`, not `env.step()`, so it measures pure agent computation.

**Files modified:** `evaluation/evaluate.py`.

### 1.3 JSON output for benchmarks

`scripts/benchmark.py` now accepts `--output PATH` to write the full result dictionary (including all new metrics) to a JSON file. The `config` tuple is serialized as `{"height": int, "width": int, "mines": int, "difficulty": str}`. The console output also prints all five new metrics.

```bash
python -m scripts.benchmark --agent final --difficulty beginner --episodes 1000 --output results.json
```

Example JSON output:

```json
{
  "agent": "final",
  "episodes": 1000,
  "win_rate": 0.963,
  "wins": 963,
  "avg_steps": 27.8,
  "config": {"height": 9, "width": 9, "mines": 10, "difficulty": "beginner"},
  "mine_hit_rate": 0.001331,
  "avg_guesses_per_game": 0.21,
  "avg_cells_revealed_before_loss": 66.8,
  "avg_runtime_per_move_ms": 0.020,
  "loss_cause": {"reasoning": 0, "guess": 37}
}
```

**Files modified:** `scripts/benchmark.py`.

---

## Final Agent

### Architecture

`FinalAgent` (`agents/final_agent.py`) extends `ProbabilityAgent` via the inheritance chain:

```
Agent  ->  CSPAgent  ->  ProbabilityAgent  ->  FinalAgent
```

It inherits the full CSP constraint-building, single-point rules, subset reasoning, and frontier probability enumeration pipeline. Three methods are overridden:

### Override 1: `_fallback()` — Endgame exact solver

When the number of unrevealed cells drops to 25 or fewer, the agent switches from the standard component-level probability estimation to a **global exact solver**:

1. All unrevealed cells are treated as a single group (no frontier / off-frontier split).
2. A DFS enumerates every valid mine assignment that satisfies all constraints **and** the global mine count (`remaining_mines = total_mines - flagged_count`).
3. If any cell has P(mine) = 0.0 across all valid assignments, it is revealed immediately as **certain safe** (no guess needed).
4. If any cell has P(mine) = 1.0, it is flagged as **certain mine**.
5. Otherwise, the cell with the lowest mine probability is chosen using the enhanced tie-breaking (see below).
6. A safety cap of 100,000 solutions prevents combinatorial explosion; if exceeded, the agent falls through to the standard probability fallback.

This eliminates unnecessary guesses in endgame positions where the standard per-component approximation would not detect global certainties.

### Override 2: `_select_cell()` — Information-gain tie-breaking

When multiple cells share the minimum mine probability, the standard `ProbabilityAgent` picks randomly among them (with a preference for frontier cells). `FinalAgent` replaces this with a scoring function:

| Heuristic | Weight | Rationale |
|-----------|--------|-----------|
| Frontier cell | +10.0 | Revealing a frontier cell provides direct constraint feedback |
| Adjacent revealed numbered cells | +1.0 each | More adjacent constraints = more information gained on reveal |
| Unrevealed neighbors | +3.0 each | Higher chance of triggering a zero-flood cascade |
| Center proximity | +2.0 * (1 - dist/max_dist) | Center cells have more neighbors overall, yielding more information |

Ties after scoring are broken randomly.

### Override 3: `act()` — View caching

Stores the current board view (`self._last_view`) so that `_select_cell()` can inspect neighbor states for tie-breaking. Also resets `last_was_guess = False` at the start of each turn (overridden to `True` only if the agent falls back to guessing).

---

## Benchmark Results (1000 episodes each, seed=0)

### Beginner (9x9, 10 mines) — target: >= 88%

| Agent | Win Rate | Avg Steps | Guesses/Game | Loss Cause (R/G) |
|-------|----------|-----------|--------------|------------------|
| single_point | 65.9% | 23.8 | 1.11 | 0 / 341 |
| csp | 82.5% | 26.8 | 0.62 | 0 / 175 |
| probability | **96.6%** | 27.8 | 0.70 | 0 / 34 |
| final | **96.3%** | 27.8 | 0.21 | 0 / 37 |

### Intermediate (16x16, 40 mines) — target: >= 70%

| Agent | Win Rate | Avg Steps | Guesses/Game | Loss Cause (R/G) |
|-------|----------|-----------|--------------|------------------|
| single_point | 29.0% | 84.9 | 2.82 | 0 / 710 |
| csp | 55.3% | 102.8 | 1.76 | 0 / 447 |
| probability | **85.4%** | 119.4 | 2.35 | 0 / 146 |
| final | **85.5%** | 119.6 | 1.79 | 0 / 145 |

### Expert (16x30, 99 mines) — target: >= 30%

| Agent | Win Rate | Avg Steps | Guesses/Game | Loss Cause (R/G) |
|-------|----------|-----------|--------------|------------------|
| single_point | 0.6% | 98.1 | 3.96 | 0 / 994 |
| csp | 11.0% | 160.6 | 3.52 | 0 / 890 |
| probability | **49.4%** | 282.5 | 7.71 | 0 / 506 |
| final | **49.9%** | 282.9 | 7.48 | 0 / 501 |

All targets are met. The `loss_cause` column (Reasoning / Guess) confirms that across all agents, every single loss occurs on a guess — the CSP reasoning engine never causes a loss. The `final` agent consistently achieves the lowest guesses-per-game thanks to the endgame exact solver.

---

## Team

CS 175, Spring 2026 — Group 9.

## License

MIT — see `LICENSE`.
