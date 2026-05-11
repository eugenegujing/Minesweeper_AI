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
- `final` — placeholder for the project's final agent (not implemented yet).
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

## Team

CS 175, Spring 2026 — Group 9.

## License

MIT — see `LICENSE`.
