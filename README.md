# Minesweeper AI

CS 175 (Spring 2026) course project — a Minesweeper-playing AI built around a hybrid of constraint-satisfaction reasoning, probabilistic inference, and a deep-RL baseline.

## Project structure

```
minesweeper/        Board logic and Gymnasium-compatible environment
agents/             Random / CSP / Probability / Hybrid / DQN agents
evaluation/         Evaluation harness (win-rate over many random boards)
scripts/            CLI entry points (play one game, run benchmarks)
tests/              Unit tests for board mechanics and the CSP solver
```

## Quick start

```bash
pip install -r requirements.txt

# Play one game with the random baseline
python -m scripts.play --agent random --difficulty beginner --render

# Benchmark an agent over 1000 boards
python -m scripts.benchmark --agent csp --difficulty intermediate --episodes 1000
```

## Difficulty presets

| Name         | Size   | Mines |
|--------------|--------|-------|
| beginner     | 9x9    | 10    |
| intermediate | 16x16  | 40    |
| expert       | 16x30  | 99    |

## Agents

- `random` — picks any unrevealed cell. Floor baseline.
- `csp` — constraint-subset reasoning; random fallback when stuck.
- `probability` — exact frontier enumeration; picks min-mine-probability cell.
- `hybrid` — CSP first, probability when no certain move exists.
- `dqn` — CNN Double-DQN (training script in `scripts/train_dqn.py`).

## Targets

| Configuration | Target win rate |
|---|---|
| Beginner     | >= 88% |
| Intermediate | >= 70% |
| Expert       | >= 30% |

## Team

CS 175, Spring 2026 — Group _[fill in]_.
