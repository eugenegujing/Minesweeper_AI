"""Evaluate a trained DQN checkpoint across all three Minesweeper difficulties.

The Q-network is fully convolutional, so a checkpoint trained on one board size
can be evaluated on any size. By default this points all three difficulties at
the beginner checkpoint, which lets us measure zero-shot transfer.

Usage:
    python -m scripts.eval_dqn
    python -m scripts.eval_dqn --episodes 500 --checkpoint checkpoints/dqn_beginner.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from agents.dqn_agent import DQNAgent
from evaluation.evaluate import run_episode
from minesweeper.board import DIFFICULTIES
from minesweeper.env import MinesweeperEnv


def evaluate(checkpoint: Path, difficulty: str, episodes: int, seed: int) -> dict:
    h, w, n_mines = DIFFICULTIES[difficulty]
    env = MinesweeperEnv(height=h, width=w, n_mines=n_mines)
    agent = DQNAgent(
        height=h, width=w, n_mines=n_mines,
        rng=np.random.default_rng(seed),
        checkpoint_path=checkpoint,
    )
    wins = 0
    total_steps = 0
    total_revealed = 0
    for i in range(episodes):
        result = run_episode(env, agent, seed=seed * 1_000_003 + i)
        wins += int(result.won)
        total_steps += result.steps
        total_revealed += result.cells_revealed
    return {
        "difficulty": difficulty,
        "config": f"{h}x{w}, {n_mines} mines",
        "episodes": episodes,
        "win_rate": wins / episodes,
        "wins": wins,
        "avg_steps": total_steps / episodes,
        "avg_cells_revealed": total_revealed / episodes,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default="checkpoints/dqn_beginner.pt")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    print(f"Checkpoint: {ckpt}")
    print(f"Episodes per difficulty: {args.episodes}\n")

    print(f"{'difficulty':<14} {'config':<18} {'win rate':>10} {'avg steps':>12} "
          f"{'avg revealed':>14}")
    print("-" * 72)
    for diff in ("beginner", "intermediate", "expert"):
        r = evaluate(ckpt, diff, args.episodes, args.seed)
        print(f"{r['difficulty']:<14} {r['config']:<18} "
              f"{r['win_rate']:>10.4f} {r['avg_steps']:>12.1f} "
              f"{r['avg_cells_revealed']:>14.1f}")


if __name__ == "__main__":
    main()
