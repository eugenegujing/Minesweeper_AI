"""Evaluation harness: run an agent over many random boards and report win rate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from agents.base import Agent
from minesweeper.env import MinesweeperEnv


@dataclass
class EpisodeResult:
    won: bool
    steps: int
    cells_revealed: int


def run_episode(env: MinesweeperEnv, agent: Agent,
                seed: Optional[int] = None) -> EpisodeResult:
    obs, _ = env.reset(seed=seed)
    agent.reset()
    steps = 0
    while True:
        kind, r, c = agent.act(obs)
        action_idx = Agent.to_action_index((kind, r, c), env.w, env.h)
        obs, reward, terminated, truncated, info = env.step(action_idx)
        steps += 1
        if terminated or truncated:
            return EpisodeResult(
                won=bool(info.get("won", False)),
                steps=steps,
                cells_revealed=int(env.h * env.w - env.n_mines - info.get("n_remaining", 0)),
            )


def benchmark(agent_cls, height: int, width: int, n_mines: int,
              episodes: int = 1000, seed: int = 0,
              progress: bool = False) -> dict:
    env = MinesweeperEnv(height=height, width=width, n_mines=n_mines)
    agent = agent_cls(height=height, width=width, n_mines=n_mines,
                      rng=np.random.default_rng(seed))
    wins = 0
    total_steps = 0
    for i in range(episodes):
        ep_seed = seed * 1_000_003 + i  # deterministic per-episode seed
        result = run_episode(env, agent, seed=ep_seed)
        wins += int(result.won)
        total_steps += result.steps
        if progress and (i + 1) % max(1, episodes // 20) == 0:
            print(f"  [{i+1}/{episodes}] win rate so far: {wins / (i + 1):.3f}")
    return {
        "agent": agent.name,
        "episodes": episodes,
        "win_rate": wins / episodes,
        "wins": wins,
        "avg_steps": total_steps / episodes,
        "config": (height, width, n_mines),
    }
