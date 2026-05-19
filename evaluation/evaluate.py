"""Evaluation harness: run an agent over many random boards and report win rate."""
from __future__ import annotations

import time
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
    guess_count: int
    total_time_s: float
    loss_was_guess: Optional[bool]


def run_episode(env: MinesweeperEnv, agent: Agent,
                seed: Optional[int] = None) -> EpisodeResult:
    obs, _ = env.reset(seed=seed)
    agent.reset()
    steps = 0
    guess_count = 0
    total_time = 0.0
    last_was_guess = False

    while True:
        t0 = time.perf_counter()
        kind, r, c = agent.act(obs)
        t1 = time.perf_counter()
        total_time += (t1 - t0)

        last_was_guess = getattr(agent, "last_was_guess", False)
        if last_was_guess:
            guess_count += 1

        action_idx = Agent.to_action_index((kind, r, c), env.w, env.h)
        obs, reward, terminated, truncated, info = env.step(action_idx)
        steps += 1
        if terminated or truncated:
            won = bool(info.get("won", False))
            lost = bool(info.get("lost", False))
            return EpisodeResult(
                won=won,
                steps=steps,
                cells_revealed=int(info.get("n_revealed_safe", 0)),
                guess_count=guess_count,
                total_time_s=total_time,
                loss_was_guess=last_was_guess if lost else None,
            )


def benchmark(agent_cls, height: int, width: int, n_mines: int,
              episodes: int = 1000, seed: int = 0,
              progress: bool = False) -> dict:
    env = MinesweeperEnv(height=height, width=width, n_mines=n_mines)
    agent = agent_cls(height=height, width=width, n_mines=n_mines,
                      rng=np.random.default_rng(seed))
    wins = 0
    total_steps = 0
    total_guesses = 0
    total_time = 0.0
    cells_revealed_on_loss: list[int] = []
    loss_cause = {"reasoning": 0, "guess": 0}

    for i in range(episodes):
        ep_seed = seed * 1_000_003 + i
        result = run_episode(env, agent, seed=ep_seed)
        wins += int(result.won)
        total_steps += result.steps
        total_guesses += result.guess_count
        total_time += result.total_time_s

        if not result.won:
            cells_revealed_on_loss.append(result.cells_revealed)
            if result.loss_was_guess:
                loss_cause["guess"] += 1
            else:
                loss_cause["reasoning"] += 1

        if progress and (i + 1) % max(1, episodes // 20) == 0:
            print(f"  [{i+1}/{episodes}] win rate so far: {wins / (i + 1):.3f}")

    return {
        "agent": agent.name,
        "episodes": episodes,
        "win_rate": wins / episodes,
        "wins": wins,
        "avg_steps": total_steps / episodes,
        "config": (height, width, n_mines),
        "mine_hit_rate": (episodes - wins) / total_steps if total_steps else 0.0,
        "avg_guesses_per_game": total_guesses / episodes,
        "avg_cells_revealed_before_loss": (
            sum(cells_revealed_on_loss) / len(cells_revealed_on_loss)
            if cells_revealed_on_loss else None
        ),
        "avg_runtime_per_move_ms": (
            total_time / total_steps * 1000 if total_steps else 0.0
        ),
        "loss_cause": loss_cause,
    }
