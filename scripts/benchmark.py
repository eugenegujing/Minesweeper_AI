"""Benchmark an agent over many random boards.

Usage:
    python -m scripts.benchmark --agent csp --difficulty intermediate --episodes 1000
"""
from __future__ import annotations

import argparse

from agents import AGENT_REGISTRY
from evaluation.evaluate import benchmark
from minesweeper.board import DIFFICULTIES


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=sorted(AGENT_REGISTRY), default="random")
    p.add_argument("--difficulty", choices=sorted(DIFFICULTIES), default="beginner")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress", action="store_true")
    args = p.parse_args()

    h, w, m = DIFFICULTIES[args.difficulty]
    result = benchmark(AGENT_REGISTRY[args.agent], h, w, m,
                       episodes=args.episodes, seed=args.seed,
                       progress=args.progress)
    print(f"Agent:       {result['agent']}")
    print(f"Config:      {h}x{w}, {m} mines  ({args.difficulty})")
    print(f"Episodes:    {result['episodes']}")
    print(f"Win rate:    {result['win_rate']:.4f}  ({result['wins']}/{result['episodes']})")
    print(f"Avg steps:   {result['avg_steps']:.1f}")


if __name__ == "__main__":
    main()
