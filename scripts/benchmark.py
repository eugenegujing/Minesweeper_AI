"""Benchmark an agent over many random boards.

Usage:
    python -m scripts.benchmark --agent csp --difficulty intermediate --episodes 1000
    python -m scripts.benchmark --agent csp --difficulty beginner --episodes 500 --output results.json
"""
from __future__ import annotations

import argparse
import json

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
    p.add_argument("--output", type=str, default=None,
                   help="Write full results to a JSON file")
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
    print(f"Mine hit rate:         {result['mine_hit_rate']:.6f}")
    print(f"Avg guesses/game:      {result['avg_guesses_per_game']:.2f}")
    avg_cr = result["avg_cells_revealed_before_loss"]
    if avg_cr is not None:
        print(f"Avg cells before loss: {avg_cr:.1f}")
    else:
        print("Avg cells before loss: N/A (no losses)")
    print(f"Avg runtime/move:      {result['avg_runtime_per_move_ms']:.3f} ms")
    print(f"Loss cause:            {result['loss_cause']}")

    if args.output:
        out = dict(result)
        out["config"] = {"height": h, "width": w, "mines": m,
                         "difficulty": args.difficulty}
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
