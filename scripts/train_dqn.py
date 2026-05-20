"""Train the DQN agent on Minesweeper.

Usage:
    python -m scripts.train_dqn --difficulty beginner --steps 150000
    python -m scripts.train_dqn --difficulty beginner --steps 50000 --resume

Saves a checkpoint to checkpoints/dqn_<difficulty>.pt. The DQNAgent class
loads from the same path by default.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from collections import deque
from pathlib import Path
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from agents.dqn_agent import (
    QNetwork, encode_view, default_checkpoint_path, default_device, N_CHANNELS,
)
from minesweeper.board import CellState, DIFFICULTIES
from minesweeper.env import MinesweeperEnv


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]
# (state_enc, action_index, reward, next_state_enc, done, next_valid_mask)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque[Transition] = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.buf.append(args)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, len(self.buf), size=batch_size)
        batch = [self.buf[i] for i in idx]
        states = np.stack([b[0] for b in batch])
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.stack([b[3] for b in batch])
        dones = np.array([b[4] for b in batch], dtype=np.float32)
        next_masks = np.stack([b[5] for b in batch])
        return states, actions, rewards, next_states, dones, next_masks

    def __len__(self) -> int:
        return len(self.buf)


def valid_mask(view: np.ndarray) -> np.ndarray:
    """Boolean (H, W) mask: True where the cell is still unrevealed (legal to reveal)."""
    return (view == int(CellState.UNREVEALED))


def select_action(net: torch.nn.Module, state_enc: np.ndarray, mask: np.ndarray,
                  epsilon: float, device: torch.device,
                  rng: random.Random) -> int:
    """Epsilon-greedy over valid actions only. Returns a flat (r * W + c) index."""
    h, w = mask.shape
    valid_flat = np.flatnonzero(mask)
    if valid_flat.size == 0:
        return 0  # game effectively over; arbitrary
    if rng.random() < epsilon:
        return int(rng.choice(valid_flat.tolist()))
    with torch.no_grad():
        x = torch.from_numpy(state_enc).unsqueeze(0).to(device)
        q = net(x).squeeze(0).cpu().numpy().reshape(-1)
    q_masked = np.full_like(q, -np.inf)
    q_masked[valid_flat] = q[valid_flat]
    return int(np.argmax(q_masked))


def train(args: argparse.Namespace) -> None:
    h, w, n_mines = DIFFICULTIES[args.difficulty]
    device = default_device()
    print(f"Training DQN on {args.difficulty} ({h}x{w}, {n_mines} mines) on {device}")

    env = MinesweeperEnv(height=h, width=w, n_mines=n_mines)
    online = QNetwork(N_CHANNELS).to(device)
    target = QNetwork(N_CHANNELS).to(device)

    ckpt_path = Path(args.checkpoint) if args.checkpoint \
        else default_checkpoint_path(h, w, n_mines)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    start_step = 0
    best_win_rate = 0.0
    if args.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        online.load_state_dict(state["model"])
        start_step = int(state.get("step", 0))
        best_win_rate = float(state.get("best_win_rate", 0.0))
        print(f"Resumed from {ckpt_path} at step {start_step}, "
              f"best win rate {best_win_rate:.3f}")
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(online.parameters(), lr=args.lr)
    buf = ReplayBuffer(args.buffer_size)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    obs, _ = env.reset(seed=args.seed)
    state_enc = encode_view(obs)
    mask = valid_mask(obs)

    recent_wins: Deque[int] = deque(maxlen=200)
    recent_lens: Deque[int] = deque(maxlen=200)
    losses: Deque[float] = deque(maxlen=500)
    ep_len = 0
    t0 = time.perf_counter()

    for step in range(start_step, args.steps):
        # Linear epsilon decay over args.eps_decay_steps
        frac = min(1.0, step / max(1, args.eps_decay_steps))
        epsilon = args.eps_start + frac * (args.eps_end - args.eps_start)

        action_flat = select_action(online, state_enc, mask, epsilon, device, rng)
        next_obs, reward, terminated, truncated, info = env.step(action_flat)
        done = terminated or truncated
        next_enc = encode_view(next_obs)
        next_mask = valid_mask(next_obs)

        buf.push(state_enc, action_flat, float(reward), next_enc,
                 bool(done), next_mask)

        state_enc = next_enc
        mask = next_mask
        ep_len += 1

        if done:
            recent_wins.append(int(info.get("won", False)))
            recent_lens.append(ep_len)
            ep_len = 0
            obs, _ = env.reset(seed=args.seed + step)
            state_enc = encode_view(obs)
            mask = valid_mask(obs)

        if len(buf) >= args.warmup and (step + 1) % args.train_every == 0:
            s, a, r, sp, d, m_next = buf.sample(args.batch_size)
            s_t = torch.from_numpy(s).to(device)
            a_t = torch.from_numpy(a).to(device)
            r_t = torch.from_numpy(r).to(device)
            sp_t = torch.from_numpy(sp).to(device)
            d_t = torch.from_numpy(d).to(device)
            m_next_t = torch.from_numpy(m_next).to(device).view(args.batch_size, -1)

            q_all = online(s_t).view(args.batch_size, -1)
            q_pred = q_all.gather(1, a_t.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_next = target(sp_t).view(args.batch_size, -1)
                q_next = q_next.masked_fill(~m_next_t, float("-inf"))
                # If a row has no valid action (game-over state), max is -inf.
                # Replace with 0 — the done flag below already zeroes its contribution.
                row_has_valid = m_next_t.any(dim=1)
                q_next_max = torch.where(
                    row_has_valid, q_next.max(dim=1).values,
                    torch.zeros_like(d_t),
                )
                target_q = r_t + (1.0 - d_t) * args.gamma * q_next_max

            loss = F.smooth_l1_loss(q_pred, target_q)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.item()))

        if (step + 1) % args.target_update == 0:
            target.load_state_dict(online.state_dict())

        if (step + 1) % args.log_every == 0:
            wr = sum(recent_wins) / max(1, len(recent_wins))
            avg_len = sum(recent_lens) / max(1, len(recent_lens))
            avg_loss = sum(losses) / max(1, len(losses))
            elapsed = time.perf_counter() - t0
            sps = (step - start_step + 1) / max(elapsed, 1e-9)
            print(
                f"step {step+1:>7d}  eps {epsilon:.3f}  "
                f"win {wr:.3f}  avg_len {avg_len:.1f}  loss {avg_loss:.4f}  "
                f"{sps:.0f} steps/s",
                flush=True,
            )

            # Save "best" checkpoint by recent win rate (with warm-up)
            if len(recent_wins) >= recent_wins.maxlen and wr > best_win_rate:
                best_win_rate = wr
                torch.save(
                    {"model": online.state_dict(), "step": step + 1,
                     "best_win_rate": best_win_rate,
                     "config": {"h": h, "w": w, "n_mines": n_mines}},
                    ckpt_path,
                )
                print(f"  -> saved checkpoint (win rate {wr:.3f}) to {ckpt_path}")

    # Always save a final checkpoint regardless of best-win-rate gating.
    final_path = ckpt_path.with_name(ckpt_path.stem + "_final.pt")
    torch.save(
        {"model": online.state_dict(), "step": args.steps,
         "best_win_rate": best_win_rate,
         "config": {"h": h, "w": w, "n_mines": n_mines}},
        final_path,
    )
    print(f"Saved final checkpoint to {final_path}")
    if not ckpt_path.exists():
        # No improvement was ever recorded; copy final to the default path so the
        # agent class can load it.
        torch.save(
            {"model": online.state_dict(), "step": args.steps,
             "best_win_rate": best_win_rate,
             "config": {"h": h, "w": w, "n_mines": n_mines}},
            ckpt_path,
        )
        print(f"Also saved final to default path {ckpt_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--difficulty", choices=sorted(DIFFICULTIES), default="beginner")
    p.add_argument("--steps", type=int, default=150_000,
                   help="Total environment steps to train for")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--warmup", type=int, default=2_000)
    p.add_argument("--train-every", type=int, default=1)
    p.add_argument("--target-update", type=int, default=2_000)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=25_000)
    p.add_argument("--log-every", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Checkpoint path (default: checkpoints/dqn_<difficulty>.pt)")
    p.add_argument("--resume", action="store_true",
                   help="Load existing checkpoint and continue training")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
