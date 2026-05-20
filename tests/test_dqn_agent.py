"""Tests for the DQN agent and its observation encoding.

These tests do not require a trained checkpoint. The one test that exercises
checkpoint loading constructs a tiny random-weight checkpoint in a temp dir.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.dqn_agent import (
    DQNAgent,
    N_CHANNELS,
    QNetwork,
    encode_view,
)
from minesweeper.board import CellState
from minesweeper.env import MinesweeperEnv


# ----------------------------------------------------------------------
# encode_view
# ----------------------------------------------------------------------
def test_encode_view_shape():
    view = np.full((9, 9), int(CellState.UNREVEALED), dtype=np.int8)
    enc = encode_view(view)
    assert enc.shape == (N_CHANNELS, 9, 9)
    assert enc.dtype == np.float32


def test_encode_view_is_one_hot():
    """Every cell should be active on exactly one channel."""
    rng = np.random.default_rng(0)
    # Mix of all possible values the agent can ever observe (no MINE: -3).
    valid_values = [int(CellState.FLAGGED), int(CellState.UNREVEALED),
                    0, 1, 2, 3, 4, 5, 6, 7, 8]
    view = rng.choice(valid_values, size=(9, 9)).astype(np.int8)
    enc = encode_view(view)
    sums = enc.sum(axis=0)
    assert np.allclose(sums, 1.0), "every cell must be one-hot across channels"


def test_encode_view_channel_mapping():
    """Specific values must land on specific channels (stable across training/inference)."""
    view = np.array([[int(CellState.UNREVEALED), int(CellState.FLAGGED)],
                     [0, 5]], dtype=np.int8)
    enc = encode_view(view)
    # Channel 0 == FLAGGED, channel 1 == UNREVEALED, channel 2 == 0, ..., channel 7 == 5.
    assert enc[0, 0, 1] == 1.0  # flagged
    assert enc[1, 0, 0] == 1.0  # unrevealed
    assert enc[2, 1, 0] == 1.0  # count 0
    assert enc[7, 1, 1] == 1.0  # count 5


# ----------------------------------------------------------------------
# QNetwork
# ----------------------------------------------------------------------
def test_qnetwork_forward_shape():
    net = QNetwork()
    x = torch.zeros(4, N_CHANNELS, 9, 9)
    out = net(x)
    assert out.shape == (4, 9, 9)


def test_qnetwork_works_on_intermediate_size():
    """Fully convolutional => should accept any (H, W)."""
    net = QNetwork()
    x = torch.zeros(1, N_CHANNELS, 16, 16)
    out = net(x)
    assert out.shape == (1, 16, 16)


# ----------------------------------------------------------------------
# DQNAgent — fallback when no checkpoint
# ----------------------------------------------------------------------
def test_agent_random_fallback_without_checkpoint(tmp_path):
    missing = tmp_path / "does_not_exist.pt"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agent = DQNAgent(height=9, width=9, n_mines=10,
                         checkpoint_path=missing,
                         rng=np.random.default_rng(0))
    assert agent._loaded is False

    env = MinesweeperEnv(height=9, width=9, n_mines=10)
    obs, _ = env.reset(seed=0)
    kind, r, c = agent.act(obs)
    assert kind == "reveal"
    assert 0 <= r < 9 and 0 <= c < 9
    assert agent.last_was_guess is True
    assert "random fallback" in agent.last_reason


def test_agent_warns_when_checkpoint_missing(tmp_path):
    missing = tmp_path / "missing.pt"
    with pytest.warns(UserWarning, match="checkpoint not found"):
        DQNAgent(height=9, width=9, n_mines=10, checkpoint_path=missing)


# ----------------------------------------------------------------------
# DQNAgent — with a (random-weight) checkpoint
# ----------------------------------------------------------------------
@pytest.fixture
def tiny_checkpoint(tmp_path) -> Path:
    """Write a random-weight QNetwork checkpoint to a temp path."""
    net = QNetwork()
    path = tmp_path / "tiny.pt"
    torch.save({"model": net.state_dict()}, path)
    return path


def test_agent_loads_checkpoint(tiny_checkpoint):
    agent = DQNAgent(height=9, width=9, n_mines=10,
                     checkpoint_path=tiny_checkpoint,
                     device=torch.device("cpu"))
    assert agent._loaded is True


def test_agent_action_is_always_legal(tiny_checkpoint):
    """Even with adversarial Q-values, the masked argmax must land on an unrevealed cell."""
    agent = DQNAgent(height=9, width=9, n_mines=10,
                     checkpoint_path=tiny_checkpoint,
                     device=torch.device("cpu"))
    env = MinesweeperEnv(height=9, width=9, n_mines=10)
    obs, _ = env.reset(seed=1)

    # Manually reveal a few cells via env to create a non-trivial mask.
    obs, *_ = env.step(0)        # reveal (0, 0)
    obs, *_ = env.step(40)       # reveal (4, 4) — center
    kind, r, c = agent.act(obs)
    assert kind == "reveal"
    assert obs[r, c] == int(CellState.UNREVEALED), \
        f"agent picked already-revealed cell ({r},{c}) with view value {obs[r, c]}"


def test_agent_plays_a_full_episode(tiny_checkpoint):
    """End-to-end: random-weight network should still play to terminal without crashing."""
    agent = DQNAgent(height=9, width=9, n_mines=10,
                     checkpoint_path=tiny_checkpoint,
                     device=torch.device("cpu"))
    env = MinesweeperEnv(height=9, width=9, n_mines=10)
    obs, _ = env.reset(seed=2)
    agent.reset()

    for _ in range(200):  # generous step cap
        kind, r, c = agent.act(obs)
        from agents.base import Agent
        idx = Agent.to_action_index((kind, r, c), env.w, env.h)
        obs, _, terminated, truncated, _ = env.step(idx)
        if terminated or truncated:
            break
    else:
        pytest.fail("agent did not terminate within step cap")


def test_agent_never_flags(tiny_checkpoint):
    """DQN is reveal-only by design — flags are never emitted."""
    agent = DQNAgent(height=9, width=9, n_mines=10,
                     checkpoint_path=tiny_checkpoint,
                     device=torch.device("cpu"))
    env = MinesweeperEnv(height=9, width=9, n_mines=10)
    obs, _ = env.reset(seed=3)
    agent.reset()

    from agents.base import Agent
    for _ in range(200):
        kind, r, c = agent.act(obs)
        assert kind == "reveal"
        idx = Agent.to_action_index((kind, r, c), env.w, env.h)
        obs, _, terminated, truncated, _ = env.step(idx)
        if terminated or truncated:
            break
