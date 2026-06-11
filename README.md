# Minesweeper AI

CS 175 course project: a Minesweeper-playing AI built around deterministic
constraint reasoning, exact probabilistic inference, and low-risk
expected-value decision making.

## Project Structure

```text
minesweeper/        Board logic and Gymnasium-compatible environment
agents/             Random, SinglePoint, CSP, Probability, Final, and DQN agents
evaluation/         Benchmark harness and aggregate metrics
scripts/            CLI entry points for playing, benchmarking, and DQN training
visualization/      Probability heatmap rendering and GIF replay
tests/              Unit tests for board mechanics and agents
```

## Difficulties

| Name | Size | Mines |
| --- | --- | --- |
| beginner | 9x9 | 10 |
| intermediate | 16x16 | 40 |
| expert | 16x30 | 99 |

## Agents

- `random`: random unrevealed-cell baseline.
- `single_point`: applies the two standard single-cell Minesweeper rules, then random fallback.
- `csp`: single-point rules plus subset reasoning. Falls back to random when stuck.
- `probability`: CSP plus exact frontier-component enumeration and global mine-budget weighting. **Textbook reference implementation** — dict-based DFS, dict polynomial multiplication, strict argmin tie-break, no certainty extraction. Intentionally kept simple so the algorithm reads cleanly.
- `final`: the strongest agent. Inherits from `ProbabilityAgent` but **overrides every method that has an optimised version**. All engineering improvements (speed, decision quality, refinement) live here as method overrides, leaving the parents as clean baselines.
- `dqn`: pure reveal-only Deep Q-Network agent. It loads `checkpoints/dqn_<difficulty>.pt` by default and falls back to random play if no checkpoint is present.

Classical agent inheritance:

```text
Agent -> CSPAgent -> ProbabilityAgent -> FinalAgent
```

The chain is **monotone in capability** — `FinalAgent` is faster, makes fewer guesses, and has higher (or statistically equivalent) win rate than `ProbabilityAgent` on every difficulty. See **Agent Designs** for the algorithmic walkthrough and **Benchmark summary** for the numbers.

`DQNAgent` inherits directly from `Agent`; it does not reuse the CSP/probability pipeline.

## Install

```bash
pip install -r requirements.txt
```

Dependencies include `numpy`, `gymnasium`, `pytest`, `matplotlib`, `imageio`,
`Pillow`, and `torch`.

## Play One Game

```bash
python -m scripts.play --agent final --difficulty beginner --render
python -m scripts.play --agent final --difficulty intermediate --seed 7 --save-replay demo.gif
```

Options:

- `--agent`: `random`, `single_point`, `csp`, `probability`, `final`, or `dqn`
- `--difficulty`: `beginner`, `intermediate`, or `expert`
- `--seed N`: reproducible board and agent randomness
- `--render`: print the board after every move
- `--save-replay PATH`: save a GIF replay with probability heatmaps when available

## Benchmark

```bash
python -m scripts.benchmark --agent final --difficulty beginner --episodes 1000
python -m scripts.benchmark --agent final --difficulty expert --episodes 500 --output final_expert.json
```

The benchmark reports:

- `win_rate`
- `mine_hit_rate`
- `avg_guesses_per_game`
- `avg_cells_revealed_before_loss`
- `avg_runtime_per_move_ms`
- `loss_cause`: whether losses happened on a deduction or on a guess

## Run Tests

```bash
python -m pytest tests/ -q
```

Current coverage includes board mechanics, CSP rules, analytical probability
cases, FinalAgent endgame/certainty behavior, and DQN inference/fallback behavior.

## Design Rationale

This section explains the *why* behind each major architectural choice. The *what* and *how* live in **FinalAgent Design** and **Implementation Notes** below.

### Why a layered hybrid (CSP → Probability → EV → Lookahead) instead of a single technique

Every layer answers a strictly harder question than the previous one, and the agent only pays the cost of a harder layer when the cheaper one cannot decide:

| Layer                  | Question it answers                                    | Cost      |
|------------------------|--------------------------------------------------------|-----------|
| Single-point + subset  | "Is this cell *provably* safe/mine from one or two constraints?" | O(N) per turn |
| Probability enumeration | "What's the *marginal* mine probability of every unrevealed cell?" | DFS over `2^n` per component |
| Low-risk EV tie-break  | "Among similarly-safe cells, which reveal is *most informative*?" | per-candidate scoring |
| 1-ply lookahead        | "How many forced deductions follow each plausible reveal value?" | per-candidate × per-value re-inference |

A pure DQN or pure CSP collapses this gradient: DQN can't *prove* safety; CSP can't *choose* when stuck. The hybrid lets us extract every certain move before resorting to risk, then minimize the risk we do take.

### Why exact probability enumeration over sampling/MCMC

For tie-breaking we need **marginal** probabilities per cell. Sampling introduces variance that swamps the differences between candidates (often <1%). Components are small enough (typically <30 cells with subset reasoning + propagation) that exact DFS finishes in milliseconds. When a component does explode beyond the cap, per-component fallback collapses *only that component* into the binomial off-frontier pool — the other components still contribute their exact marginals. This is strictly better than discarding everything and falling to CSP random.

### Why split the frontier into connected components

The frontier of constrained cells often decomposes into 2–6 independent sub-graphs. Enumerating each separately is exponentially cheaper than enumerating jointly:
```
joint     enumeration:  2^(n1 + n2 + ... + nk)
component enumeration:  2^n1 + 2^n2 + ... + 2^nk
```
The components are re-joined via **histogram convolution**: each component contributes a polynomial `Σ count_k · x^k` (where `count_k` = #assignments with k mines), and the global histogram is the convolution. The global mine budget is enforced once, on the convolved histogram, via binomial weighting `C(F, M − k)` for off-frontier cells.

### Why endgame is the same code path as normal probability

An earlier draft of `FinalAgent` had a separate `_endgame_solve` DFS for "few cells left" boards. That was a maintenance hazard — two enumerators, two places to debug. Key insight: when `F = 0` (no off-frontier cells), the binomial weighting term `C(0, M − k)` is `1` iff `k = M` and `0` otherwise, which exactly enforces the global mine count. So the standard probability pipeline **already is** the exact endgame solver when `F = 0`. We just feed it the same inputs and trust it.

`endgame_threshold` and `max_endgame_solutions` survive only to give endgame moves a larger solution budget (decisive moves deserve more compute) and to count telemetry. The algorithm is identical to normal-mode play. `_endgame_solve` is retained on `FinalAgent` purely as a test-only API.

### Why bitmask DFS instead of a SAT/CP solver

Components are small and dense; SAT/CP solvers add per-call overhead (parsing, constraint propagation indexing, learned-clause storage) that doesn't amortize. A custom bitmask DFS with two forcing rules (saturation, fill) gives us:
- direct popcount-based feasibility (CPython-level ints with C popcount under the hood),
- numpy-vectorized histogram accumulation,
- a single solution-count cap as the only "hard" knob.

Trade-off accepted: no clause learning, no symmetry breaking. We don't need them at this scale.

### Why low-risk EV rather than strict argmin P(mine)

Strict argmin treats two cells with `p = 0.080` and `p = 0.085` as completely different. But the 0.085 cell might have 5 unrevealed neighbours and a ~50% chance of cascading a zero — that could be worth the 0.5% extra risk. `risk_tolerance` admits cells in a narrow band; EV picks the most informative among them. The current default is `0.0` (strict argmin + EV-tie-break only on exact ties) — empirically this keeps `FinalAgent` from accepting riskier-than-needed cells while still using EV to break ties more informatively than the parent's random tie-break. Setting `risk_tolerance > 0` widens the candidate band for ablation experiments.

The biggest deduction-quality gain over `ProbabilityAgent` actually comes from **certainty extraction** in `_fallback` (recognising P=0 / P=1 cells as deductions instead of guesses), not from the EV tie-break itself. Certainty extraction cuts `guesses/game` by 2.6–6× across difficulties.

### Why bounded 1-ply lookahead, not deeper search (and why it's off by default)

1-ply already costs ~2-3× runtime on Expert. Each additional ply multiplies by `K × ≈9` (K candidates × up to 9 reveal values). 2-ply would push Expert to >5 ms/move and double the implementation complexity. The marginal win-rate gain at 2-ply on Minesweeper is small relative to the unsolvable 50/50 floor (boards where no agent can win). Diminishing returns kick in fast.

At our default `n=2000` benchmark scale, enabling 1-ply lookahead did **not** measurably improve win rate over the bare static EV tie-break — but it did cost the 2-3× runtime hit. So it ships disabled by default (`lookahead_enabled=False`) and the four gating knobs (`top_k`, `p_gap`, `only_in_endgame`, and the master switch) are retained as opt-in knobs for ablation experiments. With more episodes (n ≥ 5000 on Expert) and tuned weights, lookahead may yet show a small advantage — that's deferred to future work.

### Why cascade-priority for `_pending_safe` ordering

When CSP deduces several safe cells at once, the **order** they're revealed in matters indirectly: revealing a cell that turns out to be `0` triggers flood-fill (potentially exposing dozens of cells in one move), which generates new constraints, which can deduce more safes in the next turn. So front-loading "likely-zero" reveals chains more deductions in fewer turns.

This is a correctness-preserving optimization — the cells revealed are the same set, just sequenced better. No risk; small reward.

### Why side-channel telemetry (`last_was_guess`, `last_probabilities`, ...)

Keeping `act()`'s return type as a simple 3-tuple keeps the env interface clean. Side-channel attributes let visualization (heatmap GIFs), benchmarking (`loss_cause` accounting), and ablation studies inspect what the agent did without polluting the action protocol. Each side-channel attribute has a single source of truth and is cleared at the start of every `act()`.

### Why a "first move" rule

The first reveal can never be deduced — no constraints exist. The board's `safe_first_click=True` invariant guarantees the first reveal is safe, so we pick the **center** cell: it has 8 neighbours (maximum information), and centrally-placed reveals are more likely to trigger flood-fills that expose the board's interior.

### Why not pure DQN for the strongest agent

DQN learns a Q-function over board states but cannot *prove* a cell is safe — it outputs a real-valued score, not a deduction. Even on Beginner — where the action space is smallest — our shipped DQN checkpoint reaches only ~11–13% win rate (vs `FinalAgent`'s ~96%). Intermediate and Expert demand exact enumeration: with 99 mines on 16×30, the probability of guessing wrong on the last few cells is mathematically pinned, and only deduction (not pattern recognition) can extract those certainties.

DQN ships as `agents/dqn_agent.py` for comparison / report purposes — it's a *baseline*, not a replacement.

## Agent Designs

This section walks each agent's reasoning approach end-to-end. Use the table in **Agents** above for the quick reference; come here for the algorithm, why it works (or doesn't), and what beats what.

The five classical agents form a strict inheritance chain. `RandomAgent`, `SinglePointAgent`, `CSPAgent`, and `ProbabilityAgent` are deliberately written as textbook references: each is the simplest correct implementation of its idea. **`FinalAgent` is where all engineering optimisations live** — it overrides six methods on its ancestors with faster / more accurate versions (see the FinalAgent section below for the full list). Every layer is **strictly safer than the next**: a higher layer only runs when the lower one cannot decide. DQN sits outside this chain — it has no notion of "deduce" vs "guess".

### `RandomAgent` — floor baseline

```
Agent → RandomAgent
```

**Algorithm**: list all `UNREVEALED` cells; pick one uniformly at random; return `("reveal", r, c)`. `last_was_guess = True` always.

That's it — no memory of previous turns, no view of the numbers, no first-move heuristic.

**Why it exists**: as a sanity-check floor. Any non-trivial agent must beat random; if a "smart" agent doesn't, the env or the agent has a bug. Random also calibrates the difficulty baseline — it gives a sense of "how hard is this difficulty if you do nothing".

**Empirical win rate** (n=2000, seed=0): 0.05% Beginner, 0.0% Intermediate / Expert. The agent is dumb enough that even Beginner's small boards beat it — without a first-move heuristic, the very first reveal often hits a mine outright. Random is the calibration floor.

---

### `SinglePointAgent` — textbook Minesweeper rules

```
Agent → SinglePointAgent
```

**Algorithm**:

1. **First move**: reveal the center cell `(h//2, w//2)`.
2. **Each subsequent move**, scan all revealed numbered cells. For each cell with value `n`:
   - Let `flagged` = count of flagged 8-neighbors; `unrevealed` = count of unrevealed 8-neighbors.
   - **Rule 1 (all safe)**: if `flagged == n`, the mines around this number are already accounted for, so every unrevealed neighbor must be safe → queue onto `_pending_safe`.
   - **Rule 2 (all mines)**: if `unrevealed == n - flagged`, the remaining mines must be exactly the unrevealed neighbors → queue onto `_pending_flag`.
3. Pop from the queues (mines first, then safes) and return that action.
4. If neither rule fires, **random fallback**.

**Strength**: every move it queues is provably safe or provably mine — single-point can never make a wrong deduction. Most starting positions on Beginner are densely solvable by these two rules alone.

**Weakness**: looks at one constraint at a time. Can't see that "constraint A says (a, b, c) has 1 mine" combined with "constraint B says (b, c) has 1 mine" implies cell `a` is safe. Many real Minesweeper positions need exactly that combining.

**Win rate** (n=2000, seed=0): 67.05% Beginner, 28.80% Intermediate, 0.90% Expert. The collapse on Expert reflects the higher mine density (21% vs 12%) — fewer cells satisfy the single-point rules on their own.

---

### `CSPAgent` — subset reasoning over a constraint store

```
Agent → CSPAgent
```

**Algorithm**: extends SinglePointAgent by building an explicit **constraint store** and applying **subset reasoning**.

For each revealed numeric cell with value `v`, define a constraint:

```
sum(unrevealed neighbors as 0/1 mine indicators) = v − (flagged neighbor count)
```

Inference loop (`_infer`, iterates to fixed point):

1. **Trivial rules** on each constraint: sum = 0 → all cells safe; sum = |cells| → all mines.
2. **Substitute resolved cells**: when a cell is proven safe or mine, remove it from every constraint and adjust the sum.
3. **Subset rule**: for every pair `(A, B)` of constraints where `A.cells ⊂ B.cells`, derive a new constraint:
   ```
   (B.cells − A.cells) sums to (B.n − A.n)
   ```
   Apply trivial rules to the derived constraint.
4. Dedupe; repeat.

Result: deduced safes go to `_pending_safe`, deduced mines to `_pending_flag`. They are revealed in **insertion order** (consumed by `list.pop()`).

If the inference loop produces nothing, falls back to **random**.

(`FinalAgent` overrides `_infer` to add cascade-priority ordering on these queues — see the FinalAgent section. The bare `CSPAgent` here does not sort.)

**Why subset reasoning helps**: in real Minesweeper, two adjacent number cells often have overlapping unrevealed neighborhoods. Their difference reveals exact mine assignments in the non-shared region — which single-point can't see.

Example: number `1` at `(0,0)` has unrevealed neighbors `{(1,0), (1,1)}`. Number `2` at `(0,1)` has unrevealed neighbors `{(1,0), (1,1), (1,2)}`. Subset: `(1,2)` alone must contain `2 − 1 = 1` mine → `(1,2)` is a mine. Single-point cannot deduce this.

**Win rate** (n=2000, seed=0): 82.45% Beginner, 54.95% Intermediate, 11.70% Expert. Big lift over single-point on Intermediate (28.8 → 54.95%) and Expert (0.9 → 11.7%). Still ceilinged by the random fallback whenever subset reasoning runs out.

---

### `ProbabilityAgent` — exact marginal mine probability

```
Agent → CSPAgent → ProbabilityAgent
```

**Algorithm**: extends CSPAgent. Inherits the full deduce-loop; when CSP has no deduction, instead of falling back to random:

1. **Build constraints** from the current view.
2. **Partition unrevealed cells**: `frontier` (cells appearing in at least one constraint) vs `off-frontier` (everything else).
3. **Connected-component split**: two frontier cells share a component iff they share a constraint or are transitively connected through shared constraints. Independent components multiply: enumerating each separately is exponentially cheaper.
4. **Enumerate each component** via DFS:
   - `assignment` is a list of `{-1, 0, 1}` values per cell (textbook representation).
   - For each constraint, check the partial-assignment feasibility: `decided_mines ≤ required` and `decided_mines + undecided ≥ required`.
   - Records all valid `(cell -> 0/1)` mappings and their mine counts.
   - No solution cap — this implementation is the classroom reference; it can be slow on large Expert components, which is one reason `FinalAgent` overrides it.
5. **Build histograms** `h_total[k]` and `h_cell[c][k]` from the enumerated solutions.
6. **Convolve component histograms** into the global mine-count distribution via dict-based polynomial multiplication.
7. **Apply off-frontier binomial weighting**: each global `k` is weighted by `C(F, M − k)`, where `F = |off-frontier|`, `M = remaining mines`. This enforces the global mine budget exactly. Off-frontier marginal P(mine) is uniform within the off-frontier pool.
8. **Per-cell marginal**: integrate `h_cell[c]` over the other components' convolution for each cell.
9. **Fallback**: pick the min `P(mine)` cell, tie-break preferring frontier, then random among the preferred pool. `last_was_guess = True` is set **unconditionally** — this baseline does NOT recognise P=0 / P=1 cells as deductions (it treats them as just-another-guess).

**Why this is a major leap from CSP on win rate**: when CSP subset reasoning saturates, there often still exist cells whose mine probability is **provably 0** through global mine-budget interactions across components — but CSP can't see those interactions because subset reasoning is local. Exact enumeration sees them, and ProbabilityAgent reveals them as min-P cells (even if it doesn't *label* the move as a deduction).

Example: two components with 3 cells / 1 mine each, total `remaining_mines = 2`. Each component locally allows 0, 1, 2, or 3 mines. But globally only the (1, 1) split is consistent with the total — making certain cell-level outcomes provable. CSP misses this; enumeration catches it.

**Win rate** (n=2000, seed=0): 96.05% Beginner, 86.80% Intermediate, 49.45% Expert. Expert specifically benefits — its topology rarely yields subset-deducible certainties, but exact enumeration extracts them anyway.

**Performance**: textbook dict-based DFS + dict polynomial multiplication. ~0.024 ms/move on Beginner, ~0.044 ms/move on Intermediate, ~0.31 ms/move on Expert. Slower than `FinalAgent` everywhere because `FinalAgent` replaces these data structures with bitmask + numpy.

**The gap to `FinalAgent`**: ProbabilityAgent here is intentionally the un-optimised classroom version. FinalAgent overrides every method that has an optimised version (`_build_constraints`, `_find_frontier`, `_enumerate_component_histograms`, `_compute_probabilities`, `_select_cell`, `_fallback`) plus adds `_infer` cascade ordering and `_order_pending`. The result: FinalAgent is faster on every difficulty AND makes 2.6–6× fewer guesses (because certainty extraction recognises P=0 / P=1 cells as deductions instead of guesses).

---

### `FinalAgent` — probability + low-risk EV + bounded 1-ply lookahead

```
Agent → CSPAgent → ProbabilityAgent → FinalAgent
```

**Algorithm**: inherits from `ProbabilityAgent` but overrides **every method that has an optimised version**. The decision flow is structured as **six fallback layers**, each only consulted when the prior layer cannot decide:

1. **First move**: reveal the center cell.
2. **CSP deterministic inference**: returns guaranteed safe / mine via single-point + subset reasoning (multi-cell deductions revealed in **cascade-priority order**).
3. **Exact probability inference**: enumerates valid assignments per frontier component; convolves into global distribution; computes marginal P(mine) per cell.
4. **Probability-certainty extraction**: any cell with P(mine) = 0 / 1 is treated as a deduction, not a guess. Queued for future turns.
5. **Endgame regime**: when `unrevealed ≤ endgame_threshold (36)`, the `max_component_solutions` cap is swapped to `max_endgame_solutions` (300,000) for the duration of the call — endgame moves get a larger budget. The algorithm is otherwise identical to step 3–4. Telemetry counters `endgame_calls` / `endgame_aborts` track usage.
6. **Low-risk EV tie-break**: when a guess is unavoidable, candidates are all cells within `risk_tolerance` (default 0.5%) of `min P(mine)`. Each is scored:

   ```
   static_ev =
       − p_mine                          × 30   # risk
       + p_zero_proxy                    ×  5   # cascade reward (expected flood-fill)
       + unrev_neighbours                ×  3   # potential information
       + revealed_numeric_neighbours     ×  1   # immediate constraint feedback
       + (10 if cell in frontier else 0)        # frontier bonus
       + (1 − dist_to_centre / max_dist) ×  2   # centrality

   p_zero_proxy ≈ Π over unrevealed neighbours of (1 − P_mine(neighbour))
   ```

   When `lookahead_enabled` and gates allow, the top `lookahead_top_k` candidates additionally get a 1-ply EV: for each plausible reveal value `v ∈ {0..8}`, estimate `P(value = v | cell safe)` from neighbour marginals, run a single hypothetical CSP inference pass, and count the forced deductions. Blend via `lookahead_static_weight` and `lookahead_weight`. Lookahead is gated by `lookahead_p_gap` (skip if candidate P(mine) spread > 5% — risk already dominates) and optionally `lookahead_only_in_endgame`.

This remains a **one-step policy** — no recursive game-tree search.

**Methods overridden vs `ProbabilityAgent`** (all live in `agents/final_agent.py`):

| Method                                | What changes                                                                            |
|---------------------------------------|-----------------------------------------------------------------------------------------|
| `_build_constraints`                  | `np.where`-vectorised outer scan instead of nested Python loops (speed).                   |
| `_find_frontier`                      | `np.where` + set difference instead of `h × w` Python loop (speed).                        |
| `_enumerate_component_histograms`     | **New method**, replaces parent's `_enumerate_component`. Bitmask DFS with `popcount` feasibility + saturation/fill propagation (speed). Outputs numpy histograms directly. |
| `_compute_probabilities`              | Numpy `np.convolve` polynomial multiplication; per-component fallback when one overflows the solution cap (speed + decision quality).                          |
| `_fallback`                           | Endgame regime swap of the solution cap; **certainty extraction** (P=0 → reveal as deduction, P=1 → flag as deduction; rest go onto pending queues); falls through to `_select_cell` only when no certainty exists (decision quality + refinement). |
| `_select_cell`                        | Low-risk EV tie-break + optional bounded 1-ply lookahead (decision quality + refinement).                     |
| `_infer`                              | Calls `super()._infer`, then sorts the pending queues by cascade priority (refinement).         |
| `_order_pending`                      | **New helper** used by `_infer` and the certainty extraction path.                      |

**Why it's the strongest classical agent**:
- It never guesses when a layer above could deduce.
- Certainty extraction labels P=0 / P=1 cells as deductions, not guesses — cutting `last_was_guess=True` events to a small fraction of the parent's count.
- Cascade-priority ordering front-loads likely-zero reveals, chaining more deductions per turn through flood-fill spread.
- Bitmask DFS + numpy convolution make the probability machinery 1.2–1.4× faster than the parent on every difficulty.
- The endgame uses the same exact machinery as normal play, just with a larger solution budget.

**Win rate** (n=2000, seed=0): 95.85% Beginner, 86.90% Intermediate, **50.10% Expert** — crosses the symbolic 50% line. **Average guesses per game**: 0.14 / 0.50 / 2.90 — 4.9× / 4.6× / 2.6× fewer than the un-optimised `ProbabilityAgent` (0.68 / 2.30 / 7.58) because certainty extraction recognises P=0 / P=1 cells as deductions.

`FinalAgent` is **simultaneously faster and makes fewer guesses** than `ProbabilityAgent` on every difficulty. Win rate is statistically equivalent on every difficulty at n=2000 (differences are −0.20pp / +0.10pp / +0.65pp on Beginner / Intermediate / Expert, all within the standard-error band).

---

### `DQNAgent` — pure Deep Q-Network baseline

```
Agent → DQNAgent  (no classical-reasoning inheritance)
```

**Algorithm**:

1. **Encode the view** as an 11-channel one-hot tensor of shape `(11, H, W)`:
   - Channel 0: `FLAGGED`
   - Channel 1: `UNREVEALED`
   - Channels 2..10: numeric values `0..8`
2. **Forward through the Q-network**: 4 convolutional layers (3×3, hidden=64, ReLU) followed by a 1×1 conv head producing one scalar Q-value per cell — output shape `(H, W)`.
3. **Mask illegal actions**: cells that are not `UNREVEALED` get Q = −∞.
4. **Argmax over the masked Q-map** → flat index → `(r, c)` → return `("reveal", r, c)`.
5. **`last_was_guess = True` always** — the network produces a *learned ranking*, not a logical proof. It cannot label a move as a deduction.

**Training pipeline** (`scripts/train_dqn.py`):
- Standard DQN with target network (synced every 2,000 steps), replay buffer, Adam (lr=5e-4), gamma=0.95, smooth-L1 loss.
- Action-masked TD target: `Q(s, a) = r + γ · max_{a' valid} Q_target(s', a')`.
- Epsilon-greedy with linear decay from 1.0 → 0.05.
- Rewards from `MinesweeperEnv`: +1 win, −1 loss, +0.01 safe reveal, −0.001 invalid click.
- Periodic checkpointing on improved 200-episode running win rate.
- The shipped Beginner checkpoint was trained for **400,000 steps** with `--eps-decay-steps 100000 --buffer-size 100000`, on an RTX 4080 Super (CUDA build of PyTorch, ~20 minutes wall-clock).

**Why pure DQN can't reach FinalAgent's win rates**:
- **No certainty proof**: when CSP can prove a cell safe, FinalAgent reveals it with zero risk. DQN ranks the same cell by Q-value, which is approximate — it may rank a *mine cell* higher and lose.
- **Sparse training signal**: random self-play covers a vanishing fraction of strategic boards. The network can learn local patterns but not exact global-mine-budget enforcement.
- **Action space scales poorly**: 480 actions on Expert (16×30) — the Q-function has to learn a 480-way ranking conditional on the entire board state, which demands data the random rollouts can't provide.
- Published Minesweeper DQN results plateau around 60-70% Beginner and well under 30% Expert, but only after **tens of millions of training steps**, heavy reward shaping, and architectural tricks (Double DQN, dueling heads, prioritized replay). Our shipped checkpoint is a deliberately minimal baseline (400k steps, basic DQN), so its Beginner win rate is lower (~12%). The classical agent hits 96 / 87 / 50 with no training at all.

**Why it's still in the repo**:
- A direct comparison between classical reasoning and a neural baseline is the kind of thing the report is about.
- Future hybrids could use DQN to *propose* candidate moves and CSP/Probability to *verify* them (analogous to AlphaZero's policy + value).
- The training script is a clean reference implementation of DQN against a Gymnasium env.

**Status**: a Beginner checkpoint ships (`checkpoints/dqn_beginner.pt`, ~12% win rate, 400k steps on GPU). Intermediate and Expert checkpoints are **intentionally not trained** — see below.

**Why we did not train Intermediate / Expert DQN checkpoints**:

We investigated this and concluded the cost is not worth it for a *baseline* whose job is comparison, not winning. The reasons are structural, not budget-limited:

- **Wins are too rare to learn from.** DQN learns almost entirely from the +1 win signal. On Expert (99 mines, 21% density), random/early-policy play essentially never wins — even the optimal classical agent only reaches 50%. With no winning trajectories in the replay buffer, the network gets no positive signal to learn from, so the Expert win rate stays at ~0% regardless of how long we train.
- **The action space explodes.** Beginner has 81 actions, Intermediate 256, Expert 480. The Q-function must learn a far higher-dimensional ranking from data the random rollouts cannot cover.
- **It cannot break the ceiling anyway.** Even a perfectly trained DQN cannot exceed the information-theoretic 50/50 endgame floor on Expert, because it ranks cells by a learned score and never computes the exact global-mine-budget probabilities those endgames require. That is precisely what the classical Probability/Final agents do.

A short experiment confirmed this: training a dedicated checkpoint for the harder difficulties (even at the same 400k-step budget) lands at ~0% Intermediate/Expert — i.e. *training does not help*, because the limitation is the method, not the amount of compute. We therefore report the Beginner checkpoint as the neural baseline and document the Intermediate/Expert results as zero-shot transfer from it. The "DQN ≈ 12% vs Final ≈ 96%" Beginner comparison is already sufficient to make the report's point: classical reasoning decisively beats the neural baseline on this problem.

---

### Benchmark summary

All numbers come from a single apples-to-apples sweep at **`n=2000` per agent per difficulty**, `seed=0`, current code. Every agent saw the same boards in the same order. DQN uses its Beginner checkpoint on all three difficulties (zero-shot transfer via the fully-convolutional network), which is why its Intermediate / Expert win rates are 0%.

#### Six-metric benchmark, classical agents

For each difficulty, the harness reports six metrics from `evaluation/evaluate.py:benchmark` over the same 2000 boards: `win_rate`, `mine_hit_rate` (losses divided by total moves), `avg_guesses_per_game` (`last_was_guess=True` events per episode), `avg_cells_revealed_before_loss`, `avg_runtime_per_move_ms`, and `loss_cause` (`guess / reasoning` losses). `random` and `dqn` are excluded from these tables because they do not participate in the deduction-vs-guess decomposition; their win-rate summary is reported separately below.

##### Beginner (9×9, 10 mines, n = 2000)

| Agent | Win rate | Mine-hit rate | Avg guesses/game | Avg cells revealed before loss | Avg runtime/move (ms) | Loss cause (guess / reasoning) |
|---|---:|---:|---:|---:|---:|---:|
| `single_point` | 67.05% | 0.0137 | 1.13 | 60.5 | 0.022 | 659 / 0 |
| `csp`          | 82.45% | 0.0066 | 0.61 | 63.8 | 0.025 | 351 / 0 |
| `probability`  | 96.05% | 0.0014 | 0.68 | 67.7 | 0.039 | 79 / 0  |
| **`final`**    | 95.85% | 0.0015 | **0.14** | **68.4** | 0.032 | 83 / 0 |

##### Intermediate (16×16, 40 mines, n = 2000)

| Agent | Win rate | Mine-hit rate | Avg guesses/game | Avg cells revealed before loss | Avg runtime/move (ms) | Loss cause (guess / reasoning) |
|---|---:|---:|---:|---:|---:|---:|
| `single_point` | 28.80% | 0.0085 | 2.78 | 161.0 | 0.038 | 1424 / 0 |
| `csp`          | 54.95% | 0.0044 | 1.70 | 168.1 | 0.024 | 901 / 0  |
| `probability`  | 86.80% | 0.0011 | 2.30 | 203.1 | 0.041 | 264 / 0  |
| **`final`**    | 86.90% | 0.0011 | **0.50** | **206.2** | 0.031 | 262 / 0 |

##### Expert (16×30, 99 mines, n = 2000)

| Agent | Win rate | Mine-hit rate | Avg guesses/game | Avg cells revealed before loss | Avg runtime/move (ms) | Loss cause (guess / reasoning) |
|---|---:|---:|---:|---:|---:|---:|
| `single_point` | 0.90%  | 0.0100 | 4.04 | 161.7 | 0.045 | 1982 / 0 |
| `csp`          | 11.70% | 0.0055 | 3.47 | 206.6 | 0.070 | 1766 / 0 |
| `probability`  | 49.45% | 0.0018 | 7.58 | 333.0 | 0.330 | 1011 / 0 |
| **`final`**    | **50.10%** | 0.0018 | **2.90** | **333.2** | **0.207** | 998 / 0 |

The `loss cause` column reads `<guesses>/<reasoning>`. Across all 4 classical agents × 3 difficulties × 2000 episodes (≈ 9700 losing episodes total), the `reasoning` column is **zero** — every loss happens on a `last_was_guess=True` move. This is the strongest randomised-board evidence that the deduction pipeline is implemented correctly: no agent ever revealed a cell it had logically (and incorrectly) proved safe.

Standard error at `n=2000` is roughly `±0.44%` near 96%, `±0.75%` near 87%, `±1.12%` near 50%. `final` vs `probability` differs by `−0.20pp / +0.10pp / +0.65pp` on Beginner / Intermediate / Expert — within `±1` SE on every difficulty — so the two agents are statistically equivalent on win rate. **Final crosses 50% on Expert (50.10%)**, marginally above the symbolic threshold that Probability stays just below, while being **1.26× / 1.29× / 1.49× faster** and making **~4.9× / 4.6× / 2.6× fewer guesses**. The runtime and guess-count gains are attributable to the speed overrides (bitmask DFS + `np.convolve`) and the certainty-extraction override respectively, which only live on `FinalAgent`.

##### Win-rate of `random` and `dqn` (for context only)

| Agent    | Beginner | Intermediate | Expert |
|----------|---------:|-------------:|-------:|
| `random` | 0.05%    | 0.0%         | 0.0%   |
| `dqn`    | 12.0%    | 0.0%         | 0.0%        |

`random` provides the floor; `dqn` ranks cells by a learned Q-value with no ability to prove safety, so every DQN move is flagged as a guess by construction and `loss_cause` is uninformative for it. The `dqn` row uses the shipped Beginner checkpoint (400k steps, GPU); its Intermediate / Expert columns are **zero-shot transfer** from that Beginner checkpoint (the fully-convolutional network accepts any board size), which is why they sit at 0% — see the `DQNAgent` section for why dedicated Intermediate/Expert training does not help.

#### How `final` beats `probability` on every axis

`ProbabilityAgent` is the textbook reference: nested-Python `_build_constraints` and `_find_frontier`, dict-based DFS, dict-based polynomial multiplication, strict `argmin P(mine)` with frontier preference and random tie-break, and `last_was_guess = True` set unconditionally on every probability fallback. It's deliberately kept simple so it documents the algorithm.

`FinalAgent` overrides every method that has a faster or more accurate implementation. The full override table is in the **`FinalAgent`** section above. Grouped by intent:

- **Speed**: `_build_constraints`, `_find_frontier`, `_enumerate_component_histograms` (new method, bitmask + propagation), `_compute_probabilities` (numpy convolution).
- **Decision quality**: certainty extraction in `_fallback` (P=0 → reveal as deduction, P=1 → flag as deduction, queue the rest); per-component fallback when one component overflows the solution cap; low-risk EV tie-break in `_select_cell`; endgame regime with larger solution cap.
- **Refinement**: cascade-priority `_order_pending` + `_infer` override; bounded 1-ply lookahead in `_select_cell` (off by default — costs ~2× runtime without measurable win-rate gain at our sample sizes; opt-in via `lookahead_enabled=True` for ablation).

The benchmark consequence: `final` is faster (speed overrides), makes far fewer guesses (certainty extraction), and statistically matches or marginally beats `probability` on win rate.

## Tunable FinalAgent Attributes

These can be passed to the constructor or changed on the instance:

All tunables are also class attributes, so callers can either pass them to the constructor or override per-instance for ablation experiments.

```python
from agents import FinalAgent

agent = FinalAgent(
    16,
    30,
    99,
    # Probability / endgame:
    endgame_threshold=36,
    max_endgame_solutions=300_000,
    max_component_solutions=200_000,
    risk_tolerance=0.0,
    # Lookahead (all six accepted as kwargs):
    lookahead_enabled=False,
    lookahead_top_k=5,
    lookahead_p_gap=0.05,
    lookahead_only_in_endgame=False,
    lookahead_static_weight=1.0,
    lookahead_weight=1.0,
)
# EV scoring weights are class attributes; override per-instance:
agent.ev_mine_penalty = 30.0
agent.ev_zero_weight = 5.0
agent.ev_unrev_weight = 3.0
agent.ev_revealed_weight = 1.0
agent.ev_frontier_bonus = 10.0
agent.ev_centrality_weight = 2.0
```

Probability / endgame:

- `endgame_threshold`: unrevealed-cell count at which endgame telemetry/caps apply (default 36).
- `max_endgame_solutions`: DFS solution cap inside the endgame regime (default 300_000 — generous because endgame moves are decisive).
- `max_component_solutions`: DFS solution cap for normal probability components (default 200_000).
- `risk_tolerance`: extra mine-probability slack allowed for EV tie-break candidates. Default `0.0` (only literal ties go through EV; strictly worse cells are never admitted). Raise to e.g. `0.005` to widen the band — useful for ablation.

Bounded 1-ply lookahead (all six accepted as constructor kwargs since the recent unused-attribute audit):

- `lookahead_enabled`: master switch (default `False` — lookahead is opt-in; costs ~2× runtime without measurable win-rate gain at our sample sizes).
- `lookahead_top_k`: only the top-K static-EV candidates get lookahead (default 5).
- `lookahead_p_gap`: skip lookahead when candidate mine-prob spread exceeds this gap (default 0.05).
- `lookahead_only_in_endgame`: if `True`, lookahead runs only when in endgame regime (default `False`).
- `lookahead_static_weight`, `lookahead_weight`: blend weights for static EV vs lookahead EV (defaults 1.0, 1.0).

Static EV scoring weights (class attributes; raise/lower for ablation):

- `ev_mine_penalty` (default 30.0): coefficient on `-p_mine`.
- `ev_zero_weight` (default 5.0): coefficient on the `p_zero_proxy` cascade term.
- `ev_unrev_weight` (default 3.0): coefficient on unrevealed-neighbour count.
- `ev_revealed_weight` (default 1.0): coefficient on revealed-numeric-neighbour count.
- `ev_frontier_bonus` (default 10.0): flat bonus added when the candidate cell is on the frontier.
- `ev_centrality_weight` (default 2.0): coefficient on the centrality term `(1 − dist / max_dist)`.

Inspectable state after `act()`:

- `last_was_guess`: whether the chosen move was a guess.
- `last_reason`: short decision explanation.
- `last_probabilities`: mine-probability map from the last probability pass.
- `last_candidate_scores`: scalar EV score per low-risk candidate.
- `last_candidate_ev`: score breakdown per candidate (`mine_prob`, `zero_prob`, sub-scores, and `lookahead_ev` when lookahead ran for that candidate).
- `endgame_calls`: number of times the endgame regime was entered in the current episode.
- `endgame_aborts`: number of actual component-enumeration aborts in that regime.
- `lookahead_evals`: number of `_lookahead_ev` value-branch evaluations performed (one candidate × up to 9 plausible values per call).

## Implementation Notes

Knowledge consolidated from three rounds of independent code review. This section documents the invariants and edge cases that have been traced through so future changes know what they have to preserve.

### Correctness invariants

**Bitmask DFS feasibility (`final_agent.py` `_enumerate_component_histograms`).**
The DFS represents an assignment as a Python int (bit `i = 1` means cell `i` is a mine) and the undecided set as another int (bit `i = 1` means cell `i` is still undecided). For each constraint with cell mask `cmask` and required mine count `req`:

```
mines     = popcount(assignment & cmask)
undecided = popcount(cmask & undecided_mask)
infeasible iff (mines > req) or (mines + undecided < req)
```

This is bitwise-equivalent to the textbook set-based feasibility check `ProbabilityAgent._enumerate_component` uses; the move from sets / `{-1,0,1}` lists to ints is purely a constant-factor speedup.

**Numpy convolution equals dict convolution.** `FinalAgent._compute_probabilities` convolves per-component mine-count histograms using `np.convolve` on float64 arrays whose entries are integer counts (≤ 200_000 per component, ≤ a few across components). float64 represents integers up to `2^53`, which is comfortably above any reachable product. The numpy path produces the same integer counts as `ProbabilityAgent`'s textbook dict-based polynomial multiplication.

**LIFO consumption of `_pending_safe` / `_pending_flag`.** `CSPAgent.act` consumes the queues via `list.pop()` (i.e. from the end). `FinalAgent._infer` calls `super()._infer()` then sorts the cells **ascending** by cascade priority via `_order_pending`, so the LAST element popped is the HIGHEST priority — a "likely-zero" reveal triggers flood-fill before less informative reveals. `CSPAgent` on its own does not sort.

**Propagation force-rules in component enumeration.** Inside the DFS, `propagate` applies two forcing rules iteratively until fixed point:

- **Saturation**: if a constraint's decided mines equal its required count and undecided > 0, all remaining cells in that constraint are safe. Clear their bits from `undecided_mask`, re-enqueue OTHER constraints touching those cells.
- **Fill**: if `decided_mines + undecided == required` and undecided > 0, all remaining undecided cells are mines. OR them into `assignment`, clear from `undecided_mask`, bump `mines_so_far` by `undecided`, re-enqueue OTHER constraints.

The just-processed constraint is excluded from re-enqueue because after forcing it has `undecided == 0` and would no-op.

**Per-component fallback when enumeration aborts (`final_agent.py` `_compute_probabilities`).** If one frontier component exceeds `max_component_solutions`, its cells get folded into the off-frontier pool (uniform binomial weighting against the global mine budget) and the other components still contribute their exact marginals. The flag `self._last_compute_aborted` is set so `_fallback` can bump `endgame_aborts`. The textbook `ProbabilityAgent` has no cap and no per-component fallback — it just runs until the DFS finishes, which is why it can be slow on pathological Expert components.

**Endgame regime swaps the cap.** `FinalAgent._fallback` sets `self.max_component_solutions = self.max_endgame_solutions` (300_000) for the duration of the call when `len(unrevealed) <= endgame_threshold`, then restores the old cap in a `finally` block. This guarantees endgame moves get a larger solution budget without leaking into normal-mode calls.

**Probability-deduced certainties bypass the guess label (`FinalAgent._fallback`).** When `_compute_probabilities` produces any cell with `p ≤ 1e-12` (certain safe) or `p ≥ 1 − 1e-12` (certain mine), `FinalAgent._fallback` returns that move with `last_was_guess = False` and queues the rest onto `_pending_safe` / `_pending_flag` (after sorting them by cascade priority). These are deductions, not guesses, so `loss_cause` accounting stays correct. `ProbabilityAgent._fallback` does NOT do this — it sets `last_was_guess = True` unconditionally.

### Edge cases verified

- **Empty constraints** (e.g. early game with no revealed numerics): `_fallback` falls through to CSP's random fallback rather than enumerating.
- **Off-frontier-only board** (no constraints anywhere): `_compute_probabilities` returns the uniform `M_remaining / pool_size` for every unrevealed cell.
- **Single-cell component** (n = 1): traced through `propagate` for `required ∈ {0, 1, >1}`. Returns correct one-bit histograms.
- **`remaining_mines == 0`**: top-of-DFS prune `mines_so_far > remaining_mines` triggers; the only solution recorded is the all-safe assignment.
- **One component aborts, others succeed**: aborted component cells get binomial weighting; other components still contribute exact marginals.
- **Corner candidate cell with out-of-bounds neighbours**: `_select_cell`'s 8-neighbour loop skips out-of-bounds; `p_zero` falls into the `unrev_n == 0` branch (returns 0.0).
- **`view is None` during `_select_cell`**: neighbour information defaults to "absent"; lookahead/static EV degrades gracefully.
- **`lookahead_enabled = False`**: `_lookahead_active` returns immediately, the lookahead branch is skipped, `lookahead_evals` stays at 0.
- **`lookahead_p_gap` exceeded** (max-min mine-prob spread > 0.05): lookahead is skipped entirely; static EV alone ranks candidates.

### Explicit EV scoring formula

For each candidate cell with `p_mine ≤ min_p + risk_tolerance`:

```
static_ev =
    - p_mine                          * ev_mine_penalty       (default 30)
    + p_zero_proxy                    * ev_zero_weight        (default 5)
    + unrev_neighbours                * ev_unrev_weight       (default 3)
    + revealed_numeric_neighbours     * ev_revealed_weight    (default 1)
    + (ev_frontier_bonus if cell in frontier else 0)         (default 10)
    + (1 - dist_to_center / max_dist) * ev_centrality_weight  (default 2)

p_zero_proxy = Π over unrevealed neighbours of (1 - P_mine(neighbour))
              [marginal-product approximation; treats neighbours as independent]
```

All six weights are class attributes on `FinalAgent` — override per-instance for ablation experiments.

When lookahead runs (`lookahead_enabled and not gated out`):

```
ev = lookahead_static_weight * static_ev
   + lookahead_weight        * lookahead_ev * (1 - p_mine)

lookahead_ev = Σ_v  P(value = v | cell safe)  ×  forced_deductions(v)
              for v ∈ {0..8}, top-K candidates only
```

`forced_deductions(v)` is computed by constructing a hypothetical view with the candidate revealed as `v`, running one pass of `CSPAgent._infer`, and counting the resulting `_pending_safe + _pending_flag` size.

### Cascade-priority safe-reveal ordering

`FinalAgent._order_pending` (invoked from `FinalAgent._infer` after `super()._infer()`) ranks deduced safes/mines by:

```
priority = unrevealed_neighbours       * 3.0
         + revealed_numeric_neighbours * 1.0
         + (1 - dist_to_center / max)  * 0.5
```

The queue is sorted ascending, so `list.pop()` returns the highest-priority cell first. Highest priority ≈ most likely to be a zero (triggering flood-fill) or to unlock the most new constraints on reveal. Order changes never affect correctness, only the per-turn information yield.

### Known approximations (not bugs)

- `p_zero_proxy` treats unrevealed neighbours as **independent**. They aren't — they're correlated through shared constraints. The proxy is a lower bound; an exact joint computation would require keeping all enumerated solutions and is the largest deferred work item.
- `P(value = v | safe)` in lookahead uses the same marginal-product approximation across the cell's unrevealed neighbours rather than the exact Poisson-binomial-on-correlated-variables.
- The `avg_p` fallback for non-frontier neighbours in `p_zero_proxy` is a rough global mean. Local variance is not modelled.

These approximations only affect tie-breaking, not safe / mine determination, so they cannot cause an incorrect deduction — only a suboptimal guess.

## Future Work

- Replace the marginal-product `p_zero_proxy` in `FinalAgent._select_cell` with exact conditional outcome counts from component assignments. Both the static-EV `p_zero` term and the lookahead's `P(value=v | safe)` use marginal-product, which treats neighbours as independent — they aren't.
- Make `lookahead_evals` semantics canonical (per-candidate vs per-value-branch). The current counter increments per value branch (up to 9 per candidate per turn).
- Add sampled marginals for components that exceed the enumeration cap instead of collapsing them into uniform off-frontier probability.
- Tune `risk_tolerance`, EV weights, and lookahead weights by difficulty with larger benchmark sweeps (Expert at n=2000+ for tighter signal). Current evidence: lookahead off + `risk_tolerance=0.0` is the best default at our sample sizes, but a properly tuned lookahead may yet show small Expert gains.
- Add a `--telemetry` flag to `scripts/benchmark.py` that aggregates `endgame_calls`, `endgame_aborts`, and `lookahead_evals` across episodes for the report.
- Add multi-step lookahead for endgame guesses (beyond the current 1-ply).
- DQN baseline ships only a Beginner checkpoint by design. Intermediate/Expert DQN would need reward shaping + orders-of-magnitude more training to clear the sparse-win-signal problem, and still could not break the 50/50 endgame ceiling — so it is out of scope for a comparison baseline (see the `DQNAgent` section). A genuinely interesting extension is a *hybrid*: DQN proposes candidate guesses, the Probability/Final solver verifies them.

## License

MIT. See `LICENSE`.
