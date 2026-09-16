# AI System Design

## Project Structure

```
src/AI/
  index.ts              # Entry point: loads NN model, exposes aiTakeTurn/aiTurnSteps/greedyTurnStepsUI
  TurnExecutor.ts       # 4-phase NN decision loop (generator)
  nn/
    StateEncoder.ts     # State → Float32Array[1148], 7 decision contexts
    NNModel.ts          # TF.js model v8: per-head context shortcut, 10 output heads
    ActionSpace.ts      # 5 action types, masks, execution

training/
  scripts/
    generateData.ts     # Only producer of simulation data → trajectory store
    trainPhase1.ts      # Phase 1: imitation learning on an imitation dataset
    trainPhase2.ts      # Phase 2: RL on a vs-random dataset (TD(λ) weighting)
    trainPhase3.ts      # Phase 3: mixed RL on a mixed dataset
    publishModel.ts     # Publish a trained model → public/model/ (web UI)
    test/               # Model comparison / value-head evaluation scripts
  src/
    Opponents.ts        # random/passive opponents (same 4-phase loop as NN)
    GreedyAI.ts         # 1-step lookahead + rollout, quantile scoring, DAgger support
    SampleTypes.ts      # Sample interface and binary serialization
    SelfPlay.ts         # RL game runners with recording (pure simulation)
    TrajectoryStore.ts  # Persistent datasets: traj format, manifests, λ materialization
    setupBackend.ts     # TF.js WASM backend init
    nodeIO.ts           # Node.js file IO handler for TF.js
    trainUtils.ts       # Shared utilities, paths, constants
  python/
    app/
      config.py         # Binary format offsets (SAMPLE_FLOATS=1203)
      model.py          # PyTorch GraphConquestNN (10 heads, mirrors NNModel.ts)
      trainer.py        # Train/eval with policyWeight-weighted losses
      export_tfjs.py    # PyTorch ↔ TF.js weight conversion
      data_io.py        # Read binary sample files
      scripts/train.py  # CLI: load data → train → export
```

## Before Running Training

Kill all stale processes first:

```bash
taskkill //F //IM python.exe 2>/dev/null
taskkill //F //IM uv.exe 2>/dev/null
tasklist | grep "node" | awk '{print $2}' | while read pid; do taskkill //F //PID $pid 2>/dev/null; done
```

Log output: `training/log/phase1.log` / `phase2.log` / `phase3.log` /
`generate.log` (cleared on each run; one fixed file per script, dataset
provenance lives in the manifests).

## Data / Training Split

Simulation is expensive and training is cheap, so they are decoupled:
`generateData.ts` is the only place games are played for data, and it persists
per-game trajectories to the store; the phase scripts only consume datasets.
A dataset stays valid until one of its simulation inputs changes: game rules,
encoders, the runners, or the weights of any model that played in it. Model
weights are tracked as md5 in the manifest and checked by the consumers
(override with ALLOW_STALE=1); code changes are the operator's judgment call
(simRev in the manifest records the generating commit).

Store layout (`training/data/store/<name>/`, format spec in TrajectoryStore.ts):
- `manifest.json` — type, params, generating-model md5s, stats, simRev
- `trajectories.bin` — traj format: per-game records/critic snapshots/outcomes;
  advantages are NOT stored, they are recomputed at materialization so one
  dataset serves any λ
- `samples.bin` — sample format (imitation only): finished trainer samples

Formats are not versioned; an incompatible file fails loudly and is
regenerated.

Typical flow:

```bash
npx tsx training/scripts/generateData.ts imitation --out imit-v1
npx tsx training/scripts/trainPhase1.ts imit-v1
npx tsx training/scripts/generateData.ts vs-random --out vsr-p1-v1 --games 2000
npx tsx training/scripts/trainPhase2.ts vsr-p1-v1
npx tsx training/scripts/generateData.ts mixed --out mix-p2-v1
npx tsx training/scripts/trainPhase3.ts mix-p2-v1
```

## Input Encoding (StateEncoder.ts, 1148 features)

1. **Game config** (5): interestRate/0.10, upkeepRate/0.20, turnCount/100, maxTurns/100, maxArmyAttacks/20
2. **Unit type stats** (18): 3 types × 6 stats (attack/9, defend/3, health/20, range/2, speed/2, cost/2)
3. **Player stats** (21): 3 players (self, opp1, opp2) × 7 (money/200, nodeIncome/68, interest/10, upkeep/20, totalUnits/200, nodeCount/16, defeated)
4. **Per-node** (880): 16 nodes × 55 (income/10, canRecruit, owner[4], 4 factions × 3 types × 2 (units/(100/cost), avgHp), 3 types × armyCount/4, 3 types × 2 (maxMoves/2, canAttack), distance[16])
5. **Context** (224): decision type one-hot[7] + 7 context blocks (inactive = all 0)

Context blocks:
- **recruit** (20): location[16] + type[3] + affordable/(200/cost)
- **army** (28): location[16] + type[3] + units/(100/cost) + avgHp + moves/2 + canAttack + actionMask[5]
- **moveTarget** (39): army info(23) + legal destination mask[16]
- **battleTarget** (16): attackable node mask[16]
- **battleSelect** (51): army info(22, zeroed for the done option) + targetNode[16] + selectedPerType[6] + remainingPerType[6] + isDone(1)
- **battleAllocate** (46): myArmy(22) + enemyArmy(21) + attackProgress + isAttacker + unitsNeeded/500
- **battleRetreat** (17): targetNode[16] + attackProgress

All features centered: value -= 0.5.

## Output Heads (NNModel.ts, 10 heads)

Architecture v9: context-free trunk with per-head context shortcut. The encoder
still emits 1148 features; the trunk consumes only the context-free core
state[0:924], so the value head (trunk only) is a pure state value whose TD
differences are not polluted by decision-context switches. Policy heads receive
the decision type and their own context block via shortcut inputs.

```
state_core[924] → Dense(1024,ReLU) → Dense(256,ReLU) = trunk[256]

Each head: concat(trunk, relevant_context) → Dense(64,ReLU) → output
  - value:            trunk only (no context)
  - action_type:      trunk + ctx_dt[7] + ctx_army[28]
  - split_fraction:   trunk + ctx_dt[7] + ctx_army[28]
  - disband_fraction: trunk + ctx_dt[7] + ctx_army[28]
  - recruit_fraction: trunk + ctx_dt[7] + ctx_rec[20]
  - move_target:      trunk + ctx_dt[7] + ctx_mov[39]
  - battle_target:    trunk + ctx_dt[7] + ctx_btgt[16]
  - battle_select:    trunk + ctx_dt[7] + ctx_bsel[51]
  - kill_fraction:    trunk + ctx_dt[7] + ctx_balloc[46]
  - battle_retreat:   trunk + ctx_dt[7] + ctx_bret[17]
```

| # | Head | Size | Activation | Decision type |
|---|------|------|------------|---------------|
| 0 | value | 1 | sigmoid | all (position quality) |
| 1 | actionType | 5 | linear | army (EXIT/MERGE/MOVE/SPLIT/DISBAND), masked softmax |
| 2 | splitFraction | 1 | sigmoid | army (SPLIT) |
| 3 | disbandFraction | 1 | sigmoid | army (DISBAND) |
| 4 | recruitFraction | 1 | sigmoid | recruit |
| 5 | moveTarget | 16 | linear | moveTarget (destination node, masked softmax over legal) |
| 6 | battleTarget | 17 | linear | battleTarget (attackable node or stop, masked softmax) |
| 7 | battleSelect | 1 | sigmoid | battleSelect (score per option, argmax over {armies, done}) |
| 8 | killFraction | 1 | sigmoid | battleAllocate (fraction of killNeeded) |
| 9 | battleRetreat | 1 | sigmoid | battleRetreat (retreat?) |

Categorical decisions (actionType/moveTarget/battleTarget) never produce "no action by
default": an option is always chosen from the masked softmax. Passivity exists only as
explicit options (EXIT, stop, done) that must outscore the alternatives.

## Decision Loop (TurnExecutor.ts, 4 phases)

```
Phase 1: Army actions (pre-battle)
  For each army:
    Select action type via masked softmax → argmax (temperature sample in training)
    If MOVE: masked softmax over 16 destination logits → one destination (always resolves)
    Execute action (MERGE/MOVE/SPLIT/DISBAND)

Phase 2: Battle loop
  fought = {}
  Each step (until stop chosen or no attackable nodes left):
    Masked softmax over {attackable nodes not in fought} ∪ {stop} (17 logits)
    stop → phase ends
    node → select armies autoregressively:
      Each step: score every remaining candidate (battleSelect head) plus a
      "done" option (only offered after the first army); pick argmax.
      A chosen node is therefore always attacked with ≥1 army.
    Start battle → battle rounds:
      Attacker turn: retreat check (battleRetreat), then allocate:
        For each army × each enemy: killFraction
        Overflow logic: if future enemies can't consume remaining,
          ask AI → if fraction < overflowPct → system auto-fills all
      Defender turn: neutral/defeated auto-play, or AI allocate
    Resolve battle, add node to fought

Phase 3: Army actions (post-battle)
  Same as Phase 1

Phase 4: Recruitment
  For each location × unit type:
    Ask recruitFraction, buy round(fraction × affordable)

endTurn
```

## Scoring Function

```
scorePlayer(game, playerIdx):
  nodeIncome = sum of income from owned nodes
  interest = floor(money × interestRate)
  upkeep = player.getUpkeep(upkeepRate)
  return nodeIncome + interest + upkeep
```

- Interest: rewards positive cash flow, punishes bankruptcy
- Upkeep: rewards maintaining military strength, encourages aggression

**Quantile**: normal CDF of (player score - mean) / totalMapIncome. Captures relative advantage with fixed scale.

## Greedy AI (GreedyAI.ts)

Two-layer architecture:

**Simple greedy** (inner, used inside rollouts):
- Each decision point: clone → try each option → quantile → pick best
- No further lookahead (prevents recursion)

**Lookahead greedy** (outer, the actual turn):
- Each decision: clone → try option → rollout (complete the remaining turn phases with simple greedy) → quantile → pick best
- Recruit: rollout additionally simulates the player's next turn (simpleNextTurn)
- Battle target: per step, simulate attacking each candidate node with all in-range armies; label = argmax over {nodes, skip}; the chosen node is attacked first (best-first order)
- Battle select: additive per step, simulate adding each remaining army (and stopping with the current selection); label = argmax; one sample per option
- Battle allocate also records **defender** samples (isAttacker=false) for training

**DAgger mode**: NN plays the game (encounters its own states), greedy provides labels at each decision point. Addresses distribution shift between greedy's states and NN's states.

**Config variance** (±25%, all phases + tests): unit stats (attack, defend, health, cost), node income, interest rate, upkeep rate, player starting money. Range and speed not randomized. Each game gets independent Graph clone. Shared via `createRandomizedGame()` in trainUtils.

**Position rotation**: greedy/NN plays as Blue(0), Red(1), Green(2), rotating across games (g%3).

## Training Pipeline

All phases consume a named dataset from the store and do not simulate.

**Phase 1 — Imitation (scripts/trainPhase1.ts <imitation dataset>)**:
1. Dataset: 10 Greedy vs Passive + 70 Greedy vs Random + 10 Greedy vs Greedy + 10 DAgger (NN vs Random) by default
2. Value target = game outcome (win=1, loss=0, draw=1/3; draw = uniform prior over 3 players, so drawn games carry no positive advantage)
3. 3 value-only samples per game (all players' perspectives)
4. Policy weight = 1 (pure imitation)
5. Python trains with --fresh flag (50 epochs)
6. Output: training/model/phase1/

**Phase 2 — Reinforcement (scripts/trainPhase2.ts <vs-random dataset>)**:
1. Dataset: NN(phase1) vs Random games; materialization assigns TD(λ)
   advantage weights (unified λ = 0.8, fixed by the paired λ sweep; see the
   TD_LAMBDA comments in the phase scripts for the evidence)
2. Advantage = λ-return over turn-level TD errors: δ = V(next own-turn start) -
   V(own-turn start) from context-free critic snapshots, the final interval
   bootstraps to the terminal outcome (win=1, loss=0, draw=1/3); the λ backward
   recursion propagates the terminal truth through the trajectory
3. Positive advantage weights only (w > 0, no magnitude threshold): reinforce
   improving turns, discard negative ones; ε-explored decisions carry only value
   labels (no policy target). A magnitude cutoff was tried and removed: the weight
   already scales the gradient, and the cutoff selection-biased later iterations
   toward the noisy tail once the critic flattened.
   INVARIANT: policy weights must stay non-negative. Negative-weight training has
   been introduced and abandoned repeatedly in this project and collapsed the
   policy every time (offline push-away objective is unbounded and never
   saturates, so ± advantage noise nets out as repulsion of all played actions).
4. Value samples: per-turn context-free snapshots + 3 terminal samples per game
5. Trains 4 epochs from the phase1 start, then the gate: an 81-game eval
   against the unified cached baseline of the base model (eval81.json in the
   model dir, measured once per weights md5, so every run is judged against
   the same reference); keep the trained model only if strictly better (wins,
   then average win turn), otherwise restore the starting point
6. Output: training/model/phase2/

**Phase 3 — Mixed Reinforcement (scripts/trainPhase3.ts <mixed dataset>)**:
1. Dataset: NN(phase2) games in two parts:
   - Part A: vs opponent rotation (Passive, Phase 1, Phase 2)
   - Part B: 3-NN self-play
2. Only current model's decisions recorded (vs opponents); all decisions recorded (self-play)
3. Same training signal and gate as Phase 2 (turn-level TD(λ) advantages,
   positive only; 81-game eval vs the unified cached phase2 baseline, keep
   only if strictly better)
4. 3 value-only samples per game (all players)
5. Loads Phase 2 model, trains 4 epochs (20 degraded in one iteration historically)
6. Output: training/model/phase3/

## Binary Sample Format (1203 floats per sample)

```
state[1148] + value(1) + policyWeight(1)
+ actionTypeTarget(1) + actionTypeMask[5]
+ splitFraction(1) + splitMask(1)
+ disbandFraction(1) + disbandMask(1)
+ recruitFraction(1) + recruitMask(1)
+ moveTargetIdx(1) + moveMask[16]
+ battleTargetIdx(1) + battleTargetMask[17]
+ battleSelect(1) + battleSelectMask(1)
+ killFraction(1) + killFracMask(1)
+ battleRetreat(1) + retreatMask(1)
```

moveTargetIdx / battleTargetIdx are class indices (-1 = no label); their masks list
the legal options (battleTargetMask[16] = stop, always 1). Both train with masked
cross-entropy. battleSelect rows are per-option (chosen = 1, others = 0, one row per
candidate army plus the done option) and train with BCE; inference takes the argmax
across the step's option scores.
