# AI System Design

## Project Structure

```
src/AI/
  index.ts              # Entry point: loads NN model, exposes aiTakeTurn/aiTurnSteps/greedyTurnStepsUI
  TurnExecutor.ts       # 4-phase NN decision loop (generator)
  nn/
    GameIndex.ts        # Fixed index order of the map nodes and the unit types
    UnitGroups.ts       # (remainingMoves, canAttack) unit groups, the unit of per-group decisions
    FractionBins.ts     # The FRACTION_BINS levels every fraction head chooses from
    StateEncoder.ts     # State → Float32Array[1360], 8 decision contexts
    NNModel.ts          # TF.js model v11: per-head context shortcut, 11 output heads, categorical fraction heads
    ActionSpace.ts      # 3 action types, masks, per-group execution

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
    GreedyAI.ts         # 1-step lookahead + rollout, quantile scoring, per-group count search, DAgger support
    SampleTypes.ts      # Sample interface and binary serialization
    SelfPlay.ts         # RL game runners with recording (pure simulation)
    TrajectoryStore.ts  # Persistent datasets: traj format, manifests, λ materialization
    setupBackend.ts     # TF.js WASM backend init
    nodeIO.ts           # Node.js file IO handler for TF.js
    trainUtils.ts       # Shared utilities, paths, constants
  python/
    app/
      config.py         # Binary format offsets (SAMPLE_FLOATS=1435)
      model.py          # PyTorch GraphConquestNN (11 heads, mirrors NNModel.ts)
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

## Unit Groups (UnitGroups.ts)

Turn state lives on units: `remainingMoves`, `canAttack`, `inBattle`. One army
per (player, node, unit type) holds units in mixed states, and a move, a
battle commitment or a disbanding takes a subset of them. The AI works on the
discrete part of that state, the **unit group** (remainingMoves, canAttack):

```
group = remainingMoves * 2 + (canAttack ? 0 : 1)
  0: 0 moves, can attack     1: 0 moves, cannot attack
  2: 1 move,  can attack     3: 1 move,  cannot attack
  4: 2 moves, can attack     5: 2 moves, cannot attack
```

Units of one army in the same group differ only in health, so the encoder
reports a unit count per group and the fraction heads output one level per
group. Which units of a group realise a count is fixed: the healthiest first
for moving and committing, the weakest first for disbanding (health
distribution is the one part of the unit state the fixed-size encoding does
not carry).

Decision-specific group lists:
- **move** (4): groups 2..5, the units with at least one move; a group is
  eligible for a destination when its moves reach it
- **commit** (3): groups 0, 2, 4, the attack-ready units by remaining moves
- **disband** (6): all groups

A per-group count is `round(fraction × group size)`; a selection that rounds
to nothing takes one unit from the group with the highest fraction, so a
chosen action always acts on at least one unit.

## Input Encoding (StateEncoder.ts, 1360 features)

1. **Game config** (5): interestRate/0.10, upkeepRate/0.20, turnCount/100, maxTurns/100, maxArmyAttacks/20
2. **Unit type stats** (18): 3 types × 6 stats (attack/9, defend/3, health/20, range/2, speed/2, cost/2)
3. **Player stats** (21): 3 players (self, opp1, opp2) × 7 (money/200, nodeIncome/68, interest/10, upkeep/20, totalUnits/200, nodeCount/16, defeated)
4. **Per-node** (1024): 16 nodes × 64 (income/10, canRecruit, owner[4], 4 factions × 3 types × 2 (units/(100/cost), avgHp), 3 types × own units per group[6], distance[16])
5. **Context** (292): decision type one-hot[8] + 8 context blocks (inactive = all 0)

Unit counts are scaled by cost/100 (units per 100 money). **Army info** (27),
shared by the army, moveTarget, moveCount and battleSelect blocks:
location[16] + type[3] + units + avgHp + units per group[6].

Context blocks:
- **recruit** (20): location[16] + type[3] + affordable/(200/cost)
- **army** (36): army info(27) + actionMask[3] + disbandMask[6] (non-empty disband groups)
- **moveTarget** (43): army info(27) + legal destination mask[16]
- **moveCount** (47): army info(27) + destination[16] + move group mask[4] (groups that reach the destination)
- **battleTarget** (16): attackable node mask[16]
- **battleSelect** (59): army info(27, zeroed for the done option) + targetNode[16] + selectedPerType[6] (committed units) + remainingPerType[6] (attack-ready units of the remaining candidates) + isDone(1) + commitMask[3]
- **battleAllocate** (46): myArmy(22) + enemyArmy(21) + attackProgress + isAttacker + unitsNeeded/500; unit counts and health are those of the battle contingents
- **battleRetreat** (17): targetNode[16] + attackProgress

All features centered: value -= 0.5.

## Output Heads (NNModel.ts, 11 heads)

Architecture v11: context-free trunk with per-head context shortcut and
categorical fraction heads. The encoder
emits 1360 features; the trunk consumes only the context-free core
state[0:1068], so the value head (trunk only) is a pure state value whose TD
differences are not polluted by decision-context switches. Policy heads receive
the decision type and their own context block via shortcut inputs.

```
state_core[1068] → Dense(1024,ReLU) → Dense(256,ReLU) = trunk[256]

Each head: concat(trunk, relevant_context) → Dense(64,ReLU) → output
  - value:            trunk only (no context)
  - action_type:      trunk + ctx_dt[8] + ctx_army[36]
  - move_fraction:    trunk + ctx_dt[8] + ctx_mcnt[47]
  - disband_fraction: trunk + ctx_dt[8] + ctx_army[36]
  - recruit_fraction: trunk + ctx_dt[8] + ctx_rec[20]
  - move_target:      trunk + ctx_dt[8] + ctx_mov[43]
  - battle_target:    trunk + ctx_dt[8] + ctx_btgt[16]
  - battle_select:    trunk + ctx_dt[8] + ctx_bsel[59]
  - commit_fraction:  trunk + ctx_dt[8] + ctx_bsel[59]
  - kill_fraction:    trunk + ctx_dt[8] + ctx_balloc[46]
  - battle_retreat:   trunk + ctx_dt[8] + ctx_bret[17]
```

| # | Head | Size | Activation | Decision type |
|---|------|------|------------|---------------|
| 0 | value | 1 | sigmoid | all (position quality) |
| 1 | actionType | 3 | linear | army (EXIT/MOVE/DISBAND), masked softmax |
| 2 | moveFraction | 4 × 5 | linear, softmax per group | moveCount (level per move group) |
| 3 | disbandFraction | 6 × 5 | linear, softmax per group | army (DISBAND, level per disband group) |
| 4 | recruitFraction | 5 | linear, softmax | recruit (level of the affordable count) |
| 5 | moveTarget | 16 | linear | moveTarget (destination node, masked softmax over legal) |
| 6 | battleTarget | 17 | linear | battleTarget (attackable node or stop, masked softmax) |
| 7 | battleSelect | 1 | sigmoid | battleSelect (score per option, argmax over {armies, done}) |
| 8 | commitFraction | 3 × 5 | linear, softmax per group | battleSelect (level per commit group of the chosen army) |
| 9 | killFraction | 5 | linear, softmax | battleAllocate (level of killNeeded) |
| 10 | battleRetreat | 1 | sigmoid | battleRetreat (retreat?) |

Categorical decisions (actionType/moveTarget/battleTarget) never produce "no action by
default": an option is always chosen from the masked softmax. Passivity exists only as
explicit options (EXIT, stop, done) that must outscore the alternatives.

Fraction heads (FractionBins.ts): a fraction is one of FRACTION_BINS = 5 levels
(0, 0.25, 0.5, 0.75, 1) with a softmax per unit group. Deterministic play takes
the expected level of each group (the mean of its softmax); data generation
samples a level per group at the temperature (or draws every group uniformly
with probability ε), so a played level is an on-policy sample that the
reinforcement phases reinforce with cross-entropy. The v10 sigmoid heads
received no reinforcement gradient: their targets were their own outputs.
Decoding the most likely level instead of the expected one scored 48W 0L 33D
over 81 games for the same imitation data (v10 regression: 71W 0L 10D): the
labels of a state scatter over the levels and their mean is the count the
labeler takes; with the expected level the same weights scored 69W 0L 12D.

## Decision Loop (TurnExecutor.ts, 4 phases)

```
Phase 1: Army actions (pre-battle)
  For each army, until EXIT:
    Select action type via masked softmax → argmax (temperature sample in training)
    If MOVE: masked softmax over 16 destination logits → one destination (always resolves),
             then a level per move group (moveFraction) → units per group that reach it
    If DISBAND: a level per disband group (disbandFraction) → units per group
    Execute action (MOVE/DISBAND)

Phase 2: Battle loop
  fought = {}
  Each step (until stop chosen or no attackable nodes left):
    Masked softmax over {attackable nodes not in fought} ∪ {stop} (17 logits)
    stop → phase ends
    node → select armies autoregressively:
      Each step: score every remaining candidate (battleSelect head) plus a
      "done" option (only offered after the first army); pick argmax; the
      chosen army commits its attack-ready units by the commitFraction of the
      same prediction. A chosen node is therefore always attacked with ≥1 army.
    Start battle → battle rounds:
      Attacker turn: retreat check (battleRetreat), then allocate:
        For each army × each enemy: killFraction (bounded by the battle contingent)
        Overflow logic: if future enemies can't consume remaining,
          ask AI → if fraction < overflowPct → system auto-fills all
      Defender turn: neutral/defeated defenders are played by the engine, else AI allocate
    Resolve battle, add node to fought

Phase 3: Army actions (post-battle)
  Same as Phase 1

Phase 4: Recruitment
  For each location × unit type:
    Ask recruitFraction, buy round(fraction × affordable)

endTurn
```

The army loops need no step budget: every MOVE spends at least one move point
and every DISBAND removes at least one unit. Units that move into an army
already processed in the phase do not act again until the next army phase.

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
- MOVE is tried all-in only (a partial move scores the same, the score reads
  node ownership); battles commit every attack-ready unit

**Lookahead greedy** (outer, the actual turn):
- Each decision: clone → try option → rollout (complete the remaining turn phases with simple greedy) → quantile → pick best
- Recruit: rollout additionally simulates the player's next turn (simpleNextTurn)
- Battle target: per step, simulate attacking each candidate node with all in-range armies; label = argmax over {nodes, skip}; the chosen node is attacked first (best-first order)
- Battle select: additive per step, evaluate adding each remaining army at its
  best per-group commitment (and stopping with the current selection); label =
  argmax; one sample per option, each army option also labelled with its
  commitment
- Battle allocate also records **defender** samples (isAttacker=false) for training

**Per-group count search** (`searchCounts`), used for MOVE, DISBAND and
commitment labels:
- Levels per group: exact counts for groups of at most 2 units, otherwise
  none / half / all
- The whole grid is evaluated when it has at most 27 vectors, otherwise two
  passes of coordinate descent from the all-in vector; the all-zero vector is
  never proposed (it is the EXIT option)
- Labels are stored as fractions of the group size, which the executor turns
  back into the same counts

**DAgger mode**: NN plays the game (encounters its own states), greedy provides labels at each decision point. Addresses distribution shift between greedy's states and NN's states. `generateData.ts imitation --dagger-model <name>` plays the DAgger games with a trained model (a fresh random-weight model without it). A second imitation round on 50 such games plus 70 greedy games did not help in this design: the model trained on it scored 16W 0L 11D over 27 games vs Random against 21W 0L 6D for the model trained on the 100-game greedy dataset alone, so phase 1 uses the default dataset.

**Config variance** (±25%, all phases + tests): unit stats (attack, defend, health, cost), node income, interest rate, upkeep rate, player starting money. Range and speed not randomized. Each game gets independent Graph clone. Shared via `createRandomizedGame()` in trainUtils.

**Position rotation**: greedy/NN plays as Blue(0), Red(1), Green(2), rotating across games (g%3).

## Training Pipeline

All phases consume a named dataset from the store and do not simulate.

**Phase 1 — Imitation (scripts/trainPhase1.ts <imitation dataset>)**:
1. Dataset: 10 Greedy vs Passive + 70 Greedy vs Random + 10 Greedy vs Greedy + 10 DAgger (NN vs Random) by default
2. Value target = game outcome (win=1, loss=0, draw=1/3; draw = uniform prior over 3 players, so drawn games carry no positive advantage)
3. 3 value-only samples per game (all players' perspectives)
4. Policy weight = 1 (pure imitation)
5. Python trains with --fresh (50 epochs) and --balance-actions 1 (env
   ACTION_BALANCE): the action-type loss is weighted by the inverse class
   frequency to that power, because MOVE is about 1 label in 6 and unweighted
   training under-learns it (MOVE recall 0.37 on the training set; the model
   then never occupies the nodes it clears and draws by hoarding). Measured on
   the 100-game default dataset, 27 games vs Random on the default config:
   power 1 gives 21W 0L 6D (MOVE recall 0.83), power 0.5 gives 14W 0L 13D
   (MOVE recall 0.65); DISBAND is over-predicted at power 1 (precision 0.06)
   without costing games
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
   labels (no policy target). Temperature sampling covers every head, the
   fraction levels included, so each played level is an on-policy sample. A magnitude cutoff was tried and removed: the weight
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

## Results of the v10 Run (2026-09-26)

Evaluations are argmax play vs Random; "81 randomized" is the unified baseline
(trainUtils.baselineEval, ±25% config variance), "27 default" is
scripts/test/testOnly.ts on the default config.

| Model | 81 randomized | 27 default |
|-------|---------------|------------|
| Greedy (GreedyAI.ts) | not run | 27W 0L 0D, avg win turn 5.3 |
| Phase 1, 100-game dataset, 50 epochs, action balance 1 (published) | 71W 0L 10D, avg win turn 15.4 (an earlier read: 68W 0L 13D) | 21W 0L 6D, avg win turn 13.8 |
| Phase 1, 200-game dataset, 80 epochs, action balance 1 | 53W 0L 28D, avg win turn 12.7 | 22W 0L 5D, avg win turn 13.9 |

Draws are games the model dominates without occupying the opponents' last
recruit nodes; it never loses to Random. With exploration (T=1, ε=0.1) the
published model won 486 of 500 vs-random games.

Both reinforcement phases were gated out for both phase 1 models: phase 2 gave
39W 0L 42D from the 68W baseline (rollback) and 52W from the 53W baseline
(no gain); phase 3 gave 44W from 53W (rollback) and 7W 0L 74D from 71W
(rollback, the draw attractor). The per-group fraction heads receive no policy
gradient in these phases (their targets are the model's own outputs, so the
head losses start at 0), and the categorical heads' losses rise during the
4 epochs. training/model/phase1, phase2 and phase3 therefore all hold the
published phase 1 model. v11 replaces the sigmoid fraction heads with
categorical levels for this reason.

## Binary Sample Format (1435 floats per sample)

```
state[1360] + value(1) + policyWeight(1)
+ actionTypeTarget(1) + actionTypeMask[3]
+ moveFraction[4] + moveFractionMask[4]
+ disbandFraction[6] + disbandMask[6]
+ recruitFraction(1) + recruitMask(1)
+ moveTargetIdx(1) + moveMask[16]
+ battleTargetIdx(1) + battleTargetMask[17]
+ battleSelect(1) + battleSelectMask(1)
+ commitFraction[3] + commitMask[3]
+ killFraction(1) + killFracMask(1)
+ battleRetreat(1) + retreatMask(1)
```

moveTargetIdx / battleTargetIdx are class indices (-1 = no label); their masks list
the legal options (battleTargetMask[16] = stop, always 1). Both train with masked
cross-entropy. battleSelect rows are per-option (chosen = 1, others = 0, one row per
candidate army plus the done option) and train with BCE; inference takes the argmax
across the step's option scores. Fraction targets are stored as fractions (the
labeler's count over the group size, or the played level); the trainer maps each to
its nearest level and trains the head with cross-entropy per group, masked by the
group masks (a group mask marks the groups the decision could draw from; the RL
recorder reads the masks back from the recorded state).
