/**
 * Turn executor v5: army actions → battles → army actions → recruit.
 *
 * Decision loop:
 *   Phase 1: Army actions (pre-battle positioning)
 *     - MOVE destination: masked softmax over 16 nodes (one pick, always resolves)
 *   Phase 2: Battle loop
 *     - Each step: softmax over {attackable nodes, stop}; stop ends the phase
 *     - Army selection: autoregressive argmax over {remaining armies, done};
 *       done is offered only after the first army, so a chosen node is always attacked
 *     - startBattle → battle rounds → resolveBattle → next step
 *   Phase 3: Army actions (post-battle movement)
 *   Phase 4: Recruitment
 *   endTurn
 */
import type GameSystem from "../lib/GameSystem";
import type Army from "../lib/Army";
import type Battle from "../lib/Battle";
import {BattlePhase, BattleResult} from "../lib/Battle";
import {calculateUnitsNeeded} from "../lib/Combat";
import type {NNModel, NNPrediction} from "./nn/NNModel";
import {applyMaskAndSoftmax, argmax, sampleFromProbs, BATTLE_TARGET_DIM, BATTLE_TARGET_STOP} from "./nn/NNModel";
import {battleStuckReport} from "./battleReport";
import {
  encodeState, NODE_ORDER, NUM_NODES, UNIT_TYPES, NUM_UNIT_TYPES,
} from "./nn/StateEncoder";
import type {DecisionContext} from "./nn/StateEncoder";
import {
  computeActionTypeMask, executeArmyAction,
  ACTION_EXIT, ACTION_MOVE, ACTION_SPLIT,
} from "./nn/ActionSpace";

const MAX_STEPS_PER_ARMY = 10;

// ─── Turn options (exploration + recording) ───

export interface TurnOptions {
  /** Temperature for action type sampling. 0 = argmax (default). */
  temperature?: number;
  /** Probability of taking a completely random legal action. */
  epsilon?: number;
  /** Called after each decision with the state, prediction, and chosen action. */
  onDecision?: (info: DecisionInfo) => void;
}

export interface DecisionInfo {
  playerIdx: number;
  state: Float32Array;
  pred: NNPrediction;
  decisionType: string;
  /** The action actually taken (after exploration). */
  action: Record<string, number>;
  /** True when the ε branch fired: the action is uniform noise, not the policy. */
  explored: boolean;
}

// ─── Exploration helpers ───

/** Sample from softmax with temperature, or argmax if temp=0. */
function chooseAction(logits: Float32Array, mask: Float32Array, temp: number, eps: number): {choice: number; explored: boolean} {
  // Epsilon-greedy: random legal action
  if (eps > 0 && Math.random() < eps) {
    const legal = [...mask.keys()].filter(i => mask[i] > 0);
    return {choice: legal[Math.floor(Math.random() * legal.length)], explored: true};
  }
  if (temp > 0) {
    // Temperature-scaled softmax sampling (on-policy, not marked as exploration)
    const scaled = Float32Array.from(logits, (v, i) => (mask[i] > 0 ? v / temp : -Infinity));
    const probs = applyMaskAndSoftmax(scaled, mask);
    return {choice: sampleFromProbs(probs), explored: false};
  }
  return {choice: argmax(applyMaskAndSoftmax(logits, mask)), explored: false};
}

/** Sample a binary decision with exploration. */
function chooseBinary(value: number, eps: number): {choice: boolean; explored: boolean} {
  if (eps > 0 && Math.random() < eps) return {choice: Math.random() > 0.5, explored: true};
  return {choice: value > 0.5, explored: false};
}

/** Add noise to a fraction [0,1]. */
function noisyFraction(value: number, eps: number): {value: number; explored: boolean} {
  if (eps > 0 && Math.random() < eps) return {value: Math.random(), explored: true};
  return {value, explored: false};
}

// ─── Phase 1 & 3: Army actions ───

function* armyActionsPhase(
  game: GameSystem, model: NNModel, playerIdx: number, opts: TurnOptions,
): Generator<void> {
  const player = game.players[playerIdx];
  const processed = new Set<object>();
  const noMerge = new Set<object>();
  // Step budgets are conserved across split/merge lineages: a split child
  // inherits the parent's remaining steps and merges never refund, so the
  // phase total is bounded even when the policy splits endlessly.
  const budgets = new Map<Army, number>();
  for (const a of player.armies) budgets.set(a, MAX_STEPS_PER_ARMY);
  const temp = opts.temperature ?? 0;
  const eps = opts.epsilon ?? 0;
  let ai = 0;
  while (ai < player.armies.length) {
    const army = player.armies[ai];
    if (processed.has(army)) { ai++; continue; }
    processed.add(army);

    let remaining = budgets.get(army) ?? MAX_STEPS_PER_ARMY;
    while (remaining > 0) {
      if (army.units.length === 0) break;
      remaining--;

      const actionMask = computeActionTypeMask(game, player, army, noMerge);
      const context: DecisionContext = {type: "army", army, actionTypeMask: actionMask};
      const state = encodeState(game, playerIdx, context);
      const pred = model.predict(state);

      const {choice: actionType, explored: actionExplored} = chooseAction(pred.actionTypeLogits, actionMask, temp, eps);
      if (actionType === ACTION_EXIT) {
        opts.onDecision?.({playerIdx, state, pred, decisionType: "army", action: {actionType}, explored: actionExplored});
        break;
      }

      const armiesBefore = actionType === ACTION_SPLIT ? new Set(player.armies) : null;

      let targetIdx = 0;
      if (actionType === ACTION_MOVE) {
        const movable = army.getMovableLocations(game.gameMap, game.enemyLocations);
        if (movable.length === 0) break;
        const legalMask = new Float32Array(NUM_NODES);
        for (const dest of movable) {
          const nodeIdx = NODE_ORDER.indexOf(dest);
          if (nodeIdx >= 0) legalMask[nodeIdx] = 1;
        }
        const moveContext: DecisionContext = {type: "moveTarget", army, legalMask};
        const moveState = encodeState(game, playerIdx, moveContext);
        const movePred = model.predict(moveState);
        const move = chooseAction(movePred.moveTargetLogits, legalMask, temp, eps);
        targetIdx = move.choice;
        opts.onDecision?.({playerIdx, state: moveState, pred: movePred, decisionType: "moveTarget",
          action: {moveTarget: targetIdx}, explored: move.explored});
      }

      const frac = actionType === ACTION_SPLIT
        ? noisyFraction(pred.splitFraction, eps)
        : noisyFraction(pred.disbandFraction, eps);
      const fraction = frac.value;

      opts.onDecision?.({playerIdx, state, pred, decisionType: "army",
        action: {actionType, targetIdx, fraction}, explored: actionExplored || frac.explored});

      if (!executeArmyAction(game, player, army, actionType, targetIdx, fraction)) break;

      if (armiesBefore) {
        for (const a of player.armies) {
          if (!armiesBefore.has(a)) {
            noMerge.add(a);
            budgets.set(a, remaining); // split child inherits remaining budget
          }
        }
      }

      const newIdx = player.armies.indexOf(army);
      if (newIdx < 0) break;
      ai = newIdx;
      yield;
    }

    const finalIdx = player.armies.indexOf(army);
    ai = finalIdx < 0 ? ai : finalIdx + 1;
  }
}

// ─── Phase 2: Battle loop ───

function computeSelectionState(selected: Army[], remaining: Army[]) {
  const selectedPerType = new Array(6).fill(0);
  const remainingPerType = new Array(6).fill(0);
  const selectedHpSums = [0, 0, 0];
  const remainingHpSums = [0, 0, 0];

  for (const army of selected) {
    const t = (UNIT_TYPES as readonly string[]).indexOf(army.unitType);
    if (t < 0) continue;
    selectedPerType[t * 2] += army.units.length;
    selectedHpSums[t] += army.units.reduce((s, u) => s + u.currentHealth / u.health, 0);
  }
  for (const army of remaining) {
    const t = (UNIT_TYPES as readonly string[]).indexOf(army.unitType);
    if (t < 0) continue;
    remainingPerType[t * 2] += army.units.length;
    remainingHpSums[t] += army.units.reduce((s, u) => s + u.currentHealth / u.health, 0);
  }

  for (let t = 0; t < NUM_UNIT_TYPES; t++) {
    const selUnits = selectedPerType[t * 2];
    selectedPerType[t * 2 + 1] = selUnits > 0 ? selectedHpSums[t] / selUnits : 0;
    const remUnits = remainingPerType[t * 2];
    remainingPerType[t * 2 + 1] = remUnits > 0 ? remainingHpSums[t] / remUnits : 0;
  }

  return {selectedPerType, remainingPerType};
}

function* battleAllocatePhase(
  game: GameSystem, model: NNModel, battle: Battle,
  playerIdx: number, isAttacker: boolean, opts: TurnOptions,
): Generator<void> {
  const armies = isAttacker ? battle.attackerArmies : battle.defenderArmies;
  const eps = opts.epsilon ?? 0;

  for (const army of [...armies]) {
    if (!battle.canAct(army)) continue;
    if (battle.result !== BattleResult.Ongoing) return;

    const targets = battle.getTargetsInRange(army);
    if (targets.length === 0) continue;

    let remaining = army.units.length;
    const allocations: Array<{target: Army, unitCount: number}> = [];

    for (let ti = 0; ti < targets.length; ti++) {
      if (remaining <= 0) break;
      const target = targets[ti];
      const killNeeded = calculateUnitsNeeded(army, target);
      const futureNeeded = targets.slice(ti + 1)
        .reduce((sum, t) => sum + calculateUnitsNeeded(army, t), 0);

      const context: DecisionContext = {
        type: "battleAllocate", army, remaining, enemyArmy: target,
        roundProgress: battle.round / battle.maxRounds, isAttacker, unitsNeeded: killNeeded,
      };
      const state = encodeState(game, playerIdx, context);
      const pred = model.predict(state);

      const kfChoice = noisyFraction(pred.killFraction, eps);
      const kf = kfChoice.value;

      opts.onDecision?.({playerIdx, state, pred, decisionType: "battleAllocate",
        action: {killFraction: kf}, explored: kfChoice.explored});

      if (futureNeeded < remaining) {
        const overflowPct = Math.min(1, (remaining - futureNeeded) / killNeeded);
        if (kf < overflowPct) {
          // NN under-allocating — force full allocation to this and all remaining targets
          const send = Math.min(killNeeded, remaining);
          if (send > 0) {
            allocations.push({target, unitCount: send});
            remaining -= send;
          }
          for (let tj = ti + 1; tj < targets.length; tj++) {
            if (remaining <= 0) break;
            const futureSend = Math.min(calculateUnitsNeeded(army, targets[tj]), remaining);
            if (futureSend > 0) {
              allocations.push({target: targets[tj], unitCount: futureSend});
              remaining -= futureSend;
            }
          }
          break;
        }
      }

      let unitCount = Math.round(kf * killNeeded);
      unitCount = Math.min(unitCount, remaining);
      if (unitCount > 0) {
        allocations.push({target, unitCount});
        remaining -= unitCount;
      }
    }

    // Always submit (empty = pass): marks the army acted so the phase can end.
    // A rejection here means an invariant broke; failing loud beats spinning forever.
    if (!battle.allocateAttack(army, allocations)) {
      throw new Error(battleStuckReport(
        `battleAllocatePhase: allocation rejected (${army.unitType}@${army.location}, ${allocations.length} allocs)`, battle));
    }
    yield;
  }
}

function* battleLoop(
  game: GameSystem, model: NNModel, battle: Battle, playerIdx: number, opts: TurnOptions,
): Generator<void> {
  const eps = opts.epsilon ?? 0;
  const maxIter = battle.maxRounds * 4 + 8;
  let iter = 0;

  while (battle.result === BattleResult.Ongoing) {
    if (++iter > maxIter) throw new Error(battleStuckReport("battleLoop stuck", battle));
    if (battle.phase === BattlePhase.AttackerTurn) {
      if (battle.actedArmies.size === 0) {
        const targetNodeIdx = NODE_ORDER.indexOf(battle.targetLocation);
        const context: DecisionContext = {
          type: "battleRetreat", targetNodeIdx, roundProgress: battle.round / battle.maxRounds,
        };
        const state = encodeState(game, playerIdx, context);
        const pred = model.predict(state);
        const retreat = chooseBinary(pred.battleRetreat, eps);
        opts.onDecision?.({playerIdx, state, pred, decisionType: "battleRetreat",
          action: {battleRetreat: retreat.choice ? 1 : 0}, explored: retreat.explored});
        if (retreat.choice) { battle.retreat(); yield; return; }
      }

      yield* battleAllocatePhase(game, model, battle, playerIdx, true, opts);
    } else {
      const defenderIdx = game.players.indexOf(battle.defenderPlayer);

      if (defenderIdx < 0 || battle.defenderPlayer.defeated) {
        battle.executeNeutralDefenderTurn();
      } else {
        yield* battleAllocatePhase(game, model, battle, defenderIdx, false, opts);
      }
      yield;
    }
  }
}

/**
 * Autoregressive army selection: each step scores every remaining candidate
 * plus a "done" option (offered once at least one army is committed) and
 * picks the argmax. Guarantees a non-empty selection.
 */
function selectBattleArmies(
  game: GameSystem, model: NNModel, playerIdx: number,
  targetNodeIdx: number, candidates: Army[], eps: number, opts: TurnOptions,
): Army[] {
  const selected: Army[] = [];
  const remaining = [...candidates];

  while (remaining.length > 0) {
    const {selectedPerType, remainingPerType} = computeSelectionState(selected, remaining);

    const options: Array<{army: Army | null; state: Float32Array; pred: NNPrediction}> = remaining.map(army => {
      const context: DecisionContext = {
        type: "battleSelect", army, targetNodeIdx, selectedPerType, remainingPerType, isDone: false,
      };
      const state = encodeState(game, playerIdx, context);
      return {army, state, pred: model.predict(state)};
    });
    if (selected.length > 0) {
      const context: DecisionContext = {
        type: "battleSelect", army: null, targetNodeIdx, selectedPerType, remainingPerType, isDone: true,
      };
      const state = encodeState(game, playerIdx, context);
      options.push({army: null, state, pred: model.predict(state)});
    }

    const explored = eps > 0 && Math.random() < eps;
    const pick = explored
      ? Math.floor(Math.random() * options.length)
      : options.reduce((best, o, i) => (o.pred.battleSelect > options[best].pred.battleSelect ? i : best), 0);

    options.forEach((o, i) => {
      opts.onDecision?.({playerIdx, state: o.state, pred: o.pred,
        decisionType: "battleSelect", action: {battleSelect: i === pick ? 1 : 0}, explored});
    });

    const chosen = options[pick];
    if (!chosen.army) break;
    selected.push(chosen.army);
    remaining.splice(remaining.indexOf(chosen.army), 1);
  }

  return selected;
}

function* battlePhase(
  game: GameSystem, model: NNModel, playerIdx: number, opts: TurnOptions,
): Generator<void> {
  const temp = opts.temperature ?? 0;
  const eps = opts.epsilon ?? 0;
  const fought = new Set<string>();

  // Terminates: every iteration either stops or adds a node to `fought`
  while (true) {
    const attackable = Array.from(game.attackableLocations).filter(n => !fought.has(n));
    if (attackable.length === 0) break;

    const attackableMask = new Float32Array(NUM_NODES);
    for (const location of attackable) {
      const nodeIdx = NODE_ORDER.indexOf(location);
      if (nodeIdx >= 0) attackableMask[nodeIdx] = 1;
    }
    const optionMask = new Float32Array(BATTLE_TARGET_DIM);
    optionMask.set(attackableMask);
    optionMask[BATTLE_TARGET_STOP] = 1;

    const targetContext: DecisionContext = {type: "battleTarget", attackableMask};
    const targetState = encodeState(game, playerIdx, targetContext);
    const targetPred = model.predict(targetState);
    const target = chooseAction(targetPred.battleTargetLogits, optionMask, temp, eps);
    const choice = target.choice;
    opts.onDecision?.({playerIdx, state: targetState, pred: targetPred, decisionType: "battleTarget",
      action: {battleTarget: choice}, explored: target.explored});
    if (choice === BATTLE_TARGET_STOP) break;

    const location = NODE_ORDER[choice];
    fought.add(location);
    const candidates = game.getArmiesInRange(location);
    if (candidates.length === 0) continue;

    const selected = selectBattleArmies(game, model, playerIdx, choice, candidates, eps, opts);
    if (selected.length === 0) continue;

    const battle = game.startBattle(location, selected);
    if (!battle) continue;

    yield;
    yield* battleLoop(game, model, battle, playerIdx, opts);
    game.resolveBattle();
    yield;
  }
}

// ─── Phase 4: Recruitment ───

function* recruitPhase(
  game: GameSystem, model: NNModel, playerIdx: number, opts: TurnOptions,
): Generator<void> {
  const player = game.players[playerIdx];
  const recruitLocs = game.recruitLocations;
  const eps = opts.epsilon ?? 0;

  for (const location of recruitLocs) {
    const locIdx = NODE_ORDER.indexOf(location);
    if (locIdx < 0) continue;

    for (const unitType of UNIT_TYPES) {
      // At the army cap the engine rejects the recruit; skip so no phantom decision is recorded
      if (game.countArmiesOfTypeAtLocation(player, unitType, location) >= game.maxArmiesPerTypeAtNode(location)) continue;
      const cost = game.unitStatsMap[unitType].cost;
      const affordable = Math.floor(player.money / cost);
      if (affordable <= 0) continue;

      const context: DecisionContext = {type: "recruit", locationIdx: locIdx, unitType, affordableCount: affordable};
      const state = encodeState(game, playerIdx, context);
      const pred = model.predict(state);

      const frac = noisyFraction(pred.recruitFraction, eps);
      opts.onDecision?.({playerIdx, state, pred, decisionType: "recruit",
        action: {recruitFraction: frac.value}, explored: frac.explored});

      const count = Math.round(frac.value * affordable);
      if (count <= 0) continue;
      if (!player.canBuy(unitType, count)) continue;

      if (!game.recruitPlayerArmy(unitType, location, count)) continue;
      yield;
    }
  }
}

// ─── Main entry points ───

const DEFAULT_OPTS: TurnOptions = {};

export function* executeNNTurnSteps(game: GameSystem, model: NNModel, opts?: TurnOptions): Generator<void> {
  const o = opts ?? DEFAULT_OPTS;
  const playerIdx = game.currentPlayerIndex;

  if (game.currentPlayer.defeated || game.gameOver) {
    game.endTurn();
    return;
  }

  // A loaded save can carry an in-progress battle; finish it before the phases
  if (game.currentBattle) {
    yield* battleLoop(game, model, game.currentBattle, playerIdx, o);
    game.resolveBattle();
    yield;
  }

  yield* armyActionsPhase(game, model, playerIdx, o);
  yield* battlePhase(game, model, playerIdx, o);
  yield* armyActionsPhase(game, model, playerIdx, o);
  yield* recruitPhase(game, model, playerIdx, o);

  game.endTurn();
}

export function executeNNTurn(game: GameSystem, model: NNModel, opts?: TurnOptions): void {
  const gen = executeNNTurnSteps(game, model, opts);
  while (!gen.next().done) { /* drain */ }
}

/** Execute one defender phase using NN model. */
export function executeNNDefenderPhase(game: GameSystem, model: NNModel, battle: Battle): void {
  const defenderIdx = game.players.indexOf(battle.defenderPlayer);
  if (defenderIdx < 0 || battle.defenderPlayer.defeated) {
    battle.executeNeutralDefenderTurn();
    return;
  }
  const gen = battleAllocatePhase(game, model, battle, defenderIdx, false, {});
  while (!gen.next().done) { /* drain */ }
}
