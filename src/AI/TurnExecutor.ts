/**
 * Turn executor v7: army actions → battles → army actions → recruit.
 *
 * Decision loop:
 *   Phase 1: Army actions (pre-battle positioning)
 *     - action type: masked softmax over {EXIT, MOVE, DISBAND}
 *     - MOVE destination: masked softmax over 16 nodes (one pick, always resolves)
 *     - MOVE count: a level per move group of the units that can reach it
 *       (every fraction head is a softmax over FRACTION_BINS levels per group)
 *     - DISBAND count: a level per disband group
 *   Phase 2: Battle loop
 *     - Each step: softmax over {attackable nodes, stop}; stop ends the phase
 *     - Army selection: autoregressive argmax over {remaining armies, done};
 *       done is offered only after the first army, so a chosen node is always
 *       attacked; the chosen army commits a fraction per commit group of its
 *       attack-ready units
 *     - startBattle → battle rounds → resolveBattle → next step
 *   Phase 3: Army actions (post-battle movement)
 *   Phase 4: Recruitment
 *   endTurn
 */
import type GameSystem from "../lib/GameSystem";
import type Army from "../lib/Army";
import type Unit from "../lib/Unit";
import type Battle from "../lib/Battle";
import {BattlePhase, BattleResult} from "../lib/Battle";
import {calculateUnitsNeeded} from "../lib/Combat";
import type {NNModel, NNPrediction} from "./nn/NNModel";
import {
  applyMaskAndSoftmax, softmax, argmax, expectedFractions, groupLogits, sampleFromProbs, BATTLE_TARGET_DIM, BATTLE_TARGET_STOP,
} from "./nn/NNModel";
import {battleStuckReport} from "./battleReport";
import {NODE_ORDER, NUM_NODES, UNIT_TYPES} from "./nn/GameIndex";
import {encodeState, perTypeState} from "./nn/StateEncoder";
import type {DecisionContext} from "./nn/StateEncoder";
import {
  computeActionTypeMask, executeArmyAction, unitsToCommit,
  ACTION_EXIT, ACTION_MOVE, ACTION_DISBAND,
} from "./nn/ActionSpace";
import {NUM_MOVE_GROUPS, NUM_DISBAND_GROUPS, NUM_COMMIT_GROUPS, moveGroups, commitGroups, disbandGroups, groupMask} from "./nn/UnitGroups";
import {FRACTION_BINS, binToFraction} from "./nn/FractionBins";

// ─── Turn options (exploration + recording) ───

export interface TurnOptions {
  /** Temperature for action type sampling. 0 = argmax (default). */
  temperature?: number;
  /** Probability of taking a completely random legal action. */
  epsilon?: number;
  /** Called after each decision with the state, prediction, and chosen action. */
  onDecision?: (info: DecisionInfo) => void;
}

/** The action taken at a decision point (after exploration), one shape per decision type. */
export type DecisionAction =
  | {type: "recruit"; recruitFraction: number}
  | {type: "army"; actionType: number; disbandFraction: Float32Array}
  | {type: "moveTarget"; moveTarget: number}
  | {type: "moveCount"; moveFraction: Float32Array}
  | {type: "battleTarget"; battleTarget: number}
  | {type: "battleSelect"; chosen: number; commitFraction: Float32Array}
  | {type: "battleAllocate"; killFraction: number}
  | {type: "battleRetreat"; battleRetreat: number};

export interface DecisionInfo {
  playerIdx: number;
  state: Float32Array;
  pred: NNPrediction;
  action: DecisionAction;
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

/**
 * Choose a fraction per group of a fraction head: with probability eps every
 * group is drawn uniformly (exploration, value label only), otherwise each
 * group samples a level from its softmax at the temperature, or takes its
 * expected level at temp=0.
 */
function chooseFractions(logits: Float32Array, groups: number, temp: number, eps: number): {value: Float32Array; explored: boolean} {
  if (eps > 0 && Math.random() < eps) {
    return {value: Float32Array.from({length: groups}, () => binToFraction(Math.floor(Math.random() * FRACTION_BINS))), explored: true};
  }
  if (temp === 0) return {value: expectedFractions(logits, groups), explored: false};
  const value = new Float32Array(groups);
  for (let g = 0; g < groups; g++) {
    value[g] = binToFraction(sampleFromProbs(softmax(Float32Array.from(groupLogits(logits, g), v => v / temp))));
  }
  return {value, explored: false};
}

// ─── Phase 1 & 3: Army actions ───

// Terminates without a step budget: every MOVE spends at least one move point
// and every DISBAND removes at least one unit.
function* armyActionsPhase(
  game: GameSystem, model: NNModel, playerIdx: number, opts: TurnOptions,
): Generator<void> {
  const player = game.players[playerIdx];
  const processed = new Set<Army>();
  const temp = opts.temperature ?? 0;
  const eps = opts.epsilon ?? 0;
  let ai = 0;
  while (ai < player.armies.length) {
    const army = player.armies[ai];
    if (processed.has(army)) { ai++; continue; }
    processed.add(army);

    for (;;) {
      const actionMask = computeActionTypeMask(game, army);
      const disbandMask = groupMask(disbandGroups(army.units));
      const context: DecisionContext = {type: "army", army, actionTypeMask: actionMask, disbandMask};
      const state = encodeState(game, playerIdx, context);
      const pred = model.predict(state);

      const {choice: actionType, explored: actionExplored} = chooseAction(pred.actionTypeLogits, actionMask, temp, eps);
      const disband = chooseFractions(pred.disbandFractionLogits, NUM_DISBAND_GROUPS, temp, eps);
      opts.onDecision?.({playerIdx, state, pred, action: {type: "army", actionType, disbandFraction: disband.value},
        explored: actionExplored || (actionType === ACTION_DISBAND && disband.explored)});
      if (actionType === ACTION_EXIT) break;

      let targetIdx = 0;
      let fractions = disband.value;
      if (actionType === ACTION_MOVE) {
        const legalMask = new Float32Array(NUM_NODES);
        for (const dest of army.getMovableLocations(game.gameMap, game.enemyLocations)) {
          legalMask[NODE_ORDER.indexOf(dest)] = 1;
        }
        const moveContext: DecisionContext = {type: "moveTarget", army, legalMask};
        const moveState = encodeState(game, playerIdx, moveContext);
        const movePred = model.predict(moveState);
        const move = chooseAction(movePred.moveTargetLogits, legalMask, temp, eps);
        targetIdx = move.choice;
        opts.onDecision?.({playerIdx, state: moveState, pred: movePred, action: {type: "moveTarget", moveTarget: targetIdx},
          explored: move.explored});

        const candidates = army.getMoveCandidates(NODE_ORDER[targetIdx], game.gameMap, game.enemyLocations);
        const countContext: DecisionContext = {
          type: "moveCount", army, destinationIdx: targetIdx, groupMask: groupMask(moveGroups(candidates)),
        };
        const countState = encodeState(game, playerIdx, countContext);
        const countPred = model.predict(countState);
        const count = chooseFractions(countPred.moveFractionLogits, NUM_MOVE_GROUPS, temp, eps);
        fractions = count.value;
        opts.onDecision?.({playerIdx, state: countState, pred: countPred, action: {type: "moveCount", moveFraction: fractions},
          explored: count.explored});
      }

      if (!executeArmyAction(game, army, actionType, targetIdx, fractions)) break;

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

function* battleAllocatePhase(
  game: GameSystem, model: NNModel, battle: Battle,
  playerIdx: number, isAttacker: boolean, opts: TurnOptions,
): Generator<void> {
  const armies = isAttacker ? battle.attackerArmies : battle.defenderArmies;
  const temp = opts.temperature ?? 0;
  const eps = opts.epsilon ?? 0;

  for (const army of [...armies]) {
    if (!battle.canAct(army)) continue;
    if (battle.result !== BattleResult.Ongoing) return;

    const targets = battle.getTargetsInRange(army);

    let remaining = army.battleUnits.length;
    const allocations = new Map<Army, number>();

    for (let ti = 0; ti < targets.length; ti++) {
      if (remaining <= 0) break;
      const target = targets[ti];
      const killNeeded = calculateUnitsNeeded(army, target);
      const futureNeeded = targets.slice(ti + 1)
        .reduce((sum, t) => sum + calculateUnitsNeeded(army, t), 0);

      const context: DecisionContext = {
        type: "battleAllocate", army, remaining, enemyArmy: target,
        attackProgress: 1 - battle.getRemainingAttacks(army) / battle.maxArmyAttacks, isAttacker, unitsNeeded: killNeeded,
      };
      const state = encodeState(game, playerIdx, context);
      const pred = model.predict(state);

      const kfChoice = chooseFractions(pred.killFractionLogits, 1, temp, eps);
      const kf = kfChoice.value[0];

      opts.onDecision?.({playerIdx, state, pred, action: {type: "battleAllocate", killFraction: kf}, explored: kfChoice.explored});

      if (futureNeeded < remaining) {
        const overflowPct = Math.min(1, (remaining - futureNeeded) / killNeeded);
        if (kf < overflowPct) {
          // NN under-allocating — force full allocation to this and all remaining targets
          const send = Math.min(killNeeded, remaining);
          if (send > 0) {
            allocations.set(target, send);
            remaining -= send;
          }
          for (let tj = ti + 1; tj < targets.length; tj++) {
            if (remaining <= 0) break;
            const futureSend = Math.min(calculateUnitsNeeded(army, targets[tj]), remaining);
            if (futureSend > 0) {
              allocations.set(targets[tj], futureSend);
              remaining -= futureSend;
            }
          }
          break;
        }
      }

      let unitCount = Math.round(kf * killNeeded);
      unitCount = Math.min(unitCount, remaining);
      if (unitCount > 0) {
        allocations.set(target, unitCount);
        remaining -= unitCount;
      }
    }

    // Always submit (empty = pass): marks the army acted so the phase can end.
    // A rejection here means an invariant broke; failing loud beats spinning forever.
    if (!battle.allocateAttack(army, allocations)) {
      throw new Error(battleStuckReport(
        `battleAllocatePhase: allocation rejected (${army.unitType}@${army.location}, ${allocations.size} allocs)`, battle));
    }
    yield;
  }
}

function* battleLoop(
  game: GameSystem, model: NNModel, battle: Battle, playerIdx: number, opts: TurnOptions,
): Generator<void> {
  const eps = opts.epsilon ?? 0;
  const maxIter = battle.maxArmyAttacks * 4 + 8;
  let iter = 0;

  while (battle.result === BattleResult.Ongoing) {
    if (++iter > maxIter) throw new Error(battleStuckReport("battleLoop stuck", battle));
    if (battle.phase === BattlePhase.AttackerTurn) {
      if (battle.actedArmies.size === 0) {
        const targetNodeIdx = NODE_ORDER.indexOf(battle.targetLocation);
        const remainingAttacks = Math.min(...battle.attackerArmies.map((a) => battle.getRemainingAttacks(a)));
        const context: DecisionContext = {
          type: "battleRetreat", targetNodeIdx, attackProgress: 1 - remainingAttacks / battle.maxArmyAttacks,
        };
        const state = encodeState(game, playerIdx, context);
        const pred = model.predict(state);
        const retreat = chooseBinary(pred.battleRetreat, eps);
        opts.onDecision?.({playerIdx, state, pred, action: {type: "battleRetreat", battleRetreat: retreat.choice ? 1 : 0},
          explored: retreat.explored});
        if (retreat.choice) { battle.retreat(); yield; return; }
      }

      yield* battleAllocatePhase(game, model, battle, playerIdx, true, opts);
    } else {
      const defenderIdx = game.players.indexOf(battle.defenderPlayer);
      yield* battleAllocatePhase(game, model, battle, defenderIdx, false, opts);
      yield;
    }
  }
}

/**
 * Autoregressive army selection: each step scores every remaining candidate
 * plus a "done" option (offered once at least one army is committed) and
 * picks the argmax; the chosen army commits its attack-ready units by the
 * per-group fractions of the same prediction. Guarantees a non-empty selection.
 */
function selectBattleArmies(
  game: GameSystem, model: NNModel, playerIdx: number,
  targetNodeIdx: number, candidates: Army[], opts: TurnOptions,
): Map<Army, Unit[]> {
  const temp = opts.temperature ?? 0;
  const eps = opts.epsilon ?? 0;
  const selected = new Map<Army, Unit[]>();
  const remaining = [...candidates];

  while (remaining.length > 0) {
    const selectedPerType = perTypeState(selected);
    const remainingPerType = perTypeState(remaining.map(army => [army, army.attackCandidates] as const));

    const options: Array<{army: Army | null; state: Float32Array; pred: NNPrediction}> = remaining.map(army => {
      const context: DecisionContext = {
        type: "battleSelect", army, targetNodeIdx, selectedPerType, remainingPerType, isDone: false,
        commitMask: groupMask(commitGroups(army.attackCandidates)),
      };
      const state = encodeState(game, playerIdx, context);
      return {army, state, pred: model.predict(state)};
    });
    if (selected.size > 0) {
      const context: DecisionContext = {
        type: "battleSelect", army: null, targetNodeIdx, selectedPerType, remainingPerType, isDone: true,
        commitMask: new Float32Array(NUM_COMMIT_GROUPS),
      };
      const state = encodeState(game, playerIdx, context);
      options.push({army: null, state, pred: model.predict(state)});
    }

    const explored = eps > 0 && Math.random() < eps;
    const pick = explored
      ? Math.floor(Math.random() * options.length)
      : options.reduce((best, o, i) => (o.pred.battleSelect > options[best].pred.battleSelect ? i : best), 0);
    const chosen = options[pick];
    const commit = chosen.army ? chooseFractions(chosen.pred.commitFractionLogits, NUM_COMMIT_GROUPS, temp, eps) : null;

    options.forEach((o, i) => {
      opts.onDecision?.({playerIdx, state: o.state, pred: o.pred,
        action: {type: "battleSelect", chosen: i === pick ? 1 : 0, commitFraction: i === pick && commit ? commit.value : expectedFractions(o.pred.commitFractionLogits, NUM_COMMIT_GROUPS)},
        explored: explored || (i === pick && commit !== null && commit.explored)});
    });

    if (!chosen.army || !commit) break;
    selected.set(chosen.army, unitsToCommit(chosen.army, commit.value));
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
    opts.onDecision?.({playerIdx, state: targetState, pred: targetPred, action: {type: "battleTarget", battleTarget: choice},
      explored: target.explored});
    if (choice === BATTLE_TARGET_STOP) break;

    const location = NODE_ORDER[choice];
    fought.add(location);
    const candidates = game.getArmiesInRange(location);
    if (candidates.length === 0) continue;

    const selected = selectBattleArmies(game, model, playerIdx, choice, candidates, opts);
    if (selected.size === 0) continue;

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
  const temp = opts.temperature ?? 0;
  const eps = opts.epsilon ?? 0;

  for (const location of recruitLocs) {
    const locIdx = NODE_ORDER.indexOf(location);
    if (locIdx < 0) continue;

    for (const unitType of UNIT_TYPES) {
      const cost = game.unitStatsMap[unitType].cost;
      const affordable = Math.floor(player.money / cost);
      if (affordable <= 0) continue;

      const context: DecisionContext = {type: "recruit", locationIdx: locIdx, unitType, affordableCount: affordable};
      const state = encodeState(game, playerIdx, context);
      const pred = model.predict(state);

      const frac = chooseFractions(pred.recruitFractionLogits, 1, temp, eps);
      opts.onDecision?.({playerIdx, state, pred, action: {type: "recruit", recruitFraction: frac.value[0]}, explored: frac.explored});

      const count = Math.round(frac.value[0] * affordable);
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
  const gen = battleAllocatePhase(game, model, battle, defenderIdx, false, {});
  while (!gen.next().done) { /* drain */ }
}
