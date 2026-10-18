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
 *     - startBattle → battle rounds (the engine resolves the battle when it ends) → next step
 *   Phase 3: Army actions (post-battle movement)
 *   Phase 4: Recruitment
 *   endTurn
 */
import type GameAPI from "../lib/GameAPI";
import type {ArmySelection, AttackAllocation, ArmyData, UnitData, BattleData} from "../lib/GameAPI";
import {BattlePhase} from "../lib/GameAPI";
import type {NNModel, NNPrediction} from "./nn/NNModel";
import {
  applyMaskAndSoftmax, softmax, argmax, expectedFractions, groupLogits, sampleFromProbs, BATTLE_TARGET_DIM, BATTLE_TARGET_STOP,
} from "./nn/NNModel";
import {battleStuckReport} from "./battleReport";
import {NODE_ORDER, NUM_NODES, UNIT_TYPES} from "./nn/GameIndex";
import {keyOf, armyIdOf, attackReady, battleUnits, ArmyWalk} from "./nn/Armies";
import {encodeState, perTypeState} from "./nn/StateEncoder";
import type {DecisionContext} from "./nn/StateEncoder";
import {
  computeActionTypeMask, executeArmyAction, unitsToCommit,
  ACTION_EXIT, ACTION_MOVE, ACTION_DISBAND,
} from "./nn/ActionSpace";
import {NUM_MOVE_GROUPS, NUM_DISBAND_GROUPS, NUM_COMMIT_GROUPS, moveGroups, commitGroups, disbandGroups, groupMask, selectionsOf} from "./nn/UnitGroups";
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
  api: GameAPI, model: NNModel, playerIdx: number, opts: TurnOptions,
): Generator<void> {
  const temp = opts.temperature ?? 0;
  const eps = opts.epsilon ?? 0;
  const walk = new ArmyWalk(api, playerIdx);
  for (let next = walk.next(); next; next = walk.next()) {
    const {armyId} = next;
    let army = next.army;
    for (;;) {
      const state = walk.state;
      const actionMask = computeActionTypeMask(api, armyId);
      const disbandMask = groupMask(disbandGroups(army.units));
      const context: DecisionContext = {type: "army", army, actionTypeMask: actionMask, disbandMask};
      const input = encodeState(state, api, playerIdx, context);
      const pred = model.predict(input);

      const {choice: actionType, explored: actionExplored} = chooseAction(pred.actionTypeLogits, actionMask, temp, eps);
      const disband = chooseFractions(pred.disbandFractionLogits, NUM_DISBAND_GROUPS, temp, eps);
      opts.onDecision?.({playerIdx, state: input, pred, action: {type: "army", actionType, disbandFraction: disband.value},
        explored: actionExplored || (actionType === ACTION_DISBAND && disband.explored)});
      if (actionType === ACTION_EXIT) break;

      let targetIdx = 0;
      let fractions = disband.value;
      if (actionType === ACTION_MOVE) {
        const legalMask = new Float32Array(NUM_NODES);
        for (const dest of api.getMovableLocations(armyId)) {
          legalMask[NODE_ORDER.indexOf(dest)] = 1;
        }
        const moveContext: DecisionContext = {type: "moveTarget", army, legalMask};
        const moveInput = encodeState(state, api, playerIdx, moveContext);
        const movePred = model.predict(moveInput);
        const move = chooseAction(movePred.moveTargetLogits, legalMask, temp, eps);
        targetIdx = move.choice;
        opts.onDecision?.({playerIdx, state: moveInput, pred: movePred, action: {type: "moveTarget", moveTarget: targetIdx},
          explored: move.explored});

        const candidates = api.getMoveCandidates(armyId, NODE_ORDER[targetIdx]);
        const countContext: DecisionContext = {
          type: "moveCount", army, destinationIdx: targetIdx, groupMask: groupMask(moveGroups(candidates)),
        };
        const countInput = encodeState(state, api, playerIdx, countContext);
        const countPred = model.predict(countInput);
        const count = chooseFractions(countPred.moveFractionLogits, NUM_MOVE_GROUPS, temp, eps);
        fractions = count.value;
        opts.onDecision?.({playerIdx, state: countInput, pred: countPred, action: {type: "moveCount", moveFraction: fractions},
          explored: count.explored});
      }

      if (!executeArmyAction(api, armyId, army, actionType, targetIdx, fractions)) break;

      const after = walk.acted();
      if (!after) break;
      army = after;
      yield;
    }
  }
}

// ─── Phase 2: Battle loop ───

function* battleAllocatePhase(
  api: GameAPI, model: NNModel, battle: BattleData,
  playerIdx: number, isAttacker: boolean, opts: TurnOptions,
): Generator<void> {
  const owner = (isAttacker ? battle.attackerPlayer : battle.defenderPlayer).name;
  const armies = isAttacker ? battle.attackerArmies : battle.defenderArmies;
  const temp = opts.temperature ?? 0;
  const eps = opts.epsilon ?? 0;

  for (const army of armies) {
    const state = api.getState();
    const current = state.currentBattle;
    if (!current) return;
    const armyId = armyIdOf(owner, army);
    if (!api.canAct(armyId)) continue;

    const enemyOwner = (isAttacker ? current.defenderPlayer : current.attackerPlayer).name;
    const enemies = isAttacker ? current.defenderArmies : current.attackerArmies;
    const targetKeys = new Set(api.getTargetsInRange(armyId).map(keyOf));
    const targets = enemies.filter(enemy => targetKeys.has(keyOf(enemy)));
    const unitsNeeded = (target: ArmyData): number => api.getUnitsNeeded(armyId, armyIdOf(enemyOwner, target));

    let remaining = battleUnits(army).length;
    const allocations: AttackAllocation[] = [];

    for (let ti = 0; ti < targets.length; ti++) {
      if (remaining <= 0) break;
      const target = targets[ti];
      const killNeeded = unitsNeeded(target);
      const futureNeeded = targets.slice(ti + 1)
        .reduce((sum, t) => sum + unitsNeeded(t), 0);

      const context: DecisionContext = {
        type: "battleAllocate", army, remaining, enemyArmy: target,
        attackProgress: 1 - api.getRemainingAttacks(armyId) / current.maxArmyAttacks, isAttacker, unitsNeeded: killNeeded,
      };
      const input = encodeState(state, api, playerIdx, context);
      const pred = model.predict(input);

      const kfChoice = chooseFractions(pred.killFractionLogits, 1, temp, eps);
      const kf = kfChoice.value[0];

      opts.onDecision?.({playerIdx, state: input, pred, action: {type: "battleAllocate", killFraction: kf}, explored: kfChoice.explored});

      if (futureNeeded < remaining) {
        const overflowPct = Math.min(1, (remaining - futureNeeded) / killNeeded);
        if (kf < overflowPct) {
          // NN under-allocating — force full allocation to this and all remaining targets
          const send = Math.min(killNeeded, remaining);
          if (send > 0) {
            allocations.push({targetId: armyIdOf(enemyOwner, target), count: send});
            remaining -= send;
          }
          for (let tj = ti + 1; tj < targets.length; tj++) {
            if (remaining <= 0) break;
            const futureSend = Math.min(unitsNeeded(targets[tj]), remaining);
            if (futureSend > 0) {
              allocations.push({targetId: armyIdOf(enemyOwner, targets[tj]), count: futureSend});
              remaining -= futureSend;
            }
          }
          break;
        }
      }

      let unitCount = Math.round(kf * killNeeded);
      unitCount = Math.min(unitCount, remaining);
      if (unitCount > 0) {
        allocations.push({targetId: armyIdOf(enemyOwner, target), count: unitCount});
        remaining -= unitCount;
      }
    }

    // Always submit (empty = pass): marks the army acted so the phase can end.
    // A rejection here means an invariant broke; failing loud beats spinning forever.
    if (!api.allocateAttack(armyId, allocations)) {
      throw new Error(battleStuckReport(
        `battleAllocatePhase: allocation rejected (${army.unitType}@${army.location}, ${allocations.length} allocs)`, current));
    }
    yield;
  }
}

function* battleLoop(
  api: GameAPI, model: NNModel, playerIdx: number, opts: TurnOptions,
): Generator<void> {
  const eps = opts.epsilon ?? 0;
  let iter = 0;

  for (;;) {
    const state = api.getState();
    const battle = state.currentBattle;
    if (!battle) return;
    if (++iter > battle.maxArmyAttacks * 4 + 8) throw new Error(battleStuckReport("battleLoop stuck", battle));
    if (battle.phase === BattlePhase.AttackerTurn) {
      if (battle.actedArmies.size === 0) {
        const targetNodeIdx = NODE_ORDER.indexOf(battle.targetLocation);
        const attackerOwner = battle.attackerPlayer.name;
        const remainingAttacks = Math.min(...battle.attackerArmies.map(army => api.getRemainingAttacks(armyIdOf(attackerOwner, army))));
        const context: DecisionContext = {
          type: "battleRetreat", targetNodeIdx, attackProgress: 1 - remainingAttacks / battle.maxArmyAttacks,
        };
        const input = encodeState(state, api, playerIdx, context);
        const pred = model.predict(input);
        const retreat = chooseBinary(pred.battleRetreat, eps);
        opts.onDecision?.({playerIdx, state: input, pred, action: {type: "battleRetreat", battleRetreat: retreat.choice ? 1 : 0},
          explored: retreat.explored});
        if (retreat.choice) { api.retreat(); yield; return; }
      }

      yield* battleAllocatePhase(api, model, battle, playerIdx, true, opts);
    } else {
      const defenderIdx = state.players.indexOf(battle.defenderPlayer);
      yield* battleAllocatePhase(api, model, battle, defenderIdx, false, opts);
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
  api: GameAPI, state: ReturnType<GameAPI["getState"]>, model: NNModel, playerIdx: number,
  targetNodeIdx: number, candidates: ArmyData[], opts: TurnOptions,
): ArmySelection[] {
  const owner = state.players[playerIdx].name;
  const temp = opts.temperature ?? 0;
  const eps = opts.epsilon ?? 0;
  const selected = new Map<ArmyData, UnitData[]>();
  const remaining = [...candidates];

  while (remaining.length > 0) {
    const selectedPerType = perTypeState(selected);
    const remainingPerType = perTypeState(remaining.map(army => [army, attackReady(army)] as const));

    const options: Array<{army: ArmyData | null; input: Float32Array; pred: NNPrediction}> = remaining.map(army => {
      const context: DecisionContext = {
        type: "battleSelect", army, targetNodeIdx, selectedPerType, remainingPerType, isDone: false,
        commitMask: groupMask(commitGroups(attackReady(army))),
      };
      const input = encodeState(state, api, playerIdx, context);
      return {army, input, pred: model.predict(input)};
    });
    if (selected.size > 0) {
      const context: DecisionContext = {
        type: "battleSelect", army: null, targetNodeIdx, selectedPerType, remainingPerType, isDone: true,
        commitMask: new Float32Array(NUM_COMMIT_GROUPS),
      };
      const input = encodeState(state, api, playerIdx, context);
      options.push({army: null, input, pred: model.predict(input)});
    }

    const explored = eps > 0 && Math.random() < eps;
    const pick = explored
      ? Math.floor(Math.random() * options.length)
      : options.reduce((best, o, i) => (o.pred.battleSelect > options[best].pred.battleSelect ? i : best), 0);
    const chosen = options[pick];
    const commit = chosen.army ? chooseFractions(chosen.pred.commitFractionLogits, NUM_COMMIT_GROUPS, temp, eps) : null;

    options.forEach((o, i) => {
      opts.onDecision?.({playerIdx, state: o.input, pred: o.pred,
        action: {type: "battleSelect", chosen: i === pick ? 1 : 0, commitFraction: i === pick && commit ? commit.value : expectedFractions(o.pred.commitFractionLogits, NUM_COMMIT_GROUPS)},
        explored: explored || (i === pick && commit !== null && commit.explored)});
    });

    if (!chosen.army || !commit) break;
    selected.set(chosen.army, unitsToCommit(chosen.army, commit.value));
    remaining.splice(remaining.indexOf(chosen.army), 1);
  }

  return [...selected].map(([army, units]) => ({armyId: armyIdOf(owner, army), selections: selectionsOf(units)}));
}

function* battlePhase(
  api: GameAPI, model: NNModel, playerIdx: number, opts: TurnOptions,
): Generator<void> {
  const temp = opts.temperature ?? 0;
  const eps = opts.epsilon ?? 0;
  const fought = new Set<string>();

  // Terminates: every iteration either stops or adds a node to `fought`
  while (true) {
    const attackable = [...api.getAttackableLocations()].filter(n => !fought.has(n));
    if (attackable.length === 0) break;

    const attackableMask = new Float32Array(NUM_NODES);
    for (const location of attackable) {
      const nodeIdx = NODE_ORDER.indexOf(location);
      if (nodeIdx >= 0) attackableMask[nodeIdx] = 1;
    }
    const optionMask = new Float32Array(BATTLE_TARGET_DIM);
    optionMask.set(attackableMask);
    optionMask[BATTLE_TARGET_STOP] = 1;

    const state = api.getState();
    const targetContext: DecisionContext = {type: "battleTarget", attackableMask};
    const targetInput = encodeState(state, api, playerIdx, targetContext);
    const targetPred = model.predict(targetInput);
    const target = chooseAction(targetPred.battleTargetLogits, optionMask, temp, eps);
    const choice = target.choice;
    opts.onDecision?.({playerIdx, state: targetInput, pred: targetPred, action: {type: "battleTarget", battleTarget: choice},
      explored: target.explored});
    if (choice === BATTLE_TARGET_STOP) break;

    const location = NODE_ORDER[choice];
    fought.add(location);
    const candidateKeys = new Set(api.getArmiesInRange(location).map(keyOf));
    const candidates = state.players[playerIdx].armies.filter(army => candidateKeys.has(keyOf(army)));
    if (candidates.length === 0) continue;

    const attackers = selectBattleArmies(api, state, model, playerIdx, choice, candidates, opts);
    if (attackers.length === 0) continue;

    if (!api.startBattle(attackers, location)) continue;

    yield;
    yield* battleLoop(api, model, playerIdx, opts);
    yield;
  }
}

// ─── Phase 4: Recruitment ───

function* recruitPhase(
  api: GameAPI, model: NNModel, playerIdx: number, opts: TurnOptions,
): Generator<void> {
  const recruitLocs = api.getRecruitLocations();
  const temp = opts.temperature ?? 0;
  const eps = opts.epsilon ?? 0;

  for (const location of recruitLocs) {
    const locIdx = NODE_ORDER.indexOf(location);
    if (locIdx < 0) continue;

    for (const unitType of UNIT_TYPES) {
      const state = api.getState();
      const cost = state.unitStatsMap[unitType].cost;
      const affordable = Math.floor(state.players[playerIdx].money / cost);
      if (affordable <= 0) continue;

      const context: DecisionContext = {type: "recruit", locationIdx: locIdx, unitType, affordableCount: affordable};
      const input = encodeState(state, api, playerIdx, context);
      const pred = model.predict(input);

      const frac = chooseFractions(pred.recruitFractionLogits, 1, temp, eps);
      opts.onDecision?.({playerIdx, state: input, pred, action: {type: "recruit", recruitFraction: frac.value[0]}, explored: frac.explored});

      const count = Math.round(frac.value[0] * affordable);
      if (count <= 0) continue;
      if (!api.canBuy(unitType, count)) continue;

      if (!api.recruit(unitType, location, count)) continue;
      yield;
    }
  }
}

// ─── Main entry points ───

const DEFAULT_OPTS: TurnOptions = {};

export function* executeNNTurnSteps(api: GameAPI, model: NNModel, opts?: TurnOptions): Generator<void> {
  const o = opts ?? DEFAULT_OPTS;
  const state = api.getState();
  const playerIdx = state.currentPlayerIndex;

  if (state.players[playerIdx].defeated || api.isGameOver()) {
    api.endTurn();
    return;
  }

  // A loaded save can carry an in-progress battle; finish it before the phases
  if (api.hasBattle()) {
    yield* battleLoop(api, model, playerIdx, o);
    yield;
  }

  yield* armyActionsPhase(api, model, playerIdx, o);
  yield* battlePhase(api, model, playerIdx, o);
  yield* armyActionsPhase(api, model, playerIdx, o);
  yield* recruitPhase(api, model, playerIdx, o);

  api.endTurn();
}

export function executeNNTurn(api: GameAPI, model: NNModel, opts?: TurnOptions): void {
  const gen = executeNNTurnSteps(api, model, opts);
  while (!gen.next().done) { /* drain */ }
}

/** Execute one defender phase using NN model. */
export function executeNNDefenderPhase(api: GameAPI, model: NNModel, battle: BattleData): void {
  const defenderIdx = api.getState().players.findIndex(player => player.name === battle.defenderPlayer.name);
  const gen = battleAllocatePhase(api, model, battle, defenderIdx, false, {});
  while (!gen.next().done) { /* drain */ }
}
