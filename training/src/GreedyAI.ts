/**
 * Greedy AI: 1-step lookahead with greedy rollout to end of turn.
 *
 * Each decision point:
 *   1. List options
 *   2. For each option: fork → execute → greedy rollout rest of turn → quantile
 *   3. Pick highest quantile
 *
 * Per-group counts (move, commit, disband) are searched by `searchCounts`: the
 * whole grid of per-group levels when it is small, otherwise coordinate descent
 * from the all-in vector.
 *
 * All lookahead functions are generators that yield after each real game action,
 * allowing the UI to render intermediate states.
 *
 * Pass samples array to greedyTurn to record decisions for imitation learning.
 */
import type GameAPI from "../../src/lib/GameAPI";
import type {ArmyId, ArmySelection, AttackAllocation} from "../../src/lib/GameAPI";
import type {GameData} from "../../src/lib/GameSystem";
import type {ArmyData} from "../../src/lib/Army";
import type {UnitData} from "../../src/lib/Unit";
import type {BattleData} from "../../src/lib/Battle";
import {BattlePhase} from "../../src/lib/Battle";
import {NODE_ORDER, NUM_NODES, UNIT_TYPES} from "../../src/AI/nn/GameIndex";
import {keyOf, armyIdOf, attackReady, battleUnits, ArmyWalk} from "../../src/AI/nn/Armies";
import {encodeState, perTypeState} from "../../src/AI/nn/StateEncoder";
import type {DecisionContext, Rules} from "../../src/AI/nn/StateEncoder";
import {
  computeActionTypeMask, executeArmyAction, unitsToCommit,
  ACTION_EXIT, ACTION_MOVE, ACTION_DISBAND,
} from "../../src/AI/nn/ActionSpace";
import {
  NUM_UNIT_GROUPS, NUM_MOVE_GROUPS, NUM_DISBAND_GROUPS, NUM_COMMIT_GROUPS,
  moveGroups, commitGroups, disbandGroups, groupMask, countsToFractions, takeByFractions, selectionsOf,
} from "../../src/AI/nn/UnitGroups";
import type {NNModel} from "../../src/AI/nn/NNModel";
import {applyMaskAndSoftmax, argmax, expectedFractions, BATTLE_TARGET_DIM, BATTLE_TARGET_STOP} from "../../src/AI/nn/NNModel";
import {battleStuckReport} from "../../src/AI/battleReport";
import type {Sample} from "./SampleTypes";
import {emptySample} from "./SampleTypes";

/** All-in fractions, long enough for any group list. */
const FULL_FRACTIONS = new Float32Array(NUM_UNIT_GROUPS).fill(1);

/** Largest per-group grid that is searched exhaustively; larger grids use coordinate descent. */
const EXHAUSTIVE_CAP = 27;

/** Recruit fractions of the affordable count tried by the labeler: the non-zero levels of the recruit head. */
const RECRUIT_LEVELS = [0.25, 0.5, 0.75, 1.0];

/** Legal destination mask [16] for an army's current movable locations. */
function moveLegalMask(api: GameAPI, armyId: ArmyId): Float32Array {
  const mask = new Float32Array(NUM_NODES);
  for (const dest of api.getMovableLocations(armyId)) {
    const ni = NODE_ORDER.indexOf(dest);
    if (ni >= 0) mask[ni] = 1;
  }
  return mask;
}

// ─── Score & quantile ───

/** Execute one defender phase using greedy logic. */
export function greedyDefenderPhase(api: GameAPI, battle: BattleData): void {
  simpleBattleAllocate(api, battle, false);
}

export function scorePlayer(state: GameData, rules: Rules, playerIdx: number): number {
  const player = state.players[playerIdx];
  let nodeIncome = 0;
  for (const [node, {data}] of state.gameMap.nodes) {
    if (state.nodeOwnership.get(node) === player) nodeIncome += data.income;
  }
  const interest = Math.floor(player.money * state.interestRate);
  const upkeep = rules.getUpkeep(player.name);
  return nodeIncome + interest + upkeep;
}

function normalCDF(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804014327;
  const p = d * Math.exp(-x * x / 2) * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.8212560 + t * 1.3302744))));
  return x >= 0 ? 1 - p : p;
}

let _cachedTotalIncome = 0;
function quantile(api: GameAPI, playerIdx: number): number {
  const state = api.getState();
  if (_cachedTotalIncome === 0) {
    for (const [, {data}] of state.gameMap.nodes) {
      _cachedTotalIncome += data.income;
    }
  }
  const scores = [0, 1, 2].map(i => scorePlayer(state, api, i));
  const mean = (scores[0] + scores[1] + scores[2]) / 3;
  return normalCDF((scores[playerIdx] - mean) / _cachedTotalIncome);
}

// ─── Forks and selections ───

/** Every army with all of its attack-ready units. */
function fullSelections(owner: string, armies: ArmyData[]): ArmySelection[] {
  return armies.map(army => ({army: armyIdOf(owner, army), selections: selectionsOf(attackReady(army))}));
}

/** The attackers of a battle: chosen armies, each with its chosen units. */
function attackersOf(owner: string, chosen: Iterable<readonly [ArmyData, UnitData[]]>): ArmySelection[] {
  return [...chosen].map(([army, units]) => ({army: armyIdOf(owner, army), selections: selectionsOf(units)}));
}

/** The player's armies in range of a location, in army order. */
function armiesInRange(api: GameAPI, state: GameData, playerIdx: number, location: string): ArmyData[] {
  const keys = new Set(api.getArmiesInRange(location).map(keyOf));
  return state.players[playerIdx].armies.filter(army => keys.has(keyOf(army)));
}

/** Ends turns, every player passing, until it is the player's turn; false when the game ends first. */
function passTo(api: GameAPI, playerIdx: number): boolean {
  while (api.getState().currentPlayerIndex !== playerIdx) {
    if (!api.endTurn()) return false;
  }
  return true;
}

// ─── Per-group count search ───

/** Count levels of one group for the labeler: exact counts for groups of at most 2 units, otherwise none, half, all. */
function groupLevels(size: number): number[] {
  if (size <= 2) return Array.from({length: size + 1}, (_, i) => i);
  return [0, Math.round(size / 2), size];
}

/**
 * Search the per-group counts that maximise `evaluate`: the whole grid when it
 * has at most EXHAUSTIVE_CAP vectors, otherwise two passes of coordinate
 * descent from the all-in vector. The all-zero vector is never proposed (it is
 * the EXIT option). Returns null when every group is empty.
 */
function searchCounts(groups: UnitData[][], evaluate: (counts: number[]) => number): {counts: number[]; q: number} | null {
  const levels = groups.map(group => groupLevels(group.length));
  const gridSize = levels.reduce((n, l) => n * l.length, 1);
  if (gridSize === 1) return null;

  let bestCounts: number[] | null = null;
  let bestQ = -Infinity;
  const consider = (counts: number[]): boolean => {
    if (counts.every(c => c === 0)) return false;
    const q = evaluate(counts);
    if (bestCounts !== null && q <= bestQ) return false;
    bestCounts = [...counts];
    bestQ = q;
    return true;
  };

  if (gridSize <= EXHAUSTIVE_CAP) {
    const index = new Array<number>(levels.length).fill(0);
    for (let n = 0; n < gridSize; n++) {
      consider(levels.map((l, g) => l[index[g]]));
      for (let g = 0; g < levels.length; g++) {
        if (++index[g] < levels[g].length) break;
        index[g] = 0;
      }
    }
  } else {
    const counts = groups.map(group => group.length);
    consider(counts);
    for (let pass = 0; pass < 2; pass++) {
      for (let g = 0; g < levels.length; g++) {
        for (const level of levels[g]) {
          if (level === counts[g]) continue;
          const trial = [...counts];
          trial[g] = level;
          if (consider(trial)) counts[g] = level;
        }
      }
    }
  }
  return bestCounts === null ? null : {counts: bestCounts, q: bestQ};
}

// ─── Simple greedy (no rollout, used inside rollout) ───

function simpleArmyActions(api: GameAPI, playerIdx: number): void {
  const walk = new ArmyWalk(api, playerIdx);
  for (let next = walk.next(); next; next = walk.next()) {
    const {armyId} = next;
    let army = next.army;
    for (;;) {
      const mask = computeActionTypeMask(api, armyId);
      let bestAction = ACTION_EXIT, bestTarget = 0, bestFractions = FULL_FRACTIONS;
      let bestQ = quantile(api, playerIdx);

      const tryAction = (actionType: number, targetIdx: number, fractions: Float32Array): number => {
        const c = api.fork();
        if (!executeArmyAction(c, armyId, army, actionType, targetIdx, fractions)) return -Infinity;
        return quantile(c, playerIdx);
      };

      // Moving a part of the reachable units scores the same as moving all of them
      // (the score reads node ownership), so only the all-in move is tried
      if (mask[ACTION_MOVE] > 0) {
        for (const dest of api.getMovableLocations(armyId)) {
          const ni = NODE_ORDER.indexOf(dest);
          const q = tryAction(ACTION_MOVE, ni, FULL_FRACTIONS);
          if (q > bestQ) { bestQ = q; bestAction = ACTION_MOVE; bestTarget = ni; bestFractions = FULL_FRACTIONS; }
        }
      }
      for (const f of [0.3, 0.5, 1.0]) {
        const fractions = new Float32Array(NUM_UNIT_GROUPS).fill(f);
        const q = tryAction(ACTION_DISBAND, 0, fractions);
        if (q > bestQ) { bestQ = q; bestAction = ACTION_DISBAND; bestFractions = fractions; }
      }

      if (bestAction === ACTION_EXIT) break;
      if (!executeArmyAction(api, armyId, army, bestAction, bestTarget, bestFractions)) break;
      const after = walk.acted();
      if (!after) break;
      army = after;
    }
  }
}

function simpleBattleAllocate(api: GameAPI, battle: BattleData, isAttacker: boolean): void {
  const owner = (isAttacker ? battle.attackerPlayer : battle.defenderPlayer).name;
  const armies = isAttacker ? battle.attackerArmies : battle.defenderArmies;
  for (const army of armies) {
    if (!api.hasBattle()) return;
    const armyId = armyIdOf(owner, army);
    if (!api.canAct(armyId)) continue;
    const targets = api.getTargetsInRange(armyId);
    let remaining = battleUnits(army).length;
    const allocations: AttackAllocation[] = [];
    for (const target of targets) {
      if (remaining <= 0) break;
      const killNeeded = api.getUnitsNeeded(armyId, target);
      const unitCount = Math.min(killNeeded, remaining);
      if (unitCount > 0) allocations.push({target, count: unitCount});
      remaining -= killNeeded;
    }
    // Always submit (empty = pass) so the army is marked acted; reject = broken invariant
    if (!api.allocateAttack(armyId, allocations)) {
      throw new Error(battleStuckReport(
        `simpleBattleAllocate: allocation rejected (${army.unitType}@${army.location})`, battle));
    }
  }
}

function simpleBattleLoop(api: GameAPI): void {
  let iter = 0;
  for (;;) {
    const battle = api.getState().currentBattle;
    if (!battle) return;
    if (++iter > battle.maxArmyAttacks * 4 + 8) {
      throw new Error(battleStuckReport("simpleBattleLoop stuck", battle));
    }
    simpleBattleAllocate(api, battle, battle.phase === BattlePhase.AttackerTurn);
  }
}

function simpleBattlePhase(api: GameAPI, playerIdx: number, excludeLocations?: Set<string>): void {
  const visitedNodes = new Set<string>(excludeLocations);
  while (true) {
    const attackable = [...api.getAttackableLocations()].filter(n => !visitedNodes.has(n));
    if (attackable.length === 0) break;
    let attacked = false;
    for (const location of attackable) {
      visitedNodes.add(location);
      const state = api.getState();
      const owner = state.players[playerIdx].name;
      const candidates = armiesInRange(api, state, playerIdx, location);
      if (candidates.length === 0) continue;
      const qBefore = quantile(api, playerIdx);

      const c = api.fork();
      if (!c.startBattle(fullSelections(owner, candidates), location)) continue;
      simpleBattleLoop(c);
      // Rollout: move armies after battle to capture cleared nodes
      simpleArmyActions(c, playerIdx);
      if (quantile(c, playerIdx) <= qBefore) continue;

      const selected = [...candidates];
      for (const army of candidates) {
        if (selected.length <= 1) break;
        const without = selected.filter(a => a !== army);
        const cW = api.fork();
        if (!cW.startBattle(fullSelections(owner, without), location)) continue;
        simpleBattleLoop(cW);
        const qW = quantile(cW, playerIdx);
        if (qW >= qBefore) {
          const idx = selected.indexOf(army);
          if (idx >= 0) selected.splice(idx, 1);
        }
      }

      if (selected.length === 0) continue;
      if (!api.startBattle(fullSelections(owner, selected), location)) continue;
      simpleBattleLoop(api);
      attacked = true; break;
    }
    if (!attacked) break;
  }
}

function simpleRecruit(api: GameAPI, playerIdx: number): void {
  for (const location of api.getRecruitLocations()) {
    for (const unitType of UNIT_TYPES) {
      const state = api.getState();
      const cost = state.unitStatsMap[unitType].cost;
      const affordable = Math.floor(state.players[playerIdx].money / cost);
      if (affordable <= 0) continue;
      let bestFrac = 0, bestQ = quantile(api, playerIdx);
      for (const frac of RECRUIT_LEVELS) {
        const count = Math.round(frac * affordable);
        if (count <= 0 || !api.canBuy(unitType, count)) continue;
        const c = api.fork();
        c.recruit(unitType, location, count);
        const q = quantile(c, playerIdx);
        if (q > bestQ) { bestQ = q; bestFrac = frac; }
      }
      const count = Math.round(bestFrac * affordable);
      if (count > 0 && api.canBuy(unitType, count)) api.recruit(unitType, location, count);
    }
  }
}

/** Simulate the player's next turn on a fork: every player passes until the turn comes back, then a full simple turn. */
function simpleNextTurn(api: GameAPI, playerIdx: number): void {
  if (!api.endTurn()) return;
  if (!passTo(api, playerIdx)) return;
  simpleArmyActions(api, playerIdx);
  simpleBattlePhase(api, playerIdx);
  simpleArmyActions(api, playerIdx);
  simpleRecruit(api, playerIdx);
}

function simpleRollout(api: GameAPI, playerIdx: number, fromPhase: number, excludeBattleLocations?: Set<string>): void {
  if (fromPhase <= 1) simpleArmyActions(api, playerIdx);
  if (fromPhase <= 2) simpleBattlePhase(api, playerIdx, excludeBattleLocations);
  if (fromPhase <= 3) simpleArmyActions(api, playerIdx);
  if (fromPhase <= 4) simpleRecruit(api, playerIdx);
}

// ─── Lookahead generators ───

function* lookaheadArmyActions(api: GameAPI, playerIdx: number, samples: Sample[] | null, rolloutFrom = 2, model?: NNModel | null): Generator<void> {
  const walk = new ArmyWalk(api, playerIdx);
  for (let next = walk.next(); next; next = walk.next()) {
    const {armyId} = next;
    let army = next.army;
    for (;;) {
      const state = walk.state;
      const mask = computeActionTypeMask(api, armyId);
      const disbandMask = groupMask(disbandGroups(army.units));

      // EXIT baseline with rollout
      const cExit = api.fork();
      simpleRollout(cExit, playerIdx, rolloutFrom);
      let bestQ = quantile(cExit, playerIdx);
      let bestAction = ACTION_EXIT, bestTarget = 0;
      let bestFractions: Float32Array = new Float32Array(NUM_UNIT_GROUPS);

      const tryAction = (actionType: number, targetIdx: number, fractions: Float32Array): number => {
        const c = api.fork();
        if (!executeArmyAction(c, armyId, army, actionType, targetIdx, fractions)) return -Infinity;
        simpleRollout(c, playerIdx, rolloutFrom);
        return quantile(c, playerIdx);
      };

      if (mask[ACTION_MOVE] > 0) {
        for (const dest of api.getMovableLocations(armyId)) {
          const ni = NODE_ORDER.indexOf(dest);
          const groups = moveGroups(api.getMoveCandidates(armyId, dest));
          const result = searchCounts(groups, counts => tryAction(ACTION_MOVE, ni, countsToFractions(groups, counts)));
          if (result && result.q > bestQ) {
            bestQ = result.q; bestAction = ACTION_MOVE; bestTarget = ni; bestFractions = countsToFractions(groups, result.counts);
          }
        }
      }
      {
        const groups = disbandGroups(army.units);
        const result = searchCounts(groups, counts => tryAction(ACTION_DISBAND, 0, countsToFractions(groups, counts)));
        if (result && result.q > bestQ) {
          bestQ = result.q; bestAction = ACTION_DISBAND; bestFractions = countsToFractions(groups, result.counts);
        }
      }

      // Record
      if (samples) {
        const s = emptySample(playerIdx);
        s.state = encodeState(state, api, playerIdx, {type: "army", army, actionTypeMask: mask, disbandMask});
        s.actionTypeTarget = bestAction;
        s.actionTypeMask = new Float32Array(mask);
        if (bestAction === ACTION_DISBAND) { s.disbandFraction = bestFractions; s.disbandMask = disbandMask; }
        s.value = bestQ;
        samples.push(s);

        if (bestAction === ACTION_MOVE) {
          const legalMask = moveLegalMask(api, armyId);
          const ms = emptySample(playerIdx);
          ms.state = encodeState(state, api, playerIdx, {type: "moveTarget", army, legalMask});
          ms.moveTargetIdx = bestTarget;
          ms.moveMask = legalMask;
          ms.value = bestQ;
          samples.push(ms);

          const moveGroupMask = groupMask(moveGroups(api.getMoveCandidates(armyId, NODE_ORDER[bestTarget])));
          const cs = emptySample(playerIdx);
          cs.state = encodeState(state, api, playerIdx, {type: "moveCount", army, destinationIdx: bestTarget, groupMask: moveGroupMask});
          cs.moveFraction = bestFractions;
          cs.moveFractionMask = moveGroupMask;
          cs.value = bestQ;
          samples.push(cs);
        }
      }

      // Decide what to actually execute: NN (DAgger) or greedy
      let execAction = bestAction, execTarget = bestTarget, execFractions = bestFractions;
      if (model) {
        const pred = model.predict(encodeState(state, api, playerIdx, {type: "army", army, actionTypeMask: mask, disbandMask}));
        execAction = argmax(applyMaskAndSoftmax(pred.actionTypeLogits, mask));
        if (execAction === ACTION_MOVE) {
          const legalMask = moveLegalMask(api, armyId);
          const movePred = model.predict(encodeState(state, api, playerIdx, {type: "moveTarget", army, legalMask}));
          execTarget = argmax(applyMaskAndSoftmax(movePred.moveTargetLogits, legalMask));
          const moveGroupMask = groupMask(moveGroups(api.getMoveCandidates(armyId, NODE_ORDER[execTarget])));
          const countPred = model.predict(encodeState(state, api, playerIdx, {type: "moveCount", army, destinationIdx: execTarget, groupMask: moveGroupMask}));
          execFractions = expectedFractions(countPred.moveFractionLogits, NUM_MOVE_GROUPS);
        } else if (execAction === ACTION_DISBAND) {
          execFractions = expectedFractions(pred.disbandFractionLogits, NUM_DISBAND_GROUPS);
        }
      }

      if (execAction === ACTION_EXIT) break;
      if (!executeArmyAction(api, armyId, army, execAction, execTarget, execFractions)) break;
      yield; // UI update after each action
      const after = walk.acted();
      if (!after) break;
      army = after;
    }
  }
}

function* lookaheadBattleAllocate(
  api: GameAPI, battle: BattleData, isAttacker: boolean, playerIdx: number, samples: Sample[] | null,
): Generator<void> {
  const owner = (isAttacker ? battle.attackerPlayer : battle.defenderPlayer).name;
  const armies = isAttacker ? battle.attackerArmies : battle.defenderArmies;
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
      const overflowPct = futureNeeded < remaining
        ? Math.min(1, (remaining - futureNeeded) / killNeeded) : 0;

      let bestFrac = 0;
      let bestQ = -Infinity;
      for (const frac of [0, 0.25, 0.5, 0.75, 1.0]) {
        const unitCount = Math.round(frac * killNeeded);
        if (unitCount > remaining) continue;
        const c = api.fork();
        const trial = [...allocations];
        if (unitCount > 0) trial.push({target: armyIdOf(enemyOwner, target), count: unitCount});
        if (trial.length > 0) c.allocateAttack(armyId, trial);
        simpleBattleLoop(c);
        // A defender evaluates its own turn: the attacker and anyone between pass first
        if (isAttacker || passTo(c, playerIdx)) simpleRollout(c, playerIdx, 3);
        const q = quantile(c, playerIdx);
        if (q > bestQ) { bestQ = q; bestFrac = frac; }
      }

      if (samples) {
        const s = emptySample(playerIdx);
        s.state = encodeState(state, api, playerIdx, {
          type: "battleAllocate", army, remaining, enemyArmy: target,
          attackProgress: 1 - api.getRemainingAttacks(armyId) / current.maxArmyAttacks, isAttacker, unitsNeeded: killNeeded,
        });
        s.killFraction = bestFrac; s.killFracMask = 1;
        s.value = bestQ;
        samples.push(s);
      }

      if (bestFrac < overflowPct) {
        // Under-allocating — force full allocation to this and all remaining targets
        const send = Math.min(killNeeded, remaining);
        if (send > 0) {
          allocations.push({target: armyIdOf(enemyOwner, target), count: send});
          remaining -= send;
        }
        for (let tj = ti + 1; tj < targets.length; tj++) {
          if (remaining <= 0) break;
          const futureSend = Math.min(unitsNeeded(targets[tj]), remaining);
          if (futureSend > 0) {
            allocations.push({target: armyIdOf(enemyOwner, targets[tj]), count: futureSend});
            remaining -= futureSend;
          }
        }
        break;
      }

      const unitCount = Math.min(Math.round(bestFrac * killNeeded), remaining);
      if (unitCount > 0) {
        allocations.push({target: armyIdOf(enemyOwner, target), count: unitCount});
        remaining -= unitCount;
      }
    }
    // Always submit (empty = pass) so the army is marked acted; reject = broken invariant
    if (!api.allocateAttack(armyId, allocations)) {
      throw new Error(battleStuckReport(
        `lookaheadBattleAllocate: allocation rejected (${army.unitType}@${army.location})`, current));
    }
    yield;
  }
}

function* lookaheadBattleLoop(api: GameAPI, playerIdx: number, samples: Sample[] | null): Generator<void> {
  let iter = 0;
  for (;;) {
    const state = api.getState();
    const battle = state.currentBattle;
    if (!battle) return;
    if (++iter > battle.maxArmyAttacks * 4 + 8) throw new Error(battleStuckReport("lookaheadBattleLoop stuck", battle));
    if (battle.phase === BattlePhase.AttackerTurn) {
      if (battle.actedArmies.size === 0) {
        const cRet = api.fork();
        cRet.retreat();
        simpleRollout(cRet, playerIdx, 3);
        const qRetreat = quantile(cRet, playerIdx);

        const cFight = api.fork();
        simpleBattleLoop(cFight);
        simpleRollout(cFight, playerIdx, 3);
        const qFight = quantile(cFight, playerIdx);

        const doRetreat = qRetreat > qFight;

        if (samples) {
          const attackerOwner = battle.attackerPlayer.name;
          const remainingAttacks = Math.min(...battle.attackerArmies.map(army => api.getRemainingAttacks(armyIdOf(attackerOwner, army))));
          const s = emptySample(playerIdx);
          s.state = encodeState(state, api, playerIdx, {
            type: "battleRetreat",
            targetNodeIdx: NODE_ORDER.indexOf(battle.targetLocation),
            attackProgress: 1 - remainingAttacks / battle.maxArmyAttacks,
          });
          s.battleRetreat = doRetreat ? 1 : 0; s.retreatMask = 1;
          s.value = doRetreat ? qRetreat : qFight;
          samples.push(s);
        }

        if (doRetreat) {
          api.retreat();
          yield;
          return;
        }
      }
      yield* lookaheadBattleAllocate(api, battle, true, playerIdx, samples);
    } else {
      const defIdx = state.players.indexOf(battle.defenderPlayer);
      yield* lookaheadBattleAllocate(api, battle, false, defIdx, samples);
      yield;
    }
  }
}

/**
 * Simulate attacking `location` with the given attackers (all in-range armies
 * with all their attack-ready units when null), then rollout from phase 3.
 * Returns the resulting quantile, or null if the battle cannot start.
 */
function evalBattle(api: GameAPI, state: GameData, playerIdx: number, location: string, attackers: ArmySelection[] | null): number | null {
  const chosen = attackers ?? fullSelections(state.players[playerIdx].name, armiesInRange(api, state, playerIdx, location));
  if (chosen.length === 0) return null;
  const c = api.fork();
  if (!c.startBattle(chosen, location)) return null;
  simpleBattleLoop(c);
  simpleRollout(c, playerIdx, 3);
  return quantile(c, playerIdx);
}

/**
 * Additive army selection with per-step labels.
 * Each step: evaluate adding each remaining army at its best per-group
 * commitment (and "done" once one army is committed), label the argmax, record
 * one sample per option, and execute the greedy choice (or the NN's choice in
 * DAgger mode).
 */
function lookaheadSelectArmies(
  api: GameAPI, state: GameData, playerIdx: number, location: string, targetNodeIdx: number,
  candidates: ArmyData[], samples: Sample[] | null, model?: NNModel | null,
): ArmySelection[] {
  const owner = state.players[playerIdx].name;
  const selected = new Map<ArmyData, UnitData[]>();
  const remaining = [...candidates];

  while (remaining.length > 0) {
    const selectedPerType = perTypeState(selected);
    const remainingPerType = perTypeState(remaining.map(army => [army, attackReady(army)] as const));
    const doneIdx = selected.size > 0 ? remaining.length : -1;

    // Each candidate at its best commitment given the current selection
    const options = remaining.map((army): {q: number; fractions: Float32Array; commitMask: Float32Array} => {
      const groups = commitGroups(attackReady(army));
      const result = searchCounts(groups, counts => {
        const units = takeByFractions(groups, countsToFractions(groups, counts), false);
        return evalBattle(api, state, playerIdx, location, attackersOf(owner, [...selected, [army, units] as const])) ?? -Infinity;
      });
      return {
        q: result ? result.q : -Infinity,
        fractions: result ? countsToFractions(groups, result.counts) : new Float32Array(NUM_COMMIT_GROUPS),
        commitMask: groupMask(groups),
      };
    });
    const optionQ = options.map(o => o.q);
    if (doneIdx >= 0) {
      optionQ.push(evalBattle(api, state, playerIdx, location, attackersOf(owner, selected)) ?? -Infinity);
    }

    const labelPick = optionQ.reduce((best, q, i) => (q > optionQ[best] ? i : best), 0);

    const optionStates: Float32Array[] = remaining.map((army, i) => {
      const context: DecisionContext = {
        type: "battleSelect", army, targetNodeIdx, selectedPerType, remainingPerType, isDone: false, commitMask: options[i].commitMask,
      };
      return encodeState(state, api, playerIdx, context);
    });
    if (doneIdx >= 0) {
      optionStates.push(encodeState(state, api, playerIdx, {
        type: "battleSelect", army: null, targetNodeIdx, selectedPerType, remainingPerType, isDone: true,
        commitMask: new Float32Array(NUM_COMMIT_GROUPS),
      }));
    }

    if (samples) {
      for (let i = 0; i < optionStates.length; i++) {
        const ss = emptySample(playerIdx);
        ss.state = optionStates[i];
        ss.battleSelect = i === labelPick ? 1 : 0;
        ss.battleSelectMask = 1;
        if (i < remaining.length) {
          ss.commitFraction = options[i].fractions;
          ss.commitMask = options[i].commitMask;
        }
        ss.value = optionQ[labelPick];
        samples.push(ss);
      }
    }

    // DAgger: NN picks the option to execute and its commitment
    let execPick = labelPick;
    let execFractions: Float32Array | null = execPick < remaining.length ? options[execPick].fractions : null;
    if (model) {
      const preds = optionStates.map(st => model.predict(st));
      execPick = preds.reduce((best, p, i) => (p.battleSelect > preds[best].battleSelect ? i : best), 0);
      execFractions = execPick < remaining.length ? expectedFractions(preds[execPick].commitFractionLogits, NUM_COMMIT_GROUPS) : null;
    }

    if (execPick === doneIdx || execFractions === null) break;
    const army = remaining[execPick];
    selected.set(army, unitsToCommit(army, execFractions));
    remaining.splice(execPick, 1);
  }

  return attackersOf(owner, selected);
}

function* lookaheadBattlePhase(api: GameAPI, playerIdx: number, samples: Sample[] | null, model?: NNModel | null): Generator<void> {
  const fought = new Set<string>();

  // Terminates: every iteration either stops or adds a node to `fought`
  while (true) {
    const attackable = [...api.getAttackableLocations()].filter(n => !fought.has(n));
    if (attackable.length === 0) break;
    const state = api.getState();

    // Baseline: no battle, rollout from phase 3 (skip battles)
    const cSkip = api.fork();
    simpleRollout(cSkip, playerIdx, 3);
    const qSkip = quantile(cSkip, playerIdx);

    // Evaluate attacking each candidate node with all in-range armies
    const attackableMask = new Float32Array(NUM_NODES);
    let bestNode = -1;
    let bestQ = qSkip;
    for (const location of attackable) {
      const ni = NODE_ORDER.indexOf(location);
      if (ni < 0) continue;
      attackableMask[ni] = 1;
      const q = evalBattle(api, state, playerIdx, location, null);
      if (q !== null && q > bestQ) { bestQ = q; bestNode = ni; }
    }
    const labelChoice = bestNode >= 0 ? bestNode : BATTLE_TARGET_STOP;

    if (samples) {
      const ts = emptySample(playerIdx);
      ts.state = encodeState(state, api, playerIdx, {type: "battleTarget", attackableMask});
      ts.battleTargetIdx = labelChoice;
      const m = new Float32Array(BATTLE_TARGET_DIM);
      m.set(attackableMask);
      m[BATTLE_TARGET_STOP] = 1;
      ts.battleTargetMask = m;
      ts.value = bestQ;
      samples.push(ts);
    }

    // DAgger: NN picks the node to attack (or stop)
    let execChoice = labelChoice;
    if (model) {
      const m = new Float32Array(BATTLE_TARGET_DIM);
      m.set(attackableMask);
      m[BATTLE_TARGET_STOP] = 1;
      const pred = model.predict(encodeState(state, api, playerIdx, {type: "battleTarget", attackableMask}));
      execChoice = argmax(applyMaskAndSoftmax(pred.battleTargetLogits, m));
    }
    if (execChoice === BATTLE_TARGET_STOP) break;

    const location = NODE_ORDER[execChoice];
    fought.add(location);
    const candidates = armiesInRange(api, state, playerIdx, location);
    if (candidates.length === 0) continue;

    const attackers = lookaheadSelectArmies(api, state, playerIdx, location, execChoice, candidates, samples, model);
    if (attackers.length === 0) continue;

    if (!api.startBattle(attackers, location)) continue;
    // Run entire battle synchronously (no yield) to prevent UI from taking control
    const battleGen = lookaheadBattleLoop(api, playerIdx, samples);
    while (!battleGen.next().done) { /* drain */ }
    yield; // UI: battle resolved
  }
}

function* lookaheadRecruit(api: GameAPI, playerIdx: number, samples: Sample[] | null, model?: NNModel | null): Generator<void> {
  for (const location of api.getRecruitLocations()) {
    const locIdx = NODE_ORDER.indexOf(location);
    if (locIdx < 0) continue;
    for (const unitType of UNIT_TYPES) {
      const state = api.getState();
      const cost = state.unitStatsMap[unitType].cost;
      const affordable = Math.floor(state.players[playerIdx].money / cost);
      if (affordable <= 0) continue;
      let bestFrac = 0;
      const cBase = api.fork();
      simpleNextTurn(cBase, playerIdx);
      let bestQ = quantile(cBase, playerIdx);
      for (const frac of RECRUIT_LEVELS) {
        const count = Math.round(frac * affordable);
        if (count <= 0 || !api.canBuy(unitType, count)) continue;
        const c = api.fork();
        c.recruit(unitType, location, count);
        simpleNextTurn(c, playerIdx);
        const q = quantile(c, playerIdx);
        if (q > bestQ) { bestQ = q; bestFrac = frac; }
      }

      if (samples) {
        const s = emptySample(playerIdx);
        s.state = encodeState(state, api, playerIdx, {type: "recruit", locationIdx: locIdx, unitType, affordableCount: affordable});
        s.recruitFraction = bestFrac; s.recruitMask = 1;
        s.value = bestQ;
        samples.push(s);
      }

      // DAgger: NN decides actual fraction
      const execFrac = model
        ? expectedFractions(model.predict(encodeState(state, api, playerIdx, {type: "recruit", locationIdx: locIdx, unitType, affordableCount: affordable})).recruitFractionLogits, 1)[0]
        : bestFrac;
      const count = Math.round(execFrac * affordable);
      if (count > 0 && api.canBuy(unitType, count)) {
        if (api.recruit(unitType, location, count)) yield;
      }
    }
  }
}

// ─── Public API ───

/**
 * Generator: run one greedy turn, yielding after each game action.
 */
export function* greedyTurnSteps(api: GameAPI, samples?: Sample[]): Generator<void> {
  const state = api.getState();
  const playerIdx = state.currentPlayerIndex;
  if (state.players[playerIdx].defeated || api.getWinner() !== null) { api.endTurn(); return; }
  const s = samples ?? null;

  // A loaded save can carry an in-progress battle; finish it before the phases
  if (api.hasBattle()) {
    yield* lookaheadBattleLoop(api, playerIdx, s);
  }

  yield* lookaheadArmyActions(api, playerIdx, s);
  yield* lookaheadBattlePhase(api, playerIdx, s);
  yield* lookaheadArmyActions(api, playerIdx, s, 4);
  yield* lookaheadRecruit(api, playerIdx, s);

  api.endTurn();
}

/**
 * Synchronous wrapper: run one greedy turn to completion.
 */
export function greedyTurn(api: GameAPI, samples?: Sample[]): void {
  const gen = greedyTurnSteps(api, samples);
  while (!gen.next().done) { /* drain */ }
}

/**
 * DAgger turn: NN plays the game, greedy provides labels.
 * Records greedy's choices at NN's encountered states.
 */
export function daggerTurn(api: GameAPI, model: NNModel, samples: Sample[]): void {
  const state = api.getState();
  const playerIdx = state.currentPlayerIndex;
  if (state.players[playerIdx].defeated || api.getWinner() !== null) { api.endTurn(); return; }

  const gen = (function* () {
    yield* lookaheadArmyActions(api, playerIdx, samples, 2, model);
    yield* lookaheadBattlePhase(api, playerIdx, samples, model);
    yield* lookaheadArmyActions(api, playerIdx, samples, 4, model);
    yield* lookaheadRecruit(api, playerIdx, samples, model);
  })();
  while (!gen.next().done) { /* drain */ }

  api.endTurn();
}
