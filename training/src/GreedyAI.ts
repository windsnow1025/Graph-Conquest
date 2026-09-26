/**
 * Greedy AI: 1-step lookahead with greedy rollout to end of turn.
 *
 * Each decision point:
 *   1. List options
 *   2. For each option: clone → execute → greedy rollout rest of turn → quantile
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
import GameSystem from "../../src/lib/GameSystem";
import type Army from "../../src/lib/Army";
import type Unit from "../../src/lib/Unit";
import type Battle from "../../src/lib/Battle";
import {BattlePhase, BattleResult} from "../../src/lib/Battle";
import {calculateUnitsNeeded} from "../../src/lib/Combat";
import {NODE_ORDER, NUM_NODES, UNIT_TYPES} from "../../src/AI/nn/GameIndex";
import {encodeState, perTypeState} from "../../src/AI/nn/StateEncoder";
import type {DecisionContext} from "../../src/AI/nn/StateEncoder";
import {
  computeActionTypeMask, executeArmyAction, unitsToCommit,
  ACTION_EXIT, ACTION_MOVE, ACTION_DISBAND,
} from "../../src/AI/nn/ActionSpace";
import {
  NUM_UNIT_GROUPS, NUM_MOVE_GROUPS, NUM_DISBAND_GROUPS, NUM_COMMIT_GROUPS,
  moveGroups, commitGroups, disbandGroups, groupMask, countsToFractions, takeByFractions,
} from "../../src/AI/nn/UnitGroups";
import type {NNModel} from "../../src/AI/nn/NNModel";
import {applyMaskAndSoftmax, argmax, argmaxFractions, BATTLE_TARGET_DIM, BATTLE_TARGET_STOP} from "../../src/AI/nn/NNModel";
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
function moveLegalMask(game: GameSystem, army: Army): Float32Array {
  const mask = new Float32Array(NUM_NODES);
  for (const dest of army.getMovableLocations(game.gameMap, game.enemyLocations)) {
    const ni = NODE_ORDER.indexOf(dest);
    if (ni >= 0) mask[ni] = 1;
  }
  return mask;
}

// ─── Score & quantile ───

/** Execute one defender phase using greedy logic. */
export function greedyDefenderPhase(game: GameSystem, battle: Battle): void {
  simpleBattleAllocate(game, battle, false);
}

export function scorePlayer(game: GameSystem, playerIdx: number): number {
  const player = game.players[playerIdx];
  let nodeIncome = 0;
  for (const [node, owner] of game.nodeOwnership) {
    if (owner === player) nodeIncome += game.gameMap.getNodeData(node)?.income ?? 0;
  }
  const interest = Math.floor(player.money * game.interestRate);
  const upkeep = player.getUpkeep(game.upkeepRate);
  return nodeIncome + interest + upkeep;
}

function normalCDF(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804014327;
  const p = d * Math.exp(-x * x / 2) * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.8212560 + t * 1.3302744))));
  return x >= 0 ? 1 - p : p;
}

let _cachedTotalIncome = 0;
function quantile(game: GameSystem, playerIdx: number): number {
  if (_cachedTotalIncome === 0) {
    for (const [node] of game.gameMap.nodes) {
      _cachedTotalIncome += game.gameMap.getNodeData(node)?.income ?? 0;
    }
  }
  const scores = [0, 1, 2].map(i => scorePlayer(game, i));
  const mean = (scores[0] + scores[1] + scores[2]) / 3;
  return normalCDF((scores[playerIdx] - mean) / _cachedTotalIncome);
}

// ─── Clone ───

function cloneGame(game: GameSystem): GameSystem {
  return GameSystem.fromJSON(game.toJSON());
}

function cloneBattle(game: GameSystem): {game: GameSystem; battle: Battle} {
  const c = cloneGame(game);
  return {game: c, battle: c.currentBattle!};
}

/** Every army with all of its attack-ready units. */
function fullSelections(armies: Army[]): Map<Army, Unit[]> {
  return new Map(armies.map(army => [army, army.attackCandidates]));
}

/** The same selections on a clone: armies and units are matched by index. */
function cloneSelections(game: GameSystem, clone: GameSystem, playerIdx: number, selections: Map<Army, Unit[]>): Map<Army, Unit[]> {
  const player = game.players[playerIdx];
  const clonePlayer = clone.players[playerIdx];
  return new Map([...selections].map(([army, units]) => {
    const cloneArmy = clonePlayer.armies[player.armies.indexOf(army)];
    return [cloneArmy, units.map(unit => cloneArmy.units[army.units.indexOf(unit)])];
  }));
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
function searchCounts(groups: Unit[][], evaluate: (counts: number[]) => number): {counts: number[]; q: number} | null {
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

// Terminates without a step budget: every MOVE spends at least one move point
// and every DISBAND removes at least one unit.
function simpleArmyActions(game: GameSystem, playerIdx: number): void {
  const player = game.players[playerIdx];
  const processed = new Set<Army>();
  let ai = 0;
  while (ai < player.armies.length) {
    const army = player.armies[ai];
    if (processed.has(army)) { ai++; continue; }
    processed.add(army);
    for (;;) {
      const mask = computeActionTypeMask(game, army);
      const armyIdx = player.armies.indexOf(army);
      let bestAction = ACTION_EXIT, bestTarget = 0, bestFractions = FULL_FRACTIONS;
      let bestQ = quantile(game, playerIdx);

      const tryAction = (actionType: number, targetIdx: number, fractions: Float32Array): number => {
        const c = cloneGame(game);
        if (!executeArmyAction(c, c.players[playerIdx].armies[armyIdx], actionType, targetIdx, fractions)) return -Infinity;
        return quantile(c, playerIdx);
      };

      // Moving a part of the reachable units scores the same as moving all of them
      // (the score reads node ownership), so only the all-in move is tried
      if (mask[ACTION_MOVE] > 0) {
        for (const dest of army.getMovableLocations(game.gameMap, game.enemyLocations)) {
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
      if (!executeArmyAction(game, army, bestAction, bestTarget, bestFractions)) break;
      const newIdx = player.armies.indexOf(army); if (newIdx < 0) break; ai = newIdx;
    }
    const finalIdx = player.armies.indexOf(army);
    ai = finalIdx < 0 ? ai : finalIdx + 1;
  }
}

function simpleBattleAllocate(_game: GameSystem, battle: Battle, isAttacker: boolean): void {
  const armies = isAttacker ? battle.attackerArmies : battle.defenderArmies;
  for (const army of [...armies]) {
    if (!battle.canAct(army) || battle.result !== BattleResult.Ongoing) continue;
    const targets = battle.getTargetsInRange(army);
    let remaining = army.battleUnits.length;
    const allocations = new Map<Army, number>();
    for (const target of targets) {
      if (remaining <= 0) break;
      const killNeeded = calculateUnitsNeeded(army, target);
      const unitCount = Math.min(killNeeded, remaining);
      if (unitCount > 0) allocations.set(target, unitCount);
      remaining -= killNeeded;
    }
    // Always submit (empty = pass) so the army is marked acted; reject = broken invariant
    if (!battle.allocateAttack(army, allocations)) {
      throw new Error(battleStuckReport(
        `simpleBattleAllocate: allocation rejected (${army.unitType}@${army.location})`, battle));
    }
  }
}

function simpleBattleLoop(game: GameSystem, battle: Battle): void {
  const maxIter = battle.maxArmyAttacks * 4 + 8;
  let iter = 0;
  while (battle.result === BattleResult.Ongoing) {
    if (++iter > maxIter) {
      throw new Error(battleStuckReport("simpleBattleLoop stuck", battle));
    }
    if (battle.phase === BattlePhase.AttackerTurn) {
      simpleBattleAllocate(game, battle, true);
    } else {
      simpleBattleAllocate(game, battle, false);
    }
  }
}

function simpleBattlePhase(game: GameSystem, playerIdx: number, excludeLocations?: Set<string>): void {
  const visitedNodes = new Set<string>(excludeLocations);
  while (true) {
    const attackable = Array.from(game.attackableLocations).filter(n => !visitedNodes.has(n));
    if (attackable.length === 0) break;
    let attacked = false;
    for (const location of attackable) {
      visitedNodes.add(location);
      const candidates = game.getArmiesInRange(location);
      if (candidates.length === 0) continue;
      const qBefore = quantile(game, playerIdx);

      const c = cloneGame(game);
      const cCandidates = c.getArmiesInRange(location);
      if (cCandidates.length === 0) continue;
      const battle = c.startBattle(location, fullSelections(cCandidates));
      if (!battle) continue;
      simpleBattleLoop(c, battle);
      c.resolveBattle();
      // Rollout: move armies after battle to capture cleared nodes
      simpleArmyActions(c, playerIdx);
      if (quantile(c, playerIdx) <= qBefore) continue;

      const selected = [...candidates];
      for (const army of candidates) {
        if (selected.length <= 1) break;
        const without = selected.filter(a => a !== army);
        const cW = cloneGame(game);
        const bW = cW.startBattle(location, cloneSelections(game, cW, playerIdx, fullSelections(without)));
        if (!bW) continue;
        simpleBattleLoop(cW, bW);
        cW.resolveBattle();
        const qW = quantile(cW, playerIdx);
        if (qW >= qBefore) {
          const idx = selected.indexOf(army);
          if (idx >= 0) selected.splice(idx, 1);
        }
      }

      if (selected.length === 0) continue;
      const realBattle = game.startBattle(location, fullSelections(selected));
      if (!realBattle) continue;
      simpleBattleLoop(game, realBattle);
      game.resolveBattle();
      attacked = true; break;
    }
    if (!attacked) break;
  }
}

function simpleRecruit(game: GameSystem, playerIdx: number): void {
  const player = game.players[playerIdx];
  for (const location of game.recruitLocations) {
    for (const unitType of UNIT_TYPES) {
      const cost = game.unitStatsMap[unitType].cost;
      const affordable = Math.floor(player.money / cost);
      if (affordable <= 0) continue;
      let bestFrac = 0, bestQ = quantile(game, playerIdx);
      for (const frac of RECRUIT_LEVELS) {
        const count = Math.round(frac * affordable);
        if (count <= 0 || !player.canBuy(unitType, count)) continue;
        const c = cloneGame(game);
        c.recruitPlayerArmy(unitType, location, count);
        const q = quantile(c, playerIdx);
        if (q > bestQ) { bestQ = q; bestFrac = frac; }
      }
      const count = Math.round(bestFrac * affordable);
      if (count > 0 && player.canBuy(unitType, count)) game.recruitPlayerArmy(unitType, location, count);
    }
  }
}

/** Simulate the player's next turn on a clone (startTurn + full simple turn). */
function simpleNextTurn(game: GameSystem, playerIdx: number): void {
  const player = game.players[playerIdx];
  // startTurn logic for this player
  player.money = Math.floor(player.money * (1 + game.interestRate));
  let nodeIncome = 0;
  for (const [node, owner] of game.nodeOwnership) {
    if (owner === player) nodeIncome += game.gameMap.getNodeData(node)?.income ?? 0;
  }
  player.money += nodeIncome;
  player.money -= player.getUpkeep(game.upkeepRate);
  player.resetAllArmyTurns();
  // play a full simple turn
  simpleArmyActions(game, playerIdx);
  simpleBattlePhase(game, playerIdx);
  simpleArmyActions(game, playerIdx);
  simpleRecruit(game, playerIdx);
}

function simpleRollout(game: GameSystem, playerIdx: number, fromPhase: number, excludeBattleLocations?: Set<string>): void {
  if (fromPhase <= 1) simpleArmyActions(game, playerIdx);
  if (fromPhase <= 2) simpleBattlePhase(game, playerIdx, excludeBattleLocations);
  if (fromPhase <= 3) simpleArmyActions(game, playerIdx);
  if (fromPhase <= 4) simpleRecruit(game, playerIdx);
}

// ─── Lookahead generators ───

function* lookaheadArmyActions(game: GameSystem, playerIdx: number, samples: Sample[] | null, rolloutFrom = 2, model?: NNModel | null): Generator<void> {
  const player = game.players[playerIdx];
  const processed = new Set<Army>();
  let ai = 0;
  while (ai < player.armies.length) {
    const army = player.armies[ai];
    if (processed.has(army)) { ai++; continue; }
    processed.add(army);
    for (;;) {
      const mask = computeActionTypeMask(game, army);
      const disbandMask = groupMask(disbandGroups(army.units));
      const armyIdx = player.armies.indexOf(army);

      // EXIT baseline with rollout
      const cExit = cloneGame(game);
      simpleRollout(cExit, playerIdx, rolloutFrom);
      let bestQ = quantile(cExit, playerIdx);
      let bestAction = ACTION_EXIT, bestTarget = 0;
      let bestFractions: Float32Array = new Float32Array(NUM_UNIT_GROUPS);

      const tryAction = (actionType: number, targetIdx: number, fractions: Float32Array): number => {
        const c = cloneGame(game);
        if (!executeArmyAction(c, c.players[playerIdx].armies[armyIdx], actionType, targetIdx, fractions)) return -Infinity;
        simpleRollout(c, playerIdx, rolloutFrom);
        return quantile(c, playerIdx);
      };

      if (mask[ACTION_MOVE] > 0) {
        for (const dest of army.getMovableLocations(game.gameMap, game.enemyLocations)) {
          const ni = NODE_ORDER.indexOf(dest);
          const groups = moveGroups(army.getMoveCandidates(dest, game.gameMap, game.enemyLocations));
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
        s.state = encodeState(game, playerIdx, {type: "army", army, actionTypeMask: mask, disbandMask});
        s.actionTypeTarget = bestAction;
        s.actionTypeMask = new Float32Array(mask);
        if (bestAction === ACTION_DISBAND) { s.disbandFraction = bestFractions; s.disbandMask = disbandMask; }
        s.value = bestQ;
        samples.push(s);

        if (bestAction === ACTION_MOVE) {
          const legalMask = moveLegalMask(game, army);
          const ms = emptySample(playerIdx);
          ms.state = encodeState(game, playerIdx, {type: "moveTarget", army, legalMask});
          ms.moveTargetIdx = bestTarget;
          ms.moveMask = legalMask;
          ms.value = bestQ;
          samples.push(ms);

          const moveGroupMask = groupMask(moveGroups(army.getMoveCandidates(NODE_ORDER[bestTarget], game.gameMap, game.enemyLocations)));
          const cs = emptySample(playerIdx);
          cs.state = encodeState(game, playerIdx, {type: "moveCount", army, destinationIdx: bestTarget, groupMask: moveGroupMask});
          cs.moveFraction = bestFractions;
          cs.moveFractionMask = moveGroupMask;
          cs.value = bestQ;
          samples.push(cs);
        }
      }

      // Decide what to actually execute: NN (DAgger) or greedy
      let execAction = bestAction, execTarget = bestTarget, execFractions = bestFractions;
      if (model) {
        const pred = model.predict(encodeState(game, playerIdx, {type: "army", army, actionTypeMask: mask, disbandMask}));
        execAction = argmax(applyMaskAndSoftmax(pred.actionTypeLogits, mask));
        if (execAction === ACTION_MOVE) {
          const legalMask = moveLegalMask(game, army);
          const movePred = model.predict(encodeState(game, playerIdx, {type: "moveTarget", army, legalMask}));
          execTarget = argmax(applyMaskAndSoftmax(movePred.moveTargetLogits, legalMask));
          const moveGroupMask = groupMask(moveGroups(army.getMoveCandidates(NODE_ORDER[execTarget], game.gameMap, game.enemyLocations)));
          const countPred = model.predict(encodeState(game, playerIdx, {type: "moveCount", army, destinationIdx: execTarget, groupMask: moveGroupMask}));
          execFractions = argmaxFractions(countPred.moveFractionLogits, NUM_MOVE_GROUPS);
        } else if (execAction === ACTION_DISBAND) {
          execFractions = argmaxFractions(pred.disbandFractionLogits, NUM_DISBAND_GROUPS);
        }
      }

      if (execAction === ACTION_EXIT) break;
      if (!executeArmyAction(game, army, execAction, execTarget, execFractions)) break;
      yield; // UI update after each action
      const newIdx = player.armies.indexOf(army); if (newIdx < 0) break; ai = newIdx;
    }
    const finalIdx = player.armies.indexOf(army);
    ai = finalIdx < 0 ? ai : finalIdx + 1;
  }
}

function* lookaheadBattleAllocate(
  game: GameSystem, battle: Battle, isAttacker: boolean, playerIdx: number, samples: Sample[] | null,
): Generator<void> {
  const armies = isAttacker ? battle.attackerArmies : battle.defenderArmies;
  for (const army of [...armies]) {
    if (!battle.canAct(army) || battle.result !== BattleResult.Ongoing) continue;
    const targets = battle.getTargetsInRange(army);
    let remaining = army.battleUnits.length;
    const allocations = new Map<Army, number>();

    for (let ti = 0; ti < targets.length; ti++) {
      if (remaining <= 0) break;
      const target = targets[ti];
      const killNeeded = calculateUnitsNeeded(army, target);
      const futureNeeded = targets.slice(ti + 1)
        .reduce((sum, t) => sum + calculateUnitsNeeded(army, t), 0);
      const overflowPct = futureNeeded < remaining
        ? Math.min(1, (remaining - futureNeeded) / killNeeded) : 0;

      let bestFrac = 0;
      let bestQ = -Infinity;
      for (const frac of [0, 0.25, 0.5, 0.75, 1.0]) {
        const unitCount = Math.round(frac * killNeeded);
        if (unitCount > remaining) continue;
        const {game: cEval, battle: bEval} = cloneBattle(game);
        const cArmy = isAttacker
          ? bEval.attackerArmies[battle.attackerArmies.indexOf(army)]
          : bEval.defenderArmies[battle.defenderArmies.indexOf(army)];
        if (!cArmy) continue;
        const priorAllocs = new Map<Army, number>();
        for (const [prior, count] of allocations) {
          const cPrior = isAttacker
            ? bEval.defenderArmies[battle.defenderArmies.indexOf(prior)]
            : bEval.attackerArmies[battle.attackerArmies.indexOf(prior)];
          if (cPrior) priorAllocs.set(cPrior, count);
        }
        if (unitCount > 0) {
          const cTarget = isAttacker
            ? bEval.defenderArmies[battle.defenderArmies.indexOf(target)]
            : bEval.attackerArmies[battle.attackerArmies.indexOf(target)];
          if (cTarget) priorAllocs.set(cTarget, unitCount);
        }
        if (priorAllocs.size > 0) bEval.allocateAttack(cArmy, priorAllocs);
        simpleBattleLoop(cEval, bEval);
        cEval.resolveBattle();
        // For a defender evaluation the clone's current player is still the attacker,
        // whose GameSystem methods would act for the wrong side during the rollout
        cEval.currentPlayerIndex = playerIdx;
        simpleRollout(cEval, playerIdx, 3);
        const q = quantile(cEval, playerIdx);
        if (q > bestQ) { bestQ = q; bestFrac = frac; }
      }

      if (samples) {
        const s = emptySample(playerIdx);
        s.state = encodeState(game, playerIdx, {
          type: "battleAllocate", army, remaining, enemyArmy: target,
          attackProgress: 1 - battle.getRemainingAttacks(army) / battle.maxArmyAttacks, isAttacker, unitsNeeded: killNeeded,
        });
        s.killFraction = bestFrac; s.killFracMask = 1;
        s.value = bestQ;
        samples.push(s);
      }

      if (bestFrac < overflowPct) {
        // Under-allocating — force full allocation to this and all remaining targets
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

      const unitCount = Math.min(Math.round(bestFrac * killNeeded), remaining);
      if (unitCount > 0) {
        allocations.set(target, unitCount);
        remaining -= unitCount;
      }
    }
    // Always submit (empty = pass) so the army is marked acted; reject = broken invariant
    if (!battle.allocateAttack(army, allocations)) {
      throw new Error(battleStuckReport(
        `lookaheadBattleAllocate: allocation rejected (${army.unitType}@${army.location})`, battle));
    }
    yield;
  }
}

function* lookaheadBattleLoop(
  game: GameSystem, battle: Battle, playerIdx: number, samples: Sample[] | null,
): Generator<void> {
  const maxIter = battle.maxArmyAttacks * 4 + 8;
  let iter = 0;
  while (battle.result === BattleResult.Ongoing) {
    if (++iter > maxIter) throw new Error(battleStuckReport("lookaheadBattleLoop stuck", battle));
    if (battle.phase === BattlePhase.AttackerTurn) {
      if (battle.actedArmies.size === 0) {
        const {game: cRet, battle: bRet} = cloneBattle(game);
        bRet.retreat();
        cRet.resolveBattle();
        simpleRollout(cRet, playerIdx, 3);
        const qRetreat = quantile(cRet, playerIdx);

        const {game: cFight, battle: bFight} = cloneBattle(game);
        simpleBattleLoop(cFight, bFight);
        cFight.resolveBattle();
        simpleRollout(cFight, playerIdx, 3);
        const qFight = quantile(cFight, playerIdx);

        const doRetreat = qRetreat > qFight;

        if (samples) {
          const remainingAttacks = Math.min(...battle.attackerArmies.map((a) => battle.getRemainingAttacks(a)));
          const s = emptySample(playerIdx);
          s.state = encodeState(game, playerIdx, {
            type: "battleRetreat",
            targetNodeIdx: NODE_ORDER.indexOf(battle.targetLocation),
            attackProgress: 1 - remainingAttacks / battle.maxArmyAttacks,
          });
          s.battleRetreat = doRetreat ? 1 : 0; s.retreatMask = 1;
          s.value = doRetreat ? qRetreat : qFight;
          samples.push(s);
        }

        if (doRetreat) {
          battle.retreat();
          yield;
          return;
        }
      }
      yield* lookaheadBattleAllocate(game, battle, true, playerIdx, samples);
    } else {
      const defIdx = game.players.indexOf(battle.defenderPlayer);
      yield* lookaheadBattleAllocate(game, battle, false, defIdx, samples);
      yield;
    }
  }
}

/**
 * Simulate attacking `location` with the given selections (all in-range armies
 * with all their attack-ready units when null), then rollout from phase 3.
 * Returns the resulting quantile, or null if the battle cannot start.
 */
function evalBattle(game: GameSystem, playerIdx: number, location: string, selections: Map<Army, Unit[]> | null): number | null {
  const c = cloneGame(game);
  const cSelections = selections === null
    ? fullSelections(c.getArmiesInRange(location))
    : cloneSelections(game, c, playerIdx, selections);
  if (cSelections.size === 0) return null;
  const b = c.startBattle(location, cSelections);
  if (!b) return null;
  simpleBattleLoop(c, b);
  c.resolveBattle();
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
  game: GameSystem, playerIdx: number, location: string, targetNodeIdx: number,
  candidates: Army[], samples: Sample[] | null, model?: NNModel | null,
): Map<Army, Unit[]> {
  const selected = new Map<Army, Unit[]>();
  const remaining = [...candidates];

  while (remaining.length > 0) {
    const selectedPerType = perTypeState(selected);
    const remainingPerType = perTypeState(remaining.map(army => [army, army.attackCandidates] as const));
    const doneIdx = selected.size > 0 ? remaining.length : -1;

    // Each candidate at its best commitment given the current selection
    const options = remaining.map((army): {q: number; fractions: Float32Array; commitMask: Float32Array} => {
      const groups = commitGroups(army.attackCandidates);
      const result = searchCounts(groups, counts => {
        const units = takeByFractions(groups, countsToFractions(groups, counts), false);
        return evalBattle(game, playerIdx, location, new Map([...selected, [army, units] as [Army, Unit[]]])) ?? -Infinity;
      });
      return {
        q: result ? result.q : -Infinity,
        fractions: result ? countsToFractions(groups, result.counts) : new Float32Array(NUM_COMMIT_GROUPS),
        commitMask: groupMask(groups),
      };
    });
    const optionQ = options.map(o => o.q);
    if (doneIdx >= 0) {
      optionQ.push(evalBattle(game, playerIdx, location, selected) ?? -Infinity);
    }

    const labelPick = optionQ.reduce((best, q, i) => (q > optionQ[best] ? i : best), 0);

    const optionStates: Float32Array[] = remaining.map((army, i) => {
      const context: DecisionContext = {
        type: "battleSelect", army, targetNodeIdx, selectedPerType, remainingPerType, isDone: false, commitMask: options[i].commitMask,
      };
      return encodeState(game, playerIdx, context);
    });
    if (doneIdx >= 0) {
      optionStates.push(encodeState(game, playerIdx, {
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
      execFractions = execPick < remaining.length ? argmaxFractions(preds[execPick].commitFractionLogits, NUM_COMMIT_GROUPS) : null;
    }

    if (execPick === doneIdx || execFractions === null) break;
    const army = remaining[execPick];
    selected.set(army, unitsToCommit(army, execFractions));
    remaining.splice(execPick, 1);
  }

  return selected;
}

function* lookaheadBattlePhase(game: GameSystem, playerIdx: number, samples: Sample[] | null, model?: NNModel | null): Generator<void> {
  const fought = new Set<string>();

  // Terminates: every iteration either stops or adds a node to `fought`
  while (true) {
    const attackable = Array.from(game.attackableLocations).filter(n => !fought.has(n));
    if (attackable.length === 0) break;

    // Baseline: no battle, rollout from phase 3 (skip battles)
    const cSkip = cloneGame(game);
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
      const q = evalBattle(game, playerIdx, location, null);
      if (q !== null && q > bestQ) { bestQ = q; bestNode = ni; }
    }
    const labelChoice = bestNode >= 0 ? bestNode : BATTLE_TARGET_STOP;

    if (samples) {
      const ts = emptySample(playerIdx);
      ts.state = encodeState(game, playerIdx, {type: "battleTarget", attackableMask});
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
      const pred = model.predict(encodeState(game, playerIdx, {type: "battleTarget", attackableMask}));
      execChoice = argmax(applyMaskAndSoftmax(pred.battleTargetLogits, m));
    }
    if (execChoice === BATTLE_TARGET_STOP) break;

    const location = NODE_ORDER[execChoice];
    fought.add(location);
    const candidates = game.getArmiesInRange(location);
    if (candidates.length === 0) continue;

    const selected = lookaheadSelectArmies(game, playerIdx, location, execChoice, candidates, samples, model);
    if (selected.size === 0) continue;

    const realBattle = game.startBattle(location, selected);
    if (!realBattle) continue;
    // Run entire battle synchronously (no yield) to prevent UI from taking control
    const battleGen = lookaheadBattleLoop(game, realBattle, playerIdx, samples);
    while (!battleGen.next().done) { /* drain */ }
    game.resolveBattle();
    yield; // UI: battle resolved
  }
}

function* lookaheadRecruit(game: GameSystem, playerIdx: number, samples: Sample[] | null, model?: NNModel | null): Generator<void> {
  const player = game.currentPlayer;
  for (const location of game.recruitLocations) {
    const locIdx = NODE_ORDER.indexOf(location);
    if (locIdx < 0) continue;
    for (const unitType of UNIT_TYPES) {
      const cost = game.unitStatsMap[unitType].cost;
      const affordable = Math.floor(player.money / cost);
      if (affordable <= 0) continue;
      let bestFrac = 0;
      const cBase = cloneGame(game);
      simpleNextTurn(cBase, playerIdx);
      let bestQ = quantile(cBase, playerIdx);
      for (const frac of RECRUIT_LEVELS) {
        const count = Math.round(frac * affordable);
        if (count <= 0 || !player.canBuy(unitType, count)) continue;
        const c = cloneGame(game);
        c.recruitPlayerArmy(unitType, location, count);
        simpleNextTurn(c, playerIdx);
        const q = quantile(c, playerIdx);
        if (q > bestQ) { bestQ = q; bestFrac = frac; }
      }

      if (samples) {
        const s = emptySample(playerIdx);
        s.state = encodeState(game, playerIdx, {type: "recruit", locationIdx: locIdx, unitType, affordableCount: affordable});
        s.recruitFraction = bestFrac; s.recruitMask = 1;
        s.value = bestQ;
        samples.push(s);
      }

      // DAgger: NN decides actual fraction
      const execFrac = model
        ? argmaxFractions(model.predict(encodeState(game, playerIdx, {type: "recruit", locationIdx: locIdx, unitType, affordableCount: affordable})).recruitFractionLogits, 1)[0]
        : bestFrac;
      const count = Math.round(execFrac * affordable);
      if (count > 0 && player.canBuy(unitType, count)) {
        if (game.recruitPlayerArmy(unitType, location, count)) yield;
      }
    }
  }
}

// ─── Public API ───

/**
 * Generator: run one greedy turn, yielding after each game action.
 */
export function* greedyTurnSteps(game: GameSystem, samples?: Sample[]): Generator<void> {
  const player = game.currentPlayer;
  const playerIdx = game.currentPlayerIndex;
  if (player.defeated || game.winner) { game.endTurn(); return; }
  const s = samples ?? null;

  // A loaded save can carry an in-progress battle; finish it before the phases
  if (game.currentBattle) {
    yield* lookaheadBattleLoop(game, game.currentBattle, playerIdx, s);
    game.resolveBattle();
  }

  yield* lookaheadArmyActions(game, playerIdx, s);
  yield* lookaheadBattlePhase(game, playerIdx, s);
  yield* lookaheadArmyActions(game, playerIdx, s, 4);
  yield* lookaheadRecruit(game, playerIdx, s);

  game.endTurn();
}

/**
 * Synchronous wrapper: run one greedy turn to completion.
 */
export function greedyTurn(game: GameSystem, samples?: Sample[]): void {
  const gen = greedyTurnSteps(game, samples);
  while (!gen.next().done) { /* drain */ }
}

/**
 * DAgger turn: NN plays the game, greedy provides labels.
 * Records greedy's choices at NN's encountered states.
 */
export function daggerTurn(game: GameSystem, model: NNModel, samples: Sample[]): void {
  const player = game.currentPlayer;
  const playerIdx = game.currentPlayerIndex;
  if (player.defeated || game.winner) { game.endTurn(); return; }

  const gen = (function* () {
    yield* lookaheadArmyActions(game, playerIdx, samples, 2, model);
    yield* lookaheadBattlePhase(game, playerIdx, samples, model);
    yield* lookaheadArmyActions(game, playerIdx, samples, 4, model);
    yield* lookaheadRecruit(game, playerIdx, samples, model);
  })();
  while (!gen.next().done) { /* drain */ }

  game.endTurn();
}
