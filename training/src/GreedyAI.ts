/**
 * Greedy AI: 1-step lookahead with greedy rollout to end of turn.
 *
 * Each decision point:
 *   1. List options
 *   2. For each option: clone → execute → greedy rollout rest of turn → quantile
 *   3. Pick highest quantile
 *
 * All lookahead functions are generators that yield after each real game action,
 * allowing the UI to render intermediate states.
 *
 * Pass samples array to greedyTurn to record decisions for imitation learning.
 */
import GameSystem from "../../src/lib/GameSystem";
import type Army from "../../src/lib/Army";
import Battle, {BattlePhase, BattleResult} from "../../src/lib/Battle";
import {calculateUnitsNeeded} from "../../src/lib/Combat";
import {encodeState, NODE_ORDER, NUM_NODES, UNIT_TYPES} from "../../src/AI/nn/StateEncoder";
import {
  computeActionTypeMask, executeArmyAction,
  ACTION_EXIT, ACTION_MOVE, ACTION_SPLIT, ACTION_DISBAND, NUM_ACTION_TYPES,
} from "../../src/AI/nn/ActionSpace";
import type {NNModel} from "../../src/AI/nn/NNModel";
import {applyMaskAndSoftmax, argmax, BATTLE_TARGET_DIM, BATTLE_TARGET_STOP} from "../../src/AI/nn/NNModel";
import {battleStuckReport} from "../../src/AI/battleReport";
import type {Sample} from "./SampleTypes";
import {emptySample} from "./SampleTypes";

const MAX_STEPS_PER_ARMY = 10;

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
  const defIdx = game.players.indexOf(battle.defenderPlayer);
  if (defIdx < 0 || battle.defenderPlayer.defeated) {
    battle.executeNeutralDefenderTurn();
    return;
  }
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

// ─── Selection state helpers ───

function computeSelState(selected: Army[]): number[] {
  const r = new Array(6).fill(0);
  const hpSums = [0, 0, 0];
  for (const a of selected) {
    const t = (UNIT_TYPES as readonly string[]).indexOf(a.unitType); if (t < 0) continue;
    r[t * 2] += a.units.length;
    hpSums[t] += a.units.reduce((s, u) => s + u.currentHealth / u.health, 0);
  }
  for (let t = 0; t < 3; t++) { r[t * 2 + 1] = r[t * 2] > 0 ? hpSums[t] / r[t * 2] : 0; }
  return r;
}

// ─── Simple greedy (no rollout, used inside rollout) ───

// No lineage budgets needed here: simple greedy only takes an action when it
// strictly raises the quantile, and SPLIT/MERGE leave every score unchanged,
// so it can never enter the SPLIT/MERGE object-mint loop.
function simpleArmyActions(game: GameSystem, playerIdx: number): void {
  const player = game.players[playerIdx];
  const processed = new Set<object>();
  let ai = 0;
  while (ai < player.armies.length) {
    const army = player.armies[ai];
    if (processed.has(army)) { ai++; continue; }
    processed.add(army);
    for (let step = 0; step < MAX_STEPS_PER_ARMY; step++) {
      if (army.units.length === 0) break;
      const mask = computeActionTypeMask(game, player, army);
      const armyIdx = player.armies.indexOf(army);
      let bestAction = ACTION_EXIT, bestTarget = 0, bestFraction = 0;
      let bestQ = quantile(game, playerIdx);
      for (let a = 0; a < NUM_ACTION_TYPES; a++) {
        if (mask[a] === 0 || a === ACTION_EXIT) continue;
        if (a === ACTION_MOVE) {
          for (const dest of army.getMovableLocations(game.gameMap, game.enemyLocations)) {
            const ni = NODE_ORDER.indexOf(dest); if (ni < 0) continue;
            const c = cloneGame(game); const ca = c.players[playerIdx].armies[armyIdx]; if (!ca) continue;
            executeArmyAction(c, c.players[playerIdx], ca, a, ni, 0);
            const q = quantile(c, playerIdx);
            if (q > bestQ) { bestQ = q; bestAction = a; bestTarget = ni; }
          }
        } else if (a === ACTION_SPLIT) {
          for (const f of [0.2, 0.5]) { const c = cloneGame(game); const ca = c.players[playerIdx].armies[armyIdx]; if (!ca) continue; executeArmyAction(c, c.players[playerIdx], ca, a, 0, f); const q = quantile(c, playerIdx); if (q > bestQ) { bestQ = q; bestAction = a; bestFraction = f; } }
        } else if (a === ACTION_DISBAND) {
          for (const f of [0.3, 0.5, 1.0]) { const c = cloneGame(game); const ca = c.players[playerIdx].armies[armyIdx]; if (!ca) continue; executeArmyAction(c, c.players[playerIdx], ca, a, 0, f); const q = quantile(c, playerIdx); if (q > bestQ) { bestQ = q; bestAction = a; bestFraction = f; } }
        } else {
          const c = cloneGame(game); const ca = c.players[playerIdx].armies[armyIdx]; if (!ca) continue;
          executeArmyAction(c, c.players[playerIdx], ca, a, 0, 0);
          const q = quantile(c, playerIdx); if (q > bestQ) { bestQ = q; bestAction = a; }
        }
      }

      if (bestAction === ACTION_EXIT) break;
      const fraction = bestAction === ACTION_SPLIT || bestAction === ACTION_DISBAND ? bestFraction : 0;
      if (!executeArmyAction(game, player, army, bestAction, bestTarget, fraction)) break;
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
    if (targets.length === 0) continue;
    let remaining = army.units.length;
    const allocations: Array<{target: Army; unitCount: number}> = [];
    for (const target of targets) {
      if (remaining <= 0) break;
      const killNeeded = calculateUnitsNeeded(army, target);
      const unitCount = Math.min(killNeeded, remaining);
      if (unitCount > 0) allocations.push({target, unitCount});
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
  const maxIter = battle.maxRounds * 4 + 8;
  let iter = 0;
  while (battle.result === BattleResult.Ongoing) {
    if (++iter > maxIter) {
      throw new Error(battleStuckReport("simpleBattleLoop stuck", battle));
    }
    if (battle.phase === BattlePhase.AttackerTurn) {
      simpleBattleAllocate(game, battle, true);
    } else {
      const defIdx = game.players.indexOf(battle.defenderPlayer);
      if (defIdx < 0 || battle.defenderPlayer.defeated) battle.executeNeutralDefenderTurn();
      else simpleBattleAllocate(game, battle, false);
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
      const battle = c.startBattle(location, cCandidates);
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
        const cWCands = without.map(a => {
          const idx = game.currentPlayer.armies.indexOf(a);
          return cW.players[playerIdx].armies[idx];
        }).filter(Boolean);
        if (cWCands.length === 0) continue;
        const bW = cW.startBattle(location, cWCands);
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
      const realBattle = game.startBattle(location, selected);
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
      for (const frac of [0.2, 0.5, 0.8, 1.0]) {
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
  const processed = new Set<object>();
  // Step budgets are conserved across split/merge lineages: a split child
  // inherits the parent's remaining steps and merges never refund, so the
  // phase total is bounded even when the executed policy splits endlessly.
  const budgets = new Map<Army, number>();
  for (const a of player.armies) budgets.set(a, MAX_STEPS_PER_ARMY);
  let ai = 0;
  while (ai < player.armies.length) {
    const army = player.armies[ai];
    if (processed.has(army)) { ai++; continue; }
    processed.add(army);
    let remaining = budgets.get(army) ?? MAX_STEPS_PER_ARMY;
    while (remaining > 0) {
      if (army.units.length === 0) break;
      remaining--;
      const mask = computeActionTypeMask(game, player, army);
      const armyIdx = player.armies.indexOf(army);

      // EXIT baseline with rollout
      const cExit = cloneGame(game);
      simpleRollout(cExit, playerIdx, rolloutFrom);
      let bestQ = quantile(cExit, playerIdx);
      let bestAction = ACTION_EXIT, bestTarget = 0, bestFraction = 0;

      const tryOption = (a: number, targetIdx: number, frac: number) => {
        const c = cloneGame(game);
        const ca = c.players[playerIdx].armies[armyIdx];
        if (!ca) return;
        if (!executeArmyAction(c, c.players[playerIdx], ca, a, targetIdx, frac)) return;
        simpleRollout(c, playerIdx, rolloutFrom);
        const q = quantile(c, playerIdx);
        if (q > bestQ) { bestQ = q; bestAction = a; bestTarget = targetIdx; bestFraction = frac; }
      };

      for (let a = 0; a < NUM_ACTION_TYPES; a++) {
        if (mask[a] === 0 || a === ACTION_EXIT) continue;
        if (a === ACTION_MOVE) {
          for (const dest of army.getMovableLocations(game.gameMap, game.enemyLocations)) {
            const ni = NODE_ORDER.indexOf(dest); if (ni >= 0) tryOption(a, ni, 0);
          }
        } else if (a === ACTION_SPLIT) {
          for (const f of [0.2, 0.5]) tryOption(a, 0, f);
        } else if (a === ACTION_DISBAND) {
          for (const f of [0.3, 0.5, 1.0]) tryOption(a, 0, f);
        } else {
          tryOption(a, 0, 0);
        }
      }

      // Record
      if (samples) {
        const s = emptySample(playerIdx);
        s.state = encodeState(game, playerIdx, {type: "army", army, actionTypeMask: mask});
        s.actionTypeTarget = bestAction;
        s.actionTypeMask = new Float32Array(mask);
        if (bestAction === ACTION_SPLIT) { s.splitFraction = bestFraction; s.splitMask = 1; }
        if (bestAction === ACTION_DISBAND) { s.disbandFraction = bestFraction; s.disbandMask = 1; }
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
        }
      }

      // Decide what to actually execute: NN (DAgger) or greedy
      let execAction = bestAction, execTarget = bestTarget, execFraction = bestFraction;
      if (model) {
        const pred = model.predict(encodeState(game, playerIdx, {type: "army", army, actionTypeMask: mask}));
        const probs = applyMaskAndSoftmax(pred.actionTypeLogits, mask);
        execAction = argmax(probs);
        if (execAction === ACTION_MOVE) {
          const legalMask = moveLegalMask(game, army);
          const movePred = model.predict(encodeState(game, playerIdx, {type: "moveTarget", army, legalMask}));
          execTarget = argmax(applyMaskAndSoftmax(movePred.moveTargetLogits, legalMask));
        }
        execFraction = execAction === ACTION_SPLIT ? pred.splitFraction : pred.disbandFraction;
      }

      if (execAction === ACTION_EXIT) break;
      const fraction = execAction === ACTION_SPLIT || execAction === ACTION_DISBAND ? execFraction : 0;
      const armiesBefore = execAction === ACTION_SPLIT ? new Set(player.armies) : null;
      if (!executeArmyAction(game, player, army, execAction, execTarget, fraction)) break;
      if (armiesBefore) {
        for (const a of player.armies) {
          if (!armiesBefore.has(a)) budgets.set(a, remaining); // split child inherits remaining budget
        }
      }
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
    if (targets.length === 0) continue;
    let remaining = army.units.length;
    const allocations: Array<{target: Army; unitCount: number}> = [];

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
        const priorAllocs = allocations.map(al => ({
          target: isAttacker
            ? bEval.defenderArmies[battle.defenderArmies.indexOf(al.target)]
            : bEval.attackerArmies[battle.attackerArmies.indexOf(al.target)],
          unitCount: al.unitCount,
        })).filter(al => al.target);
        if (unitCount > 0) {
          const cTarget = isAttacker
            ? bEval.defenderArmies[battle.defenderArmies.indexOf(target)]
            : bEval.attackerArmies[battle.attackerArmies.indexOf(target)];
          if (cTarget) priorAllocs.push({target: cTarget, unitCount});
        }
        if (priorAllocs.length > 0) bEval.allocateAttack(cArmy, priorAllocs);
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
          roundProgress: battle.round / battle.maxRounds, isAttacker, unitsNeeded: killNeeded,
        });
        s.killFraction = bestFrac; s.killFracMask = 1;
        s.value = bestQ;
        samples.push(s);
      }

      if (bestFrac < overflowPct) {
        // Under-allocating — force full allocation to this and all remaining targets
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

      const unitCount = Math.min(Math.round(bestFrac * killNeeded), remaining);
      if (unitCount > 0) {
        allocations.push({target, unitCount});
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
  const maxIter = battle.maxRounds * 4 + 8;
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
          const s = emptySample(playerIdx);
          s.state = encodeState(game, playerIdx, {
            type: "battleRetreat",
            targetNodeIdx: NODE_ORDER.indexOf(battle.targetLocation),
            roundProgress: battle.round / battle.maxRounds,
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
      if (defIdx < 0 || battle.defenderPlayer.defeated) {
        battle.executeNeutralDefenderTurn();
      } else {
        yield* lookaheadBattleAllocate(game, battle, false, defIdx, samples);
      }
      yield;
    }
  }
}

/**
 * Simulate attacking `location` with a subset of the player's armies
 * (all in-range armies when subset is null), then rollout from phase 3.
 * Returns the resulting quantile, or null if the battle cannot start.
 */
function evalBattle(game: GameSystem, playerIdx: number, location: string, armies: Army[] | null): number | null {
  const c = cloneGame(game);
  let cArmies: Army[];
  if (armies === null) {
    cArmies = c.getArmiesInRange(location);
  } else {
    cArmies = armies.map(a => {
      const idx = game.players[playerIdx].armies.indexOf(a);
      return c.players[playerIdx].armies[idx];
    }).filter(Boolean) as Army[];
  }
  if (cArmies.length === 0) return null;
  const b = c.startBattle(location, cArmies);
  if (!b) return null;
  simpleBattleLoop(c, b);
  c.resolveBattle();
  simpleRollout(c, playerIdx, 3);
  return quantile(c, playerIdx);
}

/**
 * Additive army selection with per-step labels.
 * Each step: evaluate adding each remaining army (and "done" once one army is
 * committed), label the argmax, record one sample per option, and execute the
 * greedy choice (or the NN's choice in DAgger mode).
 */
function lookaheadSelectArmies(
  game: GameSystem, playerIdx: number, location: string, targetNodeIdx: number,
  candidates: Army[], samples: Sample[] | null, model?: NNModel | null,
): Army[] {
  const selected: Army[] = [];
  const remaining = [...candidates];

  while (remaining.length > 0) {
    const selectedPerType = computeSelState(selected);
    const remainingPerType = computeSelState(remaining);
    const doneIdx = selected.length > 0 ? remaining.length : -1;

    const optionQ: number[] = remaining.map(a => {
      const q = evalBattle(game, playerIdx, location, [...selected, a]);
      return q === null ? -Infinity : q;
    });
    if (doneIdx >= 0) {
      optionQ.push(evalBattle(game, playerIdx, location, selected) ?? -Infinity);
    }

    const labelPick = optionQ.reduce((best, q, i) => (q > optionQ[best] ? i : best), 0);

    const optionStates: Float32Array[] = remaining.map(a =>
      encodeState(game, playerIdx, {
        type: "battleSelect", army: a, targetNodeIdx, selectedPerType, remainingPerType, isDone: false,
      })
    );
    if (doneIdx >= 0) {
      optionStates.push(encodeState(game, playerIdx, {
        type: "battleSelect", army: null, targetNodeIdx, selectedPerType, remainingPerType, isDone: true,
      }));
    }

    if (samples) {
      for (let i = 0; i < optionStates.length; i++) {
        const ss = emptySample(playerIdx);
        ss.state = optionStates[i];
        ss.battleSelect = i === labelPick ? 1 : 0;
        ss.battleSelectMask = 1;
        ss.value = optionQ[labelPick];
        samples.push(ss);
      }
    }

    // DAgger: NN picks the option to execute
    let execPick = labelPick;
    if (model) {
      const scores = optionStates.map(st => model.predict(st).battleSelect);
      execPick = scores.reduce((best, s, i) => (s > scores[best] ? i : best), 0);
    }

    if (execPick === doneIdx) break;
    selected.push(remaining[execPick]);
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
    if (selected.length === 0) continue;

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
      // At the army cap the engine rejects the recruit; skip so no sample is recorded for it
      if (game.countArmiesOfTypeAtLocation(player, unitType, location) >= game.maxArmiesPerTypeAtNode(location)) continue;
      const cost = game.unitStatsMap[unitType].cost;
      const affordable = Math.floor(player.money / cost);
      if (affordable <= 0) continue;
      let bestFrac = 0;
      const cBase = cloneGame(game);
      simpleNextTurn(cBase, playerIdx);
      let bestQ = quantile(cBase, playerIdx);
      for (const frac of [0.2, 0.5, 0.8, 1.0]) {
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
        ? model.predict(encodeState(game, playerIdx, {type: "recruit", locationIdx: locIdx, unitType, affordableCount: affordable})).recruitFraction
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
