/**
 * Baseline opponents for evaluation and training.
 *
 * Both follow the exact same 4-phase decision loop as the NN TurnExecutor:
 *   Phase 1: Army actions (pre-battle)
 *   Phase 2: Battle loop
 *   Phase 3: Army actions (post-battle)
 *   Phase 4: Recruitment
 *
 * - passive: always picks EXIT / NO / skip at every decision point
 * - random: coin flip at every decision point
 */
import type GameSystem from "../../src/lib/GameSystem";
import type Army from "../../src/lib/Army";
import type Unit from "../../src/lib/Unit";
import {BattlePhase, BattleResult} from "../../src/lib/Battle";
import type Battle from "../../src/lib/Battle";
import {calculateUnitsNeeded} from "../../src/lib/Combat";
import {NODE_ORDER, UNIT_TYPES} from "../../src/AI/nn/GameIndex";
import {computeActionTypeMask, executeArmyAction, ACTION_EXIT, ACTION_MOVE} from "../../src/AI/nn/ActionSpace";
import {NUM_MOVE_GROUPS, NUM_DISBAND_GROUPS} from "../../src/AI/nn/UnitGroups";
import {battleStuckReport} from "../../src/AI/battleReport";

// ─── Passive ───

export function passiveTurn(game: GameSystem): void {
  game.endTurn();
}

// ─── Random ───

function randomFractions(length: number): Float32Array {
  return Float32Array.from({length}, () => Math.random());
}

// Terminates without a step budget: every MOVE spends at least one move point
// and every DISBAND removes at least one unit.
function randomArmyActions(game: GameSystem): void {
  const player = game.currentPlayer;
  const processed = new Set<Army>();
  let ai = 0;
  while (ai < player.armies.length) {
    const army = player.armies[ai];
    if (processed.has(army)) { ai++; continue; }
    processed.add(army);

    for (;;) {
      // Random action type from legal options
      const mask = computeActionTypeMask(game, army);
      const legal = [...mask.keys()].filter(a => mask[a] > 0);
      const actionType = legal[Math.floor(Math.random() * legal.length)];
      if (actionType === ACTION_EXIT) break;

      // If MOVE: pick random legal destination and random per-group fractions
      let targetIdx = 0;
      let fractions = randomFractions(NUM_DISBAND_GROUPS);
      if (actionType === ACTION_MOVE) {
        const movable = army.getMovableLocations(game.gameMap, game.enemyLocations);
        const dest = movable[Math.floor(Math.random() * movable.length)];
        targetIdx = NODE_ORDER.indexOf(dest);
        fractions = randomFractions(NUM_MOVE_GROUPS);
      }

      if (!executeArmyAction(game, army, actionType, targetIdx, fractions)) break;

      const newIdx = player.armies.indexOf(army);
      if (newIdx < 0) break;
      ai = newIdx;
    }

    const finalIdx = player.armies.indexOf(army);
    ai = finalIdx < 0 ? ai : finalIdx + 1;
  }
}

function randomBattleAllocate(battle: Battle, isAttacker: boolean): void {
  const armies = isAttacker ? battle.attackerArmies : battle.defenderArmies;

  for (const army of [...armies]) {
    if (!battle.canAct(army)) continue;
    if (battle.result !== BattleResult.Ongoing) return;

    const targets = battle.getTargetsInRange(army);

    let remaining = army.battleUnits.length;
    const allocations = new Map<Army, number>();

    for (const target of targets) {
      if (remaining <= 0) break;
      // 70% chance to attack each target
      if (Math.random() < 0.3) continue;

      const killNeeded = calculateUnitsNeeded(army, target);
      const fraction = 0.5 + Math.random() * 0.5; // 50%-100% of kill amount
      let unitCount = Math.max(1, Math.round(fraction * killNeeded));
      unitCount = Math.min(unitCount, remaining);
      if (unitCount > 0) {
        allocations.set(target, unitCount);
        remaining -= unitCount;
      }
    }

    // Always submit (empty = pass) so the army is marked acted; reject = broken invariant
    if (!battle.allocateAttack(army, allocations)) {
      throw new Error(battleStuckReport(
        `randomBattleAllocate: allocation rejected (${army.unitType}@${army.location})`, battle));
    }
  }
}

function randomBattleLoop(battle: Battle): void {
  const maxIter = battle.maxArmyAttacks * 4 + 8;
  let iter = 0;
  while (battle.result === BattleResult.Ongoing) {
    if (++iter > maxIter) {
      throw new Error(battleStuckReport("randomBattleLoop stuck", battle));
    }
    if (battle.phase === BattlePhase.AttackerTurn) {
      if (battle.actedArmies.size === 0 && Math.random() < 0.1) {
        battle.retreat();
        return;
      }
      randomBattleAllocate(battle, true);
    } else {
      randomBattleAllocate(battle, false);
    }
  }
}

function randomBattlePhase(game: GameSystem): void {
  const visitedNodes = new Set<string>();
  while (true) {
    const attackable = Array.from(game.attackableLocations).filter(n => !visitedNodes.has(n));
    if (attackable.length === 0) break;

    let attacked = false;
    for (const location of attackable) {
      visitedNodes.add(location);
      // 50% chance to attack each node
      if (Math.random() < 0.5) continue;

      const candidates = game.getArmiesInRange(location);
      if (candidates.length === 0) continue;

      // Select armies: 60% chance each, every attack-ready unit committed
      const selected = new Map<Army, Unit[]>();
      for (const army of candidates) {
        if (Math.random() < 0.6) selected.set(army, army.attackCandidates);
      }
      if (selected.size === 0) continue;

      const battle = game.startBattle(location, selected);
      if (!battle) continue;

      randomBattleLoop(battle);
      game.resolveBattle();

      attacked = true;
      break;
    }

    if (!attacked) break;
  }
}

function randomRecruit(game: GameSystem): void {
  const player = game.currentPlayer;
  const locs = game.recruitLocations;

  for (const location of locs) {
    for (const unitType of UNIT_TYPES) {
      const cost = game.unitStatsMap[unitType].cost;
      const affordable = Math.floor(player.money / cost);
      if (affordable <= 0) continue;

      const fraction = Math.random();
      const count = Math.round(fraction * affordable);
      if (count <= 0) continue;
      if (!player.canBuy(unitType, count)) continue;

      game.recruitPlayerArmy(unitType, location, count);
    }
  }
}

export function randomTurn(game: GameSystem): void {
  const player = game.currentPlayer;
  if (player.defeated || game.winner) {
    game.endTurn();
    return;
  }

  // Phase 1: Army actions (pre-battle)
  randomArmyActions(game);

  // Phase 2: Battles
  randomBattlePhase(game);

  // Phase 3: Army actions (post-battle)
  randomArmyActions(game);

  // Phase 4: Recruitment
  randomRecruit(game);

  game.endTurn();
}
