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
import type GameAPI from "../../src/lib/GameAPI";
import type {ArmySelection, AttackAllocation} from "../../src/lib/GameAPI";
import type {ArmyData} from "../../src/lib/Army";
import type {BattleData} from "../../src/lib/Battle";
import {BattlePhase} from "../../src/lib/Battle";
import {NODE_ORDER, UNIT_TYPES} from "../../src/AI/nn/GameIndex";
import {keyOf, armyIdOf, attackReady, battleUnits, ArmyWalk} from "../../src/AI/nn/Armies";
import {computeActionTypeMask, executeArmyAction, ACTION_EXIT, ACTION_MOVE} from "../../src/AI/nn/ActionSpace";
import {NUM_MOVE_GROUPS, NUM_DISBAND_GROUPS, selectionsOf} from "../../src/AI/nn/UnitGroups";
import {battleStuckReport} from "../../src/AI/battleReport";

// ─── Passive ───

export function passiveTurn(api: GameAPI): void {
  api.endTurn();
}

// ─── Random ───

function randomFractions(length: number): Float32Array {
  return Float32Array.from({length}, () => Math.random());
}

/** The current player's name and armies, read anew from the game. */
function currentArmies(api: GameAPI): {owner: string; armies: ArmyData[]} {
  const state = api.getState();
  const player = state.players[state.currentPlayerIndex];
  return {owner: player.name, armies: player.armies};
}

// Terminates without a step budget: every MOVE spends at least one move point
// and every DISBAND removes at least one unit.
function randomArmyActions(api: GameAPI): void {
  const walk = new ArmyWalk(api, api.getState().currentPlayerIndex);
  for (let next = walk.next(); next; next = walk.next()) {
    const {armyId} = next;
    let army = next.army;
    for (;;) {
      // Random action type from legal options
      const mask = computeActionTypeMask(api, armyId);
      const legal = [...mask.keys()].filter(a => mask[a] > 0);
      const actionType = legal[Math.floor(Math.random() * legal.length)];
      if (actionType === ACTION_EXIT) break;

      // If MOVE: pick random legal destination and random per-group fractions
      let targetIdx = 0;
      let fractions = randomFractions(NUM_DISBAND_GROUPS);
      if (actionType === ACTION_MOVE) {
        const movable = api.getMovableLocations(armyId);
        const dest = movable[Math.floor(Math.random() * movable.length)];
        targetIdx = NODE_ORDER.indexOf(dest);
        fractions = randomFractions(NUM_MOVE_GROUPS);
      }

      if (!executeArmyAction(api, armyId, army, actionType, targetIdx, fractions)) break;

      const after = walk.acted();
      if (!after) break;
      army = after;
    }
  }
}

function randomBattleAllocate(api: GameAPI, battle: BattleData, isAttacker: boolean): void {
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
      // 70% chance to attack each target
      if (Math.random() < 0.3) continue;

      const killNeeded = api.getUnitsNeeded(armyId, target);
      const fraction = 0.5 + Math.random() * 0.5; // 50%-100% of kill amount
      let unitCount = Math.max(1, Math.round(fraction * killNeeded));
      unitCount = Math.min(unitCount, remaining);
      if (unitCount > 0) {
        allocations.push({target, count: unitCount});
        remaining -= unitCount;
      }
    }

    // Always submit (empty = pass) so the army is marked acted; reject = broken invariant
    if (!api.allocateAttack(armyId, allocations)) {
      throw new Error(battleStuckReport(
        `randomBattleAllocate: allocation rejected (${army.unitType}@${army.location})`, battle));
    }
  }
}

function randomBattleLoop(api: GameAPI): void {
  let iter = 0;
  for (;;) {
    const battle = api.getState().currentBattle;
    if (!battle) return;
    if (++iter > battle.maxArmyAttacks * 4 + 8) {
      throw new Error(battleStuckReport("randomBattleLoop stuck", battle));
    }
    if (battle.phase === BattlePhase.AttackerTurn) {
      if (battle.actedArmies.size === 0 && Math.random() < 0.1) {
        api.retreat();
        return;
      }
      randomBattleAllocate(api, battle, true);
    } else {
      randomBattleAllocate(api, battle, false);
    }
  }
}

function randomBattlePhase(api: GameAPI): void {
  const visitedNodes = new Set<string>();
  while (true) {
    const attackable = [...api.getAttackableLocations()].filter(n => !visitedNodes.has(n));
    if (attackable.length === 0) break;

    let attacked = false;
    for (const location of attackable) {
      visitedNodes.add(location);
      // 50% chance to attack each node
      if (Math.random() < 0.5) continue;

      const candidates = new Set(api.getArmiesInRange(location).map(keyOf));
      if (candidates.size === 0) continue;

      // Select armies: 60% chance each, every attack-ready unit committed
      const {owner, armies} = currentArmies(api);
      const attackers: ArmySelection[] = [];
      for (const army of armies) {
        if (!candidates.has(keyOf(army))) continue;
        if (Math.random() < 0.6) {
          attackers.push({army: armyIdOf(owner, army), selections: selectionsOf(attackReady(army))});
        }
      }
      if (attackers.length === 0) continue;

      if (!api.startBattle(attackers, location)) continue;

      randomBattleLoop(api);

      attacked = true;
      break;
    }

    if (!attacked) break;
  }
}

function randomRecruit(api: GameAPI): void {
  const locations = api.getRecruitLocations();

  for (const location of locations) {
    for (const unitType of UNIT_TYPES) {
      const state = api.getState();
      const player = state.players[state.currentPlayerIndex];
      const cost = state.unitStatsMap[unitType].cost;
      const affordable = Math.floor(player.money / cost);
      if (affordable <= 0) continue;

      const fraction = Math.random();
      const count = Math.round(fraction * affordable);
      if (count <= 0) continue;
      if (!api.canBuy(unitType, count)) continue;

      api.recruit(unitType, location, count);
    }
  }
}

export function randomTurn(api: GameAPI): void {
  const state = api.getState();
  if (state.players[state.currentPlayerIndex].defeated || api.getWinner() !== null) {
    api.endTurn();
    return;
  }

  // Phase 1: Army actions (pre-battle)
  randomArmyActions(api);

  // Phase 2: Battles
  randomBattlePhase(api);

  // Phase 3: Army actions (post-battle)
  randomArmyActions(api);

  // Phase 4: Recruitment
  randomRecruit(api);

  api.endTurn();
}
