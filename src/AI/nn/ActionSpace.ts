/**
 * Action space v3: sequential army actions (no attack — battles are separate).
 *
 * Action types (5):
 *   0 EXIT    — stop acting with this army
 *   1 MERGE   — merge with largest same-type army at same location
 *   2 MOVE    — move this army to target node
 *   3 SPLIT   — split count_fraction units into a new army at same location
 *   4 DISBAND — disband count_fraction units
 *
 * Auxiliary outputs:
 *   target_node [16] — for MOVE destination
 *   count_fraction [0,1] — for SPLIT/DISBAND count, or recruit amount
 */
import type GameSystem from "../../lib/GameSystem";
import type Player from "../../lib/Player";
import type Army from "../../lib/Army";
import {NODE_ORDER} from "./StateEncoder";

// ─── Action type constants ───

export const NUM_ACTION_TYPES = 5;
export const ACTION_EXIT = 0;
export const ACTION_MERGE = 1;
export const ACTION_MOVE = 2;
export const ACTION_SPLIT = 3;
export const ACTION_DISBAND = 4;

export const ACTION_NAMES = ["EXIT", "MERGE", "MOVE", "SPLIT", "DISBAND"];

// ─── Action type mask ───

/**
 * Compute legal action type mask for an army.
 * Returns Float32Array[5]: 1.0 = legal, 0.0 = illegal.
 */
export function computeActionTypeMask(
  game: GameSystem,
  player: Player,
  army: Army,
  noMerge?: Set<object>,
): Float32Array {
  const mask = new Float32Array(NUM_ACTION_TYPES);

  // EXIT is always legal
  mask[ACTION_EXIT] = 1;

  // MERGE: same-type army at same location (blocked for split-created armies)
  if (!noMerge || !noMerge.has(army)) {
    const canMerge = player.armies.some(
      a => a !== army && a.location === army.location && a.unitType === army.unitType,
    );
    if (canMerge) mask[ACTION_MERGE] = 1;
  }

  // MOVE: has remaining moves and reachable locations
  if (army.remainingMoves > 0) {
    const movable = army.getMovableLocations(game.gameMap, game.enemyLocations);
    if (movable.length > 0) mask[ACTION_MOVE] = 1;
  }

  // SPLIT: need >= 2 units + not at army cap for this type at this node
  if (army.units.length >= 2) {
    const cap = game.maxArmiesPerTypeAtNode(army.location);
    if (game.countArmiesOfTypeAtLocation(player, army.unitType, army.location) < cap) {
      mask[ACTION_SPLIT] = 1;
    }
  }

  // DISBAND: has units
  if (army.units.length > 0) mask[ACTION_DISBAND] = 1;

  return mask;
}

// ─── Action execution ───

/**
 * Execute an army action.
 * Returns true if the action was successfully executed, false otherwise.
 */
export function executeArmyAction(
  game: GameSystem,
  player: Player,
  army: Army,
  actionType: number,
  targetNodeIdx: number,
  countFraction: number,
): boolean {
  if (actionType === ACTION_EXIT) return false;

  if (actionType === ACTION_MERGE) {
    const bestOther = player.armies
      .filter(other => other !== army && other.location === army.location && other.unitType === army.unitType)
      .reduce<Army | null>((best, other) => (!best || other.units.length > best.units.length ? other : best), null);
    if (bestOther) {
      game.mergePlayerArmies(army, bestOther);
      return true;
    }
    return false;
  }

  if (actionType === ACTION_MOVE) {
    const targetNode = NODE_ORDER[targetNodeIdx];
    return game.movePlayerArmy(army, targetNode);
  }

  if (actionType === ACTION_SPLIT) {
    const n = army.units.length;
    const count = Math.max(1, Math.min(n - 1, Math.round(countFraction * n)));
    const newArmy = game.splitPlayerArmy(army, count);
    return newArmy !== null;
  }

  if (actionType === ACTION_DISBAND) {
    const n = army.units.length;
    const count = Math.max(1, Math.min(n, Math.round(countFraction * n)));
    game.disbandPlayerArmy(army, count);
    return true;
  }

  return false;
}
