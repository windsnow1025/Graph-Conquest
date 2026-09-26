/**
 * Action space v4: sequential army actions (no attack — battles are separate).
 *
 * Action types (3):
 *   0 EXIT    — stop acting with this army
 *   1 MOVE    — move units to a target node; how many per move group is a
 *               separate decision taken after the destination
 *   2 DISBAND — disband units, a fraction per disband group
 *
 * Which units realise a per-group count is fixed here: the healthiest first
 * for moving and committing to battle, the weakest first for disbanding.
 */
import type GameSystem from "../../lib/GameSystem";
import type Army from "../../lib/Army";
import type Unit from "../../lib/Unit";
import {NODE_ORDER} from "./GameIndex";
import {moveGroups, commitGroups, disbandGroups, takeByFractions} from "./UnitGroups";

// ─── Action type constants ───

export const NUM_ACTION_TYPES = 3;
export const ACTION_EXIT = 0;
export const ACTION_MOVE = 1;
export const ACTION_DISBAND = 2;

export const ACTION_NAMES = ["EXIT", "MOVE", "DISBAND"];

// ─── Action type mask ───

/**
 * Compute legal action type mask for an army.
 * Returns Float32Array[3]: 1.0 = legal, 0.0 = illegal.
 */
export function computeActionTypeMask(game: GameSystem, army: Army): Float32Array {
  const mask = new Float32Array(NUM_ACTION_TYPES);

  // EXIT is always legal
  mask[ACTION_EXIT] = 1;

  // MOVE: some unit can reach some node
  if (army.getMovableLocations(game.gameMap, game.enemyLocations).length > 0) mask[ACTION_MOVE] = 1;

  // DISBAND: an army always has units
  mask[ACTION_DISBAND] = 1;

  return mask;
}

// ─── Fractions to units ───

/** The units that move to the target node under per-move-group fractions. */
function unitsToMove(game: GameSystem, army: Army, targetNode: string, fractions: ArrayLike<number>): Unit[] {
  return takeByFractions(moveGroups(army.getMoveCandidates(targetNode, game.gameMap, game.enemyLocations)), fractions, false);
}

/** The attack-ready units committed to a battle under per-commit-group fractions. */
export function unitsToCommit(army: Army, fractions: ArrayLike<number>): Unit[] {
  return takeByFractions(commitGroups(army.attackCandidates), fractions, false);
}

/** The units disbanded under per-disband-group fractions. */
function unitsToDisband(army: Army, fractions: ArrayLike<number>): Unit[] {
  return takeByFractions(disbandGroups(army.units), fractions, true);
}

// ─── Action execution ───

/**
 * Execute an army action. `fractions` is per move group for MOVE and per
 * disband group for DISBAND.
 * Returns true if the action was successfully executed, false otherwise.
 */
export function executeArmyAction(
  game: GameSystem,
  army: Army,
  actionType: number,
  targetNodeIdx: number,
  fractions: ArrayLike<number>,
): boolean {
  if (actionType === ACTION_MOVE) {
    const targetNode = NODE_ORDER[targetNodeIdx];
    return game.movePlayerUnits(army, targetNode, unitsToMove(game, army, targetNode, fractions));
  }

  if (actionType === ACTION_DISBAND) {
    return game.disbandPlayerUnits(army, unitsToDisband(army, fractions));
  }

  return false;
}
