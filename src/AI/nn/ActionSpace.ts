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
import type {ArmyData} from "../../lib/Army";
import type GameAPI from "../../lib/GameAPI";
import type {ArmyId} from "../../lib/GameAPI";
import type {UnitData} from "../../lib/Unit";
import {NODE_ORDER} from "./GameIndex";
import {attackReady} from "./Armies";
import {moveGroups, commitGroups, disbandGroups, takeByFractions, selectionsOf} from "./UnitGroups";

// ─── Action type constants ───

export const NUM_ACTION_TYPES = 3;
export const ACTION_EXIT = 0;
export const ACTION_MOVE = 1;
export const ACTION_DISBAND = 2;

export const ACTION_NAMES = ["EXIT", "MOVE", "DISBAND"];

// ─── Action type mask ───

/**
 * Compute legal action type mask for an army of the current player.
 * Returns Float32Array[3]: 1.0 = legal, 0.0 = illegal.
 */
export function computeActionTypeMask(api: GameAPI, armyId: ArmyId): Float32Array {
  const mask = new Float32Array(NUM_ACTION_TYPES);

  // EXIT is always legal
  mask[ACTION_EXIT] = 1;

  // MOVE: some unit can reach some node
  if (api.getMovableLocations(armyId).length > 0) mask[ACTION_MOVE] = 1;

  // DISBAND: an army always has units
  mask[ACTION_DISBAND] = 1;

  return mask;
}

// ─── Fractions to units ───

/** The attack-ready units committed to a battle under per-commit-group fractions. */
export function unitsToCommit(army: ArmyData, fractions: ArrayLike<number>): UnitData[] {
  return takeByFractions(commitGroups(attackReady(army)), fractions, false);
}

// ─── Action execution ───

/**
 * Execute an action of an army of the current player, named by its identifier and
 * given with its current data. `fractions` is per move group for MOVE and per
 * disband group for DISBAND.
 * Returns true if the action was successfully executed, false otherwise.
 */
export function executeArmyAction(
  api: GameAPI,
  armyId: ArmyId,
  army: ArmyData,
  actionType: number,
  targetNodeIdx: number,
  fractions: ArrayLike<number>,
): boolean {
  if (actionType === ACTION_MOVE) {
    const targetNode = NODE_ORDER[targetNodeIdx];
    const candidates = api.getMoveCandidates(armyId, targetNode);
    return api.moveUnits(armyId, selectionsOf(takeByFractions(moveGroups(candidates), fractions, false)), targetNode);
  }

  if (actionType === ACTION_DISBAND) {
    return api.disbandUnits(armyId, selectionsOf(takeByFractions(disbandGroups(army.units), fractions, true)));
  }

  return false;
}
