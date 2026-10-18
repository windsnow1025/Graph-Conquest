/**
 * Unit groups: the discrete part of a unit's turn state, (remainingMoves, canAttack).
 * Units of one army that share a group are interchangeable up to health, so the
 * encoder reports counts per group and the fraction heads decide per group.
 *
 * Group index = remainingMoves * 2 + (canAttack ? 0 : 1):
 *   0: 0 moves, can attack     1: 0 moves, cannot attack
 *   2: 1 move,  can attack     3: 1 move,  cannot attack
 *   4: 2 moves, can attack     5: 2 moves, cannot attack
 *
 * Decision-specific group lists are slices of this order:
 *   move    (4): groups 2..5, the units with at least one move; a group is
 *                eligible for a destination when its moves reach it
 *   commit  (3): groups 0, 2, 4, the attack-ready units by remaining moves
 *   disband (6): all groups
 */
import type {UnitData} from "../../lib/Unit";
import type {UnitSelection} from "../../lib/GameAPI";

const MAX_MOVES = 2; // unit speed in this game is at most 2
export const NUM_UNIT_GROUPS = (MAX_MOVES + 1) * 2;
export const NUM_MOVE_GROUPS = MAX_MOVES * 2;
export const NUM_COMMIT_GROUPS = MAX_MOVES + 1;
export const NUM_DISBAND_GROUPS = NUM_UNIT_GROUPS;

function unitGroup(unit: UnitData): number {
  return unit.remainingMoves * 2 + (unit.canAttack ? 0 : 1);
}

/** Partition units into the 6 groups, keeping the army's unit order within a group. */
export function groupUnits<T extends UnitData>(units: T[]): T[][] {
  const groups: T[][] = Array.from({length: NUM_UNIT_GROUPS}, () => []);
  for (const unit of units) groups[unitGroup(unit)].push(unit);
  return groups;
}

export function groupCounts(units: UnitData[]): number[] {
  const counts = new Array<number>(NUM_UNIT_GROUPS).fill(0);
  for (const unit of units) counts[unitGroup(unit)]++;
  return counts;
}

/** Move groups of the units that can reach a destination (groups 2..5 as 0..3). */
export function moveGroups<T extends UnitData>(candidates: T[]): T[][] {
  return groupUnits(candidates).slice(2);
}

/** Commit groups: the attack-ready units by remaining moves (groups 0, 2, 4 as 0..2). */
export function commitGroups<T extends UnitData>(attackCandidates: T[]): T[][] {
  const groups = groupUnits(attackCandidates);
  return [groups[0], groups[2], groups[4]];
}

export function disbandGroups<T extends UnitData>(units: T[]): T[][] {
  return groupUnits(units);
}

/** 1 for each non-empty group. */
export function groupMask(groups: UnitData[][]): Float32Array {
  return Float32Array.from(groups, group => (group.length > 0 ? 1 : 0));
}

/** Per-group counts as fractions of the group sizes; 0 for empty groups. */
export function countsToFractions(groups: UnitData[][], counts: ArrayLike<number>): Float32Array {
  return Float32Array.from(groups, (group, g) => (group.length > 0 ? counts[g] / group.length : 0));
}

/**
 * Take round(fraction × size) units from each group, the healthiest first (or
 * the weakest first for disbanding). A selection that rounds to nothing takes
 * one unit from the group with the highest fraction, so a chosen action always
 * acts on at least one unit.
 */
export function takeByFractions<T extends UnitData>(groups: T[][], fractions: ArrayLike<number>, weakestFirst: boolean): T[] {
  const taken: T[] = [];
  let best = -1;
  for (let g = 0; g < groups.length; g++) {
    const group = groups[g];
    if (group.length === 0) continue;
    if (best < 0 || fractions[g] > fractions[best]) best = g;
    const count = Math.round(fractions[g] * group.length);
    taken.push(...sortByHealth(group, weakestFirst).slice(0, count));
  }
  if (taken.length === 0 && best >= 0) taken.push(sortByHealth(groups[best], weakestFirst)[0]);
  return taken;
}

function sortByHealth<T extends UnitData>(units: T[], ascending: boolean): T[] {
  return [...units].sort((a, b) => (ascending ? a.currentHealth - b.currentHealth : b.currentHealth - a.currentHealth));
}

type UnitState = Pick<UnitSelection, "currentHealth" | "remainingMoves" | "canAttack">;

/** Whether two units, or a unit and a selection, agree on health, remaining moves and attack readiness. */
export function sameState(a: UnitState, b: UnitState): boolean {
  return a.currentHealth === b.currentHealth && a.remainingMoves === b.remainingMoves && a.canAttack === b.canAttack;
}

/** Units as the API names them: one state and count per state, in order of first appearance. */
export function selectionsOf(units: UnitData[]): UnitSelection[] {
  const selections = new Map<string, UnitSelection>();
  for (const unit of units) {
    const state = `${unit.currentHealth}|${unit.remainingMoves}|${unit.canAttack}`;
    const selection = selections.get(state);
    if (selection) selection.count++;
    else selections.set(state, {currentHealth: unit.currentHealth, remainingMoves: unit.remainingMoves, canAttack: unit.canAttack, count: 1});
  }
  return [...selections.values()];
}
