/**
 * Armies as the API names them: an army of a player is one per node and unit
 * type, so the two name it within the player, and the owner's name completes
 * the identifier.
 */
import type {ArmyData} from "../../lib/Army";
import type {UnitData} from "../../lib/Unit";
import type {ArmyId} from "../../lib/GameAPI";

/** The name of an army within its player: its node and unit type. */
export function keyOf(army: {location: string; unitType: string}): string {
  return `${army.location}|${army.unitType}`;
}

export function armyIdOf(owner: string, army: ArmyData): ArmyId {
  return {owner, location: army.location, unitType: army.unitType};
}

/** The units of an army that can still attack this turn. */
export function attackReady(army: ArmyData): UnitData[] {
  return army.units.filter(unit => unit.canAttack);
}

/** The units of an army that are in the current battle. */
export function battleUnits(army: ArmyData): UnitData[] {
  return army.units.filter(unit => unit.inBattle);
}
