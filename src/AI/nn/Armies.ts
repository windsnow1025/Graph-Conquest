/**
 * Armies as the API names them: an army of a player is one per node and unit
 * type, so the two name it within the player, and the owner's name completes
 * the identifier.
 */
import type GameAPI from "../../lib/GameAPI";
import type {ArmyId, GameData, ArmyData, UnitData} from "../../lib/GameAPI";

/** The name of an army within its player: its node and unit type. */
export function keyOf(army: {location: string; unitType: string}): string {
  return `${army.location}|${army.unitType}`;
}

export function armyIdOf(owner: string, army: ArmyData): ArmyId {
  return {owner, location: army.location, unitType: army.unitType};
}

/** The name of an army in the game: its owner, node and unit type. */
export function keyOfId(id: ArmyId): string {
  return `${id.owner}|${id.location}|${id.unitType}`;
}

/** The units of an army that can still attack this turn. */
export function attackReady(army: ArmyData): UnitData[] {
  return army.units.filter(unit => unit.canAttack);
}

/** The units of an army that are in the current battle. */
export function battleUnits(army: ArmyData): UnitData[] {
  return army.units.filter(unit => unit.inBattle);
}

/**
 * A walk over a player's armies for a phase of army actions, as the engine's
 * loops walked them: each army acts until it exits, an army that forms during
 * the phase acts after the existing ones, and an army that received units from
 * an army already walked does not act again. The name of an army whose units
 * have all left is forgotten, so an army that forms again at that node acts
 * like the new army it is.
 */
export class ArmyWalk {
  state: GameData;
  private readonly api: GameAPI;
  private readonly playerIdx: number;
  private readonly owner: string;
  private armies: ArmyData[];
  private readonly processed = new Set<string>();
  private ai = 0;
  private key = "";

  constructor(api: GameAPI, playerIdx: number) {
    this.api = api;
    this.playerIdx = playerIdx;
    this.state = api.getState();
    this.armies = this.state.players[playerIdx].armies;
    this.owner = this.state.players[playerIdx].name;
  }

  /** The next army to act, with its identifier, or null when every army has acted. */
  next(): {army: ArmyData; armyId: ArmyId} | null {
    if (this.key !== "") {
      const finalIdx = this.armies.findIndex(army => keyOf(army) === this.key);
      this.ai = finalIdx < 0 ? this.ai : finalIdx + 1;
      this.key = "";
    }
    while (this.ai < this.armies.length) {
      const army = this.armies[this.ai];
      const key = keyOf(army);
      if (this.processed.has(key)) { this.ai++; continue; }
      this.processed.add(key);
      this.key = key;
      return {army, armyId: armyIdOf(this.owner, army)};
    }
    return null;
  }

  /** After an action of the current army: the army as it now is, or null when its units have all left. */
  acted(): ArmyData | null {
    this.state = this.api.getState();
    this.armies = this.state.players[this.playerIdx].armies;
    for (const name of this.processed) {
      if (!this.armies.some(army => keyOf(army) === name)) this.processed.delete(name);
    }
    const newIdx = this.armies.findIndex(army => keyOf(army) === this.key);
    if (newIdx < 0) return null;
    this.ai = newIdx;
    return this.armies[newIdx];
  }
}
