/**
 * Battle diagnostics for the stuck-loop guards in the decision loops.
 * Reads the battle's data.
 */
import type {BattleData, ArmyData} from "../lib/GameAPI";
import {battleUnits} from "./nn/Armies";

export function battleStuckReport(context: string, battle: BattleData): string {
  const fmt = (armies: ArmyData[]) => armies.map(a =>
    `${a.unitType}(${battleUnits(a).length}/${a.units.length}u@${a.location} rng=${a.unitStats.range} attacks=${battle.remainingAttacks.get(a)}/${battle.maxArmyAttacks})`
  ).join(", ") || "(none)";
  return [
    `${context}: loc=${battle.targetLocation} round=${battle.round} phase=${battle.phase} result=${battle.result}`,
    `  atk: ${fmt(battle.attackerArmies)}`,
    `  def: ${fmt(battle.defenderArmies)}`,
    `  acted=${battle.actedArmies.size}`,
  ].join("\n");
}
