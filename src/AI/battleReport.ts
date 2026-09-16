/**
 * Battle diagnostics for the stuck-loop guards in the decision loops.
 * Reads only public Battle state; src/lib stays untouched.
 */
import type Battle from "../lib/Battle";
import type Army from "../lib/Army";

export function battleStuckReport(context: string, battle: Battle): string {
  const fmt = (armies: Army[]) => armies.map(a =>
    `${a.unitType}(${a.units.length}u@${a.location} rng=${a.unitStats.range})`
  ).join(", ") || "(none)";
  return [
    `${context}: loc=${battle.targetLocation} round=${battle.round}/${battle.maxRounds} phase=${battle.phase} result=${battle.result}`,
    `  atk: ${fmt(battle.attackerArmies)}`,
    `  def: ${fmt(battle.defenderArmies)}`,
    `  hasActable=${battle.hasActableArmies} acted=${battle.actedArmies.size}`,
  ].join("\n");
}
