/**
 * Draw diagnosis: play NN (argmax) vs Random on randomized configs and, for
 * every game that reaches the turn cap, print the NN's per-turn trajectory
 * (units, nodes, money, decisions by type, battles) and the largest encoded
 * feature it saw, so the failure to finish can be attributed.
 *
 * Usage: npx tsx training/scripts/test/testDraws.ts [modelDir] [games]
 */
import {setupBackend} from "../../src/setupBackend";
import {NNModel} from "../../../src/AI/nn/NNModel";
import {nodeFileSystem} from "../../src/nodeIO";
import {executeNNTurn} from "../../../src/AI/TurnExecutor";
import type {DecisionInfo, TurnOptions} from "../../../src/AI/TurnExecutor";
import {ACTION_NAMES} from "../../../src/AI/nn/ActionSpace";
import {BATTLE_TARGET_STOP} from "../../../src/AI/nn/NNModel";
import {randomTurn} from "../../src/Opponents";
import type GameAPI from "../../../src/lib/GameAPI";
import {armyIdOf} from "../../../src/AI/nn/Armies";
import {MAX_TURNS, MODEL_DIR_PHASE1, countNodes, createRandomizedGame} from "../../src/trainUtils";
import type {GameData} from "../../../src/lib/GameSystem";
import * as path from "path";

const modelDir = process.argv[2] ? path.resolve(process.argv[2]) : MODEL_DIR_PHASE1;
const numGames = parseInt(process.argv[3] || "27");

interface TurnStats {
  turn: number; units: number; nodes: number; money: number; maxFeature: number;
  actions: Record<string, number>; moves: number; battles: number; stops: number; enemyEmptyReachable: number;
}

function enemyEmptyReachable(api: GameAPI, state: GameData, nnIdx: number): number {
  // Nodes owned by an opponent with no army on them, and some own army can move there this turn
  const self = state.players[nnIdx];
  const reachable = new Set(self.armies.flatMap(army => api.getMovableLocations(armyIdOf(self.name, army))));
  let count = 0;
  for (const [node, owner] of state.nodeOwnership) {
    if (owner === null || owner === self) continue;
    if (reachable.has(node)) count++;
  }
  return count;
}

async function main() {
  await setupBackend();
  const model = new NNModel();
  await model.load(nodeFileSystem(modelDir));

  let wins = 0, draws = 0, losses = 0;
  for (let g = 0; g < numGames; g++) {
    const api = createRandomizedGame();
    const nnIdx = g % 3;
    const nnName = ["Blue", "Red", "Green"][nnIdx];
    const history: TurnStats[] = [];
    let current: TurnStats | null = null;
    const opts: TurnOptions = {
      onDecision: (info: DecisionInfo) => {
        if (!current) return;
        let max = 0;
        for (const v of info.state) if (Math.abs(v) > max) max = Math.abs(v);
        current.maxFeature = Math.max(current.maxFeature, max);
        const a = info.action;
        if (a.type === "army") current.actions[ACTION_NAMES[a.actionType]] = (current.actions[ACTION_NAMES[a.actionType]] ?? 0) + 1;
        if (a.type === "moveCount") current.moves++;
        if (a.type === "battleTarget") {
          if (a.battleTarget === BATTLE_TARGET_STOP) current.stops++;
          else current.battles++;
        }
      },
    };

    for (let turn = 0; turn < MAX_TURNS * 3 && !api.isGameOver(); turn++) {
      const state = api.getState();
      if (state.currentPlayerIndex === nnIdx) {
        const self = state.players[nnIdx];
        current = {
          turn: state.turnCount, units: self.armies.reduce((s, a) => s + a.units.length, 0), nodes: countNodes(state, self),
          money: self.money, maxFeature: 0, actions: {}, moves: 0, battles: 0, stops: 0,
          enemyEmptyReachable: enemyEmptyReachable(api, state, nnIdx),
        };
        executeNNTurn(api, model, opts);
        history.push(current);
        current = null;
      } else {
        randomTurn(api);
      }
    }

    const winner = api.getWinner() ?? "draw";
    if (winner === "draw") draws++; else if (winner === nnName) wins++; else losses++;
    const last = history[history.length - 1];
    console.log(`Game ${g + 1}/${numGames} (${nnName}): ${winner} T${api.getState().turnCount} units=${last.units} nodes=${last.nodes} money=${last.money} maxFeature=${last.maxFeature.toFixed(2)}`);
    if (winner === "draw") {
      for (const h of history) {
        if (h.turn % 10 !== 1 && h.turn !== history.length) continue;
        const acts = Object.entries(h.actions).map(([k, v]) => `${k}=${v}`).join(" ");
        console.log(`  T${String(h.turn).padStart(3)} units=${String(h.units).padStart(4)} nodes=${String(h.nodes).padStart(2)} money=${String(h.money).padStart(5)} ` +
          `maxFeat=${h.maxFeature.toFixed(2)} ${acts} moves=${h.moves} battles=${h.battles} stops=${h.stops} emptyEnemyReachable=${h.enemyEmptyReachable}`);
      }
    }
  }
  console.log(`\nSummary: ${wins}W ${losses}L ${draws}D out of ${numGames}`);
  model.dispose();
}

main().catch(e => { console.error(e); process.exit(1); });
