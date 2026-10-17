/**
 * Deterministic behavior trace of the core, for comparing it before and after
 * a refactor. Math.random is replaced by a seeded generator; the script then
 * plays random-vs-random games and greedy-vs-random games on randomized
 * configs and one 3-NN game on the default config (the published model,
 * deterministic play), and prints one line per turn: the money and unit count of every player and a hash of the
 * saved state. Two runs print the same lines exactly when the games are the
 * same.
 *
 * Usage: npx tsx training/scripts/test/traceCore.ts > <file>
 */
import {setupBackend} from "../../src/setupBackend";
import {NNModel} from "../../../src/AI/nn/NNModel";
import {nodeFileSystem} from "../../src/nodeIO";
import {executeNNTurn} from "../../../src/AI/TurnExecutor";
import DefaultConfig from "../../../src/lib/data/DefaultConfig";
import {randomTurn} from "../../src/Opponents";
import GameAPI from "../../../src/lib/GameAPI";
import {greedyTurn} from "../../src/GreedyAI";
import {createRandomizedGame, seededRandom} from "../../src/trainUtils";
import * as crypto from "crypto";
import * as path from "path";

const SEED = 20261003;
const RANDOM_GAMES = 40;
const GREEDY_GAMES = 3;

function turnLine(label: string, api: GameAPI, turn: number, playerIdx: number): string {
  const state = api.getState();
  const money = state.players.map(p => p.money).join(",");
  const units = state.players.map(p => p.armies.reduce((sum, army) => sum + army.units.length, 0)).join(",");
  const hash = crypto.createHash("md5").update(JSON.stringify(api.toJSON())).digest("hex").slice(0, 8);
  return `${label} T${turn} P${playerIdx} money=${money} units=${units} state=${hash}`;
}

function endLine(label: string, api: GameAPI): string {
  return `${label} END T${api.getState().turnCount} winner=${api.getWinner() ?? "draw"}`;
}

async function main() {
  await setupBackend();
  Math.random = seededRandom(SEED);

  for (let g = 0; g < RANDOM_GAMES; g++) {
    const api = createRandomizedGame();
    while (!api.isGameOver()) {
      const {turnCount: turn, currentPlayerIndex: playerIdx} = api.getState();
      randomTurn(api);
      console.log(turnLine(`R${g}`, api, turn, playerIdx));
    }
    console.log(endLine(`R${g}`, api));
  }

  for (let g = 0; g < GREEDY_GAMES; g++) {
    const api = createRandomizedGame();
    const greedyIdx = g % 3;
    while (!api.isGameOver()) {
      const {turnCount: turn, currentPlayerIndex: playerIdx} = api.getState();
      if (playerIdx === greedyIdx) greedyTurn(api);
      else randomTurn(api);
      console.log(turnLine(`G${g}`, api, turn, playerIdx));
    }
    console.log(endLine(`G${g}`, api));
  }

  const model = new NNModel();
  await model.load(nodeFileSystem(path.resolve("public/model")));
  const api = GameAPI.create(DefaultConfig);
  while (!api.isGameOver()) {
    const {turnCount: turn, currentPlayerIndex: playerIdx} = api.getState();
    executeNNTurn(api, model);
    console.log(turnLine("N", api, turn, playerIdx));
  }
  console.log(endLine("N", api));
  model.dispose();
}

main().catch(e => { console.error(e); process.exit(1); });
