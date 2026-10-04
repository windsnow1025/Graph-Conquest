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
import GameSystem from "../../../src/lib/GameSystem";
import Config from "../../../src/lib/data/Config";
import {randomTurn} from "../../src/Opponents";
import {greedyTurn} from "../../src/GreedyAI";
import {createRandomizedGame} from "../../src/trainUtils";
import * as crypto from "crypto";
import * as path from "path";

const SEED = 20261003;
const RANDOM_GAMES = 40;
const GREEDY_GAMES = 3;

/** mulberry32 */
function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function turnLine(label: string, game: GameSystem, turn: number, playerIdx: number): string {
  const money = game.players.map(p => p.money).join(",");
  const units = game.players.map(p => p.armies.reduce((sum, army) => sum + army.units.length, 0)).join(",");
  const state = crypto.createHash("md5").update(JSON.stringify(game.toJSON())).digest("hex").slice(0, 8);
  return `${label} T${turn} P${playerIdx} money=${money} units=${units} state=${state}`;
}

function endLine(label: string, game: GameSystem): string {
  return `${label} END T${game.turnCount} winner=${game.winner?.name ?? "draw"}`;
}

async function main() {
  await setupBackend();
  Math.random = seededRandom(SEED);

  for (let g = 0; g < RANDOM_GAMES; g++) {
    const game = createRandomizedGame();
    while (!game.gameOver) {
      const turn = game.turnCount, playerIdx = game.currentPlayerIndex;
      randomTurn(game);
      console.log(turnLine(`R${g}`, game, turn, playerIdx));
    }
    console.log(endLine(`R${g}`, game));
  }

  for (let g = 0; g < GREEDY_GAMES; g++) {
    const game = createRandomizedGame();
    const greedyIdx = g % 3;
    while (!game.gameOver) {
      const turn = game.turnCount, playerIdx = game.currentPlayerIndex;
      if (playerIdx === greedyIdx) greedyTurn(game);
      else randomTurn(game);
      console.log(turnLine(`G${g}`, game, turn, playerIdx));
    }
    console.log(endLine(`G${g}`, game));
  }

  const model = new NNModel();
  await model.load(nodeFileSystem(path.resolve("public/model")));
  const game = new GameSystem(Config);
  while (!game.gameOver) {
    const turn = game.turnCount, playerIdx = game.currentPlayerIndex;
    executeNNTurn(game, model);
    console.log(turnLine("N", game, turn, playerIdx));
  }
  console.log(endLine("N", game));
  model.dispose();
}

main().catch(e => { console.error(e); process.exit(1); });
