/**
 * Standalone test: NN vs Random, more games for statistical significance.
 * Usage: npx tsx training/testOnly.ts [modelDir] [numGames]
 */
import {setupBackend} from "../../src/setupBackend";
import {NNModel} from "../../../src/AI/nn/NNModel";
import {nodeFileSystem} from "../../src/nodeIO";
import GameSystem from "../../../src/lib/GameSystem";
import Config from "../../../src/lib/data/Config";
import {executeNNTurn} from "../../../src/AI/TurnExecutor";
import {randomTurn} from "../../src/Opponents";
import {scorePlayer} from "../../src/GreedyAI";
import {MAX_TURNS, countNodes} from "../../src/trainUtils";

const modelDir = process.argv[2] || "public/model";
const numGames = parseInt(process.argv[3] || "15");

async function main() {
  await setupBackend();

  const model = new NNModel();
  await model.load(nodeFileSystem(modelDir));

  let wins = 0, losses = 0, draws = 0;

  for (let g = 0; g < numGames; g++) {
    const game = new GameSystem(Config);
    const nnIdx = g % 3;
    for (let turn = 0; turn < MAX_TURNS * 3 && !game.gameOver; turn++) {
      if (game.currentPlayerIndex === nnIdx) executeNNTurn(game, model);
      else randomTurn(game);
    }
    const scores = [0, 1, 2].map(i => scorePlayer(game, i));
    const nodes = game.players.map(p => countNodes(game, p));
    const winner = game.winner?.name ?? "draw";
    const posName = ["B", "R", "G"][nnIdx];
    const nnName = ["Blue", "Red", "Green"][nnIdx];

    const result = winner === "draw" ? "DRAW" : winner === nnName ? "WIN " : "LOSE";
    if (result === "WIN ") wins++;
    else if (result === "LOSE") losses++;
    else draws++;

    console.log(
      `  ${result} Game ${String(g + 1).padStart(3)}/${numGames} (${posName}): ${winner.padEnd(5)} T${String(game.turnCount).padEnd(3)} ` +
      `scores=[${scores.map(s => String(s.toFixed(0)).padStart(4)).join(",")}] ` +
      `nodes=[${nodes.map(n => String(n).padStart(2)).join(",")}]`
    );
  }

  console.log(`\n  Summary: ${wins}W ${losses}L ${draws}D out of ${numGames} (winrate ${(wins / numGames * 100).toFixed(0)}%)`);
  model.dispose();
}

main().catch(console.error);
