/**
 * Test: NN vs Greedy (+ 1 Random).
 * Usage: npx tsx training/testNNvsGreedy.ts [modelDir] [numGames]
 */
import {setupBackend} from "../../src/setupBackend";
import {NNModel} from "../../../src/AI/nn/NNModel";
import {nodeFileSystem} from "../../src/nodeIO";
import GameSystem from "../../../src/lib/GameSystem";
import Config from "../../../src/lib/data/Config";
import {executeNNTurn} from "../../../src/AI/TurnExecutor";
import {randomTurn} from "../../src/Opponents";
import {greedyTurn, scorePlayer} from "../../src/GreedyAI";
import {MAX_TURNS, countNodes} from "../../src/trainUtils";

const modelDir = process.argv[2] || "training/model-phase1";
const numGames = parseInt(process.argv[3] || "9");

async function main() {
  await setupBackend();

  const model = new NNModel();
  await model.load(nodeFileSystem(modelDir));

  let wins = 0, losses = 0, draws = 0;

  for (let g = 0; g < numGames; g++) {
    const game = new GameSystem(Config);
    // Rotate: NN and Greedy swap positions, 3rd player is random
    const nnIdx = g % 3;
    const greedyIdx = (g + 1) % 3;
    const t0 = Date.now();

    for (let turn = 0; turn < MAX_TURNS * 3 && !game.gameOver; turn++) {
      if (game.currentPlayerIndex === nnIdx) executeNNTurn(game, model);
      else if (game.currentPlayerIndex === greedyIdx) greedyTurn(game);
      else randomTurn(game);
    }

    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    const scores = [0, 1, 2].map(i => scorePlayer(game, i));
    const nodes = game.players.map(p => countNodes(game, p));
    const winner = game.winner?.name ?? "draw";
    const nnName = ["Blue", "Red", "Green"][nnIdx];
    const nnPos = ["B", "R", "G"][nnIdx];
    const greedyPos = ["B", "R", "G"][greedyIdx];

    let result: string;
    if (winner === "draw") { result = "DRAW"; draws++; }
    else if (winner === nnName) { result = "WIN "; wins++; }
    else { result = "LOSE"; losses++; }

    console.log(
      `  ${result} Game ${String(g + 1).padStart(2)}/${numGames} NN(${nnPos}) G(${greedyPos}): ` +
      `${winner.padEnd(5)} T${String(game.turnCount).padEnd(3)} ` +
      `scores=[${scores.map(s => String(s.toFixed(0)).padStart(4)).join(",")}] ` +
      `nodes=[${nodes.map(n => String(n).padStart(2)).join(",")}] ${elapsed}s`
    );
  }

  console.log(`\n  Summary: ${wins}W ${losses}L ${draws}D out of ${numGames} (winrate ${(wins / numGames * 100).toFixed(0)}%)`);
  model.dispose();
}

main().catch(console.error);
