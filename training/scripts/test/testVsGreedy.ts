/**
 * Test a trained NN vs Greedy.
 * Usage: npx tsx training/scripts/test/testVsGreedy.ts [modelDir]
 */
import {setupBackend} from "../../src/setupBackend";
import {NNModel} from "../../../src/AI/nn/NNModel";
import {nodeFileSystem} from "../../src/nodeIO";
import {executeNNTurn} from "../../../src/AI/TurnExecutor";
import GameSystem from "../../../src/lib/GameSystem";
import Config from "../../../src/lib/data/Config";
import {greedyTurn, scorePlayer} from "../../src/GreedyAI";
import {MODEL_DIR_PHASE2, MAX_TURNS} from "../../src/trainUtils";

const GAMES = 30;
const modelDir = process.argv[2] || MODEL_DIR_PHASE2;

async function main() {
  await setupBackend();
  const model = new NNModel();
  await model.load(nodeFileSystem(modelDir));

  let nnWins = 0, greedyWins = 0, draws = 0;

  for (let g = 0; g < GAMES; g++) {
    const game = new GameSystem(Config);
    const nnIdx = g % 3;
    const greedyIdx = (g + 1) % 3;

    for (let turn = 0; turn < MAX_TURNS * 3 && !game.gameOver; turn++) {
      if (game.currentPlayerIndex === nnIdx) executeNNTurn(game, model);
      else if (game.currentPlayerIndex === greedyIdx) greedyTurn(game);
      else greedyTurn(game);  // 3rd player also greedy
    }

    const winner = game.winner?.name ?? "draw";
    const nnName = game.players[nnIdx].name;
    if (winner === "draw") draws++;
    else if (winner === nnName) nnWins++;
    else greedyWins++;

    const scores = [0, 1, 2].map(i => scorePlayer(game, i));
    const result = winner === "draw" ? "DRAW" : winner === nnName ? "NN  " : "GRD ";
    console.log(
      `  ${String(g + 1).padStart(2)}/${GAMES} (NN=${["B","R","G"][nnIdx]}) ${result}: ` +
      `${winner.padEnd(5)} T${String(game.turnCount).padEnd(3)} ` +
      `scores=[${scores.map(s => String(s.toFixed(0)).padStart(4)).join(",")}]`
    );
  }

  console.log(`\n  NN: ${nnWins} wins  Greedy: ${greedyWins} wins  Draws: ${draws} (${GAMES} games)\n`);
  model.dispose();
}

main().catch(console.error);
