/**
 * Test a trained NN vs Greedy.
 * Usage: npx tsx training/scripts/test/testVsGreedy.ts [modelDir]
 */
import {setupBackend} from "../../src/setupBackend";
import {NNModel} from "../../../src/AI/nn/NNModel";
import {nodeFileSystem} from "../../src/nodeIO";
import {executeNNTurn} from "../../../src/AI/TurnExecutor";
import GameAPI from "../../../src/lib/GameAPI";
import DefaultConfig from "../../../src/lib/data/DefaultConfig";
import {greedyTurn, scorePlayer, constantsOf} from "../../src/GreedyAI";
import {MODEL_DIR_PHASE2, MAX_TURNS} from "../../src/trainUtils";

const GAMES = 30;
const modelDir = process.argv[2] || MODEL_DIR_PHASE2;

async function main() {
  await setupBackend();
  const model = new NNModel();
  await model.load(nodeFileSystem(modelDir));

  let nnWins = 0, greedyWins = 0, draws = 0;

  for (let g = 0; g < GAMES; g++) {
    const api = GameAPI.create(DefaultConfig);
    const nnIdx = g % 3;
    const greedyIdx = (g + 1) % 3;

    for (let turn = 0; turn < MAX_TURNS * 3 && !api.isGameOver(); turn++) {
      const playerIdx = api.getState().currentPlayerIndex;
      if (playerIdx === nnIdx) executeNNTurn(api, model);
      else if (playerIdx === greedyIdx) greedyTurn(api);
      else greedyTurn(api);  // 3rd player also greedy
    }

    const state = api.getState();
    const winner = api.getWinner() ?? "draw";
    const nnName = state.players[nnIdx].name;
    if (winner === "draw") draws++;
    else if (winner === nnName) nnWins++;
    else greedyWins++;

    const constants = constantsOf(state);
    const scores = [0, 1, 2].map(i => scorePlayer(api, constants, i));
    const result = winner === "draw" ? "DRAW" : winner === nnName ? "NN  " : "GRD ";
    console.log(
      `  ${String(g + 1).padStart(2)}/${GAMES} (NN=${["B","R","G"][nnIdx]}) ${result}: ` +
      `${winner.padEnd(5)} T${String(state.turnCount).padEnd(3)} ` +
      `scores=[${scores.map(s => String(s.toFixed(0)).padStart(4)).join(",")}]`
    );
  }

  console.log(`\n  NN: ${nnWins} wins  Greedy: ${greedyWins} wins  Draws: ${draws} (${GAMES} games)\n`);
  model.dispose();
}

main().catch(console.error);
