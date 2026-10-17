/**
 * Teacher competence in the NN's late-game states: play NN (deterministic) vs
 * Random; when a game is still open at the handoff turn, the greedy AI takes
 * over the NN's seat and plays on. Reports whether and how fast greedy
 * finishes from the states the NN stalls in.
 *
 * Usage: npx tsx training/scripts/test/testHandoff.ts [modelDir] [games] [handoffTurn]
 */
import {setupBackend} from "../../src/setupBackend";
import {NNModel} from "../../../src/AI/nn/NNModel";
import {nodeFileSystem} from "../../src/nodeIO";
import {executeNNTurn} from "../../../src/AI/TurnExecutor";
import {greedyTurn} from "../../src/GreedyAI";
import {randomTurn} from "../../src/Opponents";
import {MAX_TURNS, MODEL_DIR_PHASE1, countNodes, createRandomizedGame} from "../../src/trainUtils";
import * as path from "path";

const modelDir = process.argv[2] ? path.resolve(process.argv[2]) : MODEL_DIR_PHASE1;
const numGames = parseInt(process.argv[3] || "27");
const handoffTurn = parseInt(process.argv[4] || "40");

async function main() {
  await setupBackend();
  const model = new NNModel();
  await model.load(nodeFileSystem(modelDir));

  let nnWins = 0, handoffs = 0, greedyWins = 0, greedyDraws = 0;
  for (let g = 0; g < numGames; g++) {
    const api = createRandomizedGame();
    const nnIdx = g % 3;
    const name = api.getState().players[nnIdx].name;
    let handedOff = false;
    const t0 = Date.now();
    for (let turn = 0; turn < MAX_TURNS * 3 && !api.isGameOver(); turn++) {
      const state = api.getState();
      if (state.currentPlayerIndex !== nnIdx) { randomTurn(api); continue; }
      if (!handedOff && state.turnCount >= handoffTurn) {
        handedOff = true;
        handoffs++;
        const self = state.players[nnIdx];
        console.log(`Game ${g + 1}/${numGames} (${name}): handoff at T${state.turnCount}, units=${self.armies.reduce((s, a) => s + a.units.length, 0)} nodes=${countNodes(state, self)} money=${self.money}`);
      }
      if (handedOff) greedyTurn(api); else executeNNTurn(api, model);
    }
    const winner = api.getWinner() ?? "draw";
    const won = winner === name;
    if (!handedOff) { if (won) nnWins++; }
    else if (won) greedyWins++; else greedyDraws++;
    console.log(`Game ${g + 1}/${numGames} (${name}): ${winner} T${api.getState().turnCount}${handedOff ? " after handoff" : ""} ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  }
  console.log(`\nSummary: NN finished ${nnWins} of ${numGames - handoffs} games before T${handoffTurn}; after ${handoffs} handoffs greedy won ${greedyWins} and drew ${greedyDraws}`);
  model.dispose();
}

main().catch(e => { console.error(e); process.exit(1); });
