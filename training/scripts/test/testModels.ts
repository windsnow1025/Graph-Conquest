/**
 * Compare Phase 1, 2, and 3 models:
 *   1. Each vs Random (30 games)
 *   2. Three-way: P1 vs P2 vs P3 (30 games, rotate positions)
 * Usage: npx tsx training/scripts/test/testModels.ts
 */
import {setupBackend} from "../../src/setupBackend";
import {NNModel} from "../../../src/AI/nn/NNModel";
import {nodeFileSystem} from "../../src/nodeIO";
import {executeNNTurn} from "../../../src/AI/TurnExecutor";
import {scorePlayer} from "../../src/GreedyAI";
import {randomTurn} from "../../src/Opponents";
import {MODEL_DIR_PHASE1, MODEL_DIR_PHASE2, MODEL_DIR_PHASE3, MAX_TURNS, createRandomizedGame} from "../../src/trainUtils";

const VS_RANDOM_GAMES = 30;
const H2H_GAMES = 30;

async function testVsRandom(label: string, model: NNModel) {
  let wins = 0, losses = 0, draws = 0;
  let totalTurns = 0;
  const winTurns: number[] = [];

  for (let g = 0; g < VS_RANDOM_GAMES; g++) {
    const game = createRandomizedGame();
    const nnIdx = g % 3;
    for (let turn = 0; turn < MAX_TURNS * 3 && !game.gameOver; turn++) {
      if (game.currentPlayerIndex === nnIdx) executeNNTurn(game, model);
      else randomTurn(game);
    }
    const winner = game.winner?.name ?? "draw";
    const nnName = ["Blue", "Red", "Green"][nnIdx];
    if (winner === "draw") draws++;
    else if (winner === nnName) { wins++; winTurns.push(game.turnCount); }
    else losses++;
    totalTurns += game.turnCount;

    const posName = ["B", "R", "G"][nnIdx];
    const result = winner === "draw" ? "DRAW" : winner === nnName ? "WIN " : "LOSE";
    const scores = [0, 1, 2].map(i => scorePlayer(game, i));
    console.log(
      `  ${String(g + 1).padStart(2)}/${VS_RANDOM_GAMES} (${posName}) ${result}: ` +
      `${winner.padEnd(5)} T${String(game.turnCount).padEnd(3)} ` +
      `scores=[${scores.map(s => String(s.toFixed(0)).padStart(4)).join(",")}]`
    );
  }

  const avgTurns = (totalTurns / VS_RANDOM_GAMES).toFixed(1);
  const avgWinTurn = winTurns.length > 0
    ? (winTurns.reduce((a, b) => a + b, 0) / winTurns.length).toFixed(1)
    : "N/A";
  console.log(`\n  ${label}: ${wins} wins, ${losses} losses, ${draws} draws (${(wins / VS_RANDOM_GAMES * 100).toFixed(0)}% win rate) avg turn: ${avgTurns}, avg win turn: ${avgWinTurn}\n`);
}

async function testThreeWay(models: {label: string; model: NNModel}[]) {
  const winsBy = [0, 0, 0]; // indexed by model
  let draws = 0;

  // Rotate: each model gets each position equally
  const arrangements = [
    [0, 1, 2], // P1=Blue, P2=Red, P3=Green
    [1, 2, 0], // P2=Blue, P3=Red, P1=Green
    [2, 0, 1], // P3=Blue, P1=Red, P2=Green
  ];

  for (let g = 0; g < H2H_GAMES; g++) {
    const game = createRandomizedGame();
    const arr = arrangements[g % 3]; // arr[playerSlot] = modelIdx

    for (let turn = 0; turn < MAX_TURNS * 3 && !game.gameOver; turn++) {
      const modelIdx = arr[game.currentPlayerIndex];
      executeNNTurn(game, models[modelIdx].model);
    }

    const winner = game.winner?.name ?? "draw";
    if (winner === "draw") {
      draws++;
    } else {
      const winnerSlot = ["Blue", "Red", "Green"].indexOf(winner);
      winsBy[arr[winnerSlot]]++;
    }

    const scores = [0, 1, 2].map(i => scorePlayer(game, i));
    const labels = [0, 1, 2].map(slot => `${models[arr[slot]].label}=${["B","R","G"][slot]}`).join(" ");
    const result = winner === "draw" ? "DRAW " : `${models[arr[["Blue","Red","Green"].indexOf(winner)]].label}`;
    console.log(
      `  ${String(g + 1).padStart(2)}/${H2H_GAMES} (${labels}) ${result.padEnd(5)}: ` +
      `${winner.padEnd(5)} T${String(game.turnCount).padEnd(3)} ` +
      `scores=[${scores.map(s => String(s.toFixed(0)).padStart(4)).join(",")}]`
    );
  }

  console.log(`\n  Three-way: ${models.map((m, i) => `${m.label}=${winsBy[i]} wins`).join("  ")}  Draws=${draws} (${H2H_GAMES} games)\n`);
}

async function main() {
  await setupBackend();

  const dirs = [
    {label: "P1", dir: MODEL_DIR_PHASE1},
    {label: "P2", dir: MODEL_DIR_PHASE2},
    {label: "P3", dir: MODEL_DIR_PHASE3},
  ];

  const models: {label: string; model: NNModel}[] = [];
  for (const {label, dir} of dirs) {
    const model = new NNModel();
    await model.load(nodeFileSystem(dir));
    models.push({label, model});
  }

  for (const {label, model} of models) {
    console.log(`=== ${label} vs Random (${VS_RANDOM_GAMES} games) ===`);
    await testVsRandom(label, model);
  }

  console.log(`=== Three-way: P1 vs P2 vs P3 (${H2H_GAMES} games) ===`);
  await testThreeWay(models);

  for (const {model} of models) model.dispose();
}

main().catch(console.error);
