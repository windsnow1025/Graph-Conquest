/**
 * Evaluate value head accuracy: how well does pred.value predict final outcome?
 *
 * Runs NN vs Random games, records every pred.value, compares to the training
 * target terminalValue (win=1, loss/eliminated=0, draw=1/3).
 * Reports: MSE, correlation, and pred.value distribution by early/mid/late game.
 *
 * Usage: npx tsx training/scripts/test/testValue.ts
 */
import {setupBackend} from "../../src/setupBackend";
import {NNModel} from "../../../src/AI/nn/NNModel";
import {nodeFileSystem} from "../../src/nodeIO";
import {executeNNTurn} from "../../../src/AI/TurnExecutor";
import type {TurnOptions, DecisionInfo} from "../../../src/AI/TurnExecutor";
import GameSystem from "../../../src/lib/GameSystem";
import Config from "../../../src/lib/data/Config";
import {encodeState} from "../../../src/AI/nn/StateEncoder";
import {randomTurn} from "../../src/Opponents";
import {terminalValue, DRAW_VALUE} from "../../src/SampleTypes";
import {MODEL_DIR_PHASE1, MODEL_DIR_PHASE2, MAX_TURNS} from "../../src/trainUtils";

const NUM_GAMES = 30;

interface ValueRecord {
  turnCount: number;
  predValue: number;
  finalOutcome: number;
}

async function evaluateValueHead(label: string, modelDir: string) {
  const model = new NNModel();
  await model.load(nodeFileSystem(modelDir));

  const allRecords: ValueRecord[] = [];
  let wins = 0, losses = 0, draws = 0;

  for (let g = 0; g < NUM_GAMES; g++) {
    const game = new GameSystem(Config);
    const nnIdx = g % 3;
    // Record value predictions for all 3 players each turn
    const perPlayerRecords: { turnCount: number; predValue: number; playerIdx: number }[] = [];

    const opts: TurnOptions = {
      onDecision: (info: DecisionInfo) => {
        // Record NN player's own prediction
        perPlayerRecords.push({
          turnCount: game.turnCount,
          predValue: info.pred.value,
          playerIdx: nnIdx,
        });
        // Also predict from other 2 players' perspectives
        for (let pi = 0; pi < 3; pi++) {
          if (pi === nnIdx) continue;
          const state = encodeState(game, pi);
          const pred = model.predict(state);
          perPlayerRecords.push({
            turnCount: game.turnCount,
            predValue: pred.value,
            playerIdx: pi,
          });
        }
      },
    };

    for (let turn = 0; turn < MAX_TURNS * 3 && !game.gameOver; turn++) {
      if (game.currentPlayerIndex === nnIdx) executeNNTurn(game, model, opts);
      else randomTurn(game);
    }

    const winner = game.winner?.name ?? "draw";
    const nnName = ["Blue", "Red", "Green"][nnIdx];
    if (winner === "draw") draws++;
    else if (winner === nnName) wins++;
    else losses++;

    for (const rec of perPlayerRecords) {
      const outcome = terminalValue(winner, game.players[rec.playerIdx]);
      allRecords.push({ turnCount: rec.turnCount, predValue: rec.predValue, finalOutcome: outcome });
    }
  }

  // Overall stats
  const n = allRecords.length;
  let mse = 0, sumPred = 0, sumActual = 0, sumPred2 = 0, sumActual2 = 0, sumProdPA = 0;
  for (const r of allRecords) {
    const err = r.predValue - r.finalOutcome;
    mse += err * err;
    sumPred += r.predValue;
    sumActual += r.finalOutcome;
    sumPred2 += r.predValue * r.predValue;
    sumActual2 += r.finalOutcome * r.finalOutcome;
    sumProdPA += r.predValue * r.finalOutcome;
  }
  mse /= n;
  const meanPred = sumPred / n;
  const meanActual = sumActual / n;
  const covPA = sumProdPA / n - meanPred * meanActual;
  const stdPred = Math.sqrt(sumPred2 / n - meanPred * meanPred);
  const stdActual = Math.sqrt(sumActual2 / n - meanActual * meanActual);
  const correlation = stdPred > 0 && stdActual > 0 ? covPA / (stdPred * stdActual) : 0;

  console.log(`\n=== ${label} Value Head Analysis ===`);
  console.log(`Games: ${NUM_GAMES} (${wins}W ${losses}L ${draws}D), Decisions: ${n}`);
  console.log(`MSE: ${mse.toFixed(6)}`);
  console.log(`Correlation(pred, actual): ${correlation.toFixed(4)}`);
  console.log(`Mean pred: ${meanPred.toFixed(4)}, Mean actual: ${meanActual.toFixed(4)}`);

  // Breakdown by game phase
  const phases = [
    { name: "Early (T1-10)", min: 1, max: 10 },
    { name: "Mid   (T11-50)", min: 11, max: 50 },
    { name: "Late  (T51-100)", min: 51, max: 100 },
  ];
  for (const phase of phases) {
    const subset = allRecords.filter(r => r.turnCount >= phase.min && r.turnCount <= phase.max);
    if (subset.length === 0) { console.log(`  ${phase.name}: no data`); continue; }
    let pMse = 0, pSum = 0, pActSum = 0;
    for (const r of subset) {
      pMse += (r.predValue - r.finalOutcome) ** 2;
      pSum += r.predValue;
      pActSum += r.finalOutcome;
    }
    console.log(
      `  ${phase.name}: n=${String(subset.length).padStart(5)}, ` +
      `MSE=${(pMse / subset.length).toFixed(6)}, ` +
      `avgPred=${(pSum / subset.length).toFixed(4)}, ` +
      `avgActual=${(pActSum / subset.length).toFixed(4)}`
    );
  }

  // Breakdown by outcome
  for (const outcome of ["win", "loss", "draw"] as const) {
    const target = outcome === "win" ? 1 : outcome === "loss" ? 0 : DRAW_VALUE;
    const subset = allRecords.filter(r => r.finalOutcome === target);
    if (subset.length === 0) { console.log(`  ${outcome}: no data`); continue; }
    let pSum = 0;
    for (const r of subset) pSum += r.predValue;
    console.log(
      `  ${outcome.padEnd(4)} games: n=${String(subset.length).padStart(5)}, ` +
      `avgPred=${(pSum / subset.length).toFixed(4)}, ` +
      `avgActual=${(subset.reduce((s, r) => s + r.finalOutcome, 0) / subset.length).toFixed(4)}`
    );
  }

  model.dispose();
}

async function main() {
  await setupBackend();
  await evaluateValueHead("Phase 1", MODEL_DIR_PHASE1);
  await evaluateValueHead("Phase 2", MODEL_DIR_PHASE2);
}

main().catch(console.error);
