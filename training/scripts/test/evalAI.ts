/**
 * AI Strength Evaluation Framework
 *
 * Scores AI models on a 0-1 scale by running them against baseline opponents.
 *
 * Baselines:
 *   - Random: picks random legal actions each turn
 *   - Passive: does nothing (only endTurn)
 *
 * Usage:
 *   npx tsx training/scripts/test/evalAI.ts                          # Eval current model
 *   npx tsx training/scripts/test/evalAI.ts --model public/model-old # Eval a specific model
 *   npx tsx training/scripts/test/evalAI.ts --games 20               # More games per matchup
 *   npx tsx training/scripts/test/evalAI.ts --compare public/model-old-v2-backup  # Compare two models
 */
import {setupBackend} from "../../src/setupBackend";
import type {GameData} from "../../../src/lib/GameAPI";
import DefaultConfig from "../../../src/lib/data/DefaultConfig";
import {executeNNTurn} from "../../../src/AI/TurnExecutor";
import {NNModel} from "../../../src/AI/nn/NNModel";
import {nodeFileSystem} from "../../src/nodeIO";
import {passiveTurn, randomTurn} from "../../src/Opponents";
import GameAPI from "../../../src/lib/GameAPI";
import * as path from "path";

// ─── Types ───

type OpponentType = "random" | "passive";

interface GameResult {
  winnerIdx: number;
  turns: number;
  finalNodes: [number, number, number];
  finalDefeated: [boolean, boolean, boolean];
}

interface MatchupResult {
  opponent: OpponentType;
  games: number;
  wins: number;
  losses: number;
  draws: number;
  winRate: number;
  avgTurns: number;
  avgNodeControl: number;
  avgEnemiesDefeated: number;
  decisiveRate: number;
}

interface EvalReport {
  modelPath: string;
  matchups: MatchupResult[];
  compositeScore: number;
  breakdown: {
    vsRandom: number;
    vsPassive: number;
    territoryScore: number;
    eliminationScore: number;
  };
}

// ─── Config ───

const MAX_TURNS = 100;

// ─── Opponent turn execution ───

// ─── Game simulation ───

function countNodes(state: GameData, playerIdx: number): number {
  let count = 0;
  const player = state.players[playerIdx];
  for (const [, owner] of state.nodeOwnership) {
    if (owner === player) count++;
  }
  return count;
}

function runGame(
  model: NNModel,
  nnPlayerIdx: number,
  opponentType: OpponentType,
): GameResult {
  const api = GameAPI.create(DefaultConfig);

  for (let turn = 0; turn < MAX_TURNS * 3 && !api.isGameOver(); turn++) {
    const playerIdx = api.getState().currentPlayerIndex;

    if (playerIdx === nnPlayerIdx) {
      executeNNTurn(api, model);
    } else {
      switch (opponentType) {
        case "random": randomTurn(api); break;
        case "passive": passiveTurn(api); break;
      }
    }
  }

  const state = api.getState();
  const winner = api.getWinner();
  const winnerIdx = winner === null ? -1 : state.players.findIndex(p => p.name === winner);

  return {
    winnerIdx,
    turns: state.turnCount,
    finalNodes: [countNodes(state, 0), countNodes(state, 1), countNodes(state, 2)],
    finalDefeated: [state.players[0].defeated, state.players[1].defeated, state.players[2].defeated],
  };
}

// ─── Matchup evaluation ───

function evaluateMatchup(
  model: NNModel,
  opponent: OpponentType,
  numGames: number,
): MatchupResult {
  let wins = 0, losses = 0, draws = 0;
  let totalTurns = 0;
  let totalNodeFraction = 0;
  let totalEnemiesDefeated = 0;

  for (let g = 0; g < numGames; g++) {
    const nnPlayerIdx = g % 3;
    const result = runGame(model, nnPlayerIdx, opponent);

    totalTurns += result.turns;
    totalNodeFraction += result.finalNodes[nnPlayerIdx] / 16;

    let enemiesDefeated = 0;
    for (let i = 0; i < 3; i++) {
      if (i !== nnPlayerIdx && result.finalDefeated[i]) enemiesDefeated++;
    }
    totalEnemiesDefeated += enemiesDefeated;

    if (result.winnerIdx === nnPlayerIdx) {
      wins++;
    } else if (result.winnerIdx >= 0) {
      losses++;
    } else {
      draws++;
    }
  }

  const decisive = wins + losses;

  return {
    opponent,
    games: numGames,
    wins,
    losses,
    draws,
    winRate: numGames > 0 ? wins / numGames : 0,
    avgTurns: numGames > 0 ? totalTurns / numGames : 0,
    avgNodeControl: numGames > 0 ? totalNodeFraction / numGames : 0,
    avgEnemiesDefeated: numGames > 0 ? totalEnemiesDefeated / numGames : 0,
    decisiveRate: numGames > 0 ? decisive / numGames : 0,
  };
}

// ─── Composite scoring ───

function computeCompositeScore(matchups: MatchupResult[]): EvalReport["breakdown"] & {composite: number} {
  const vsRandom = matchups.find(m => m.opponent === "random");
  const vsPassive = matchups.find(m => m.opponent === "passive");

  const randomScore = vsRandom ? vsRandom.winRate : 0;
  const passiveScore = vsPassive ? vsPassive.winRate : 0;

  const allMatchups = matchups.filter(Boolean);
  const territoryScore = allMatchups.length > 0
    ? allMatchups.reduce((s, m) => s + m.avgNodeControl, 0) / allMatchups.length
    : 0;

  const eliminationScore = allMatchups.length > 0
    ? allMatchups.reduce((s, m) => s + m.avgEnemiesDefeated / 2, 0) / allMatchups.length
    : 0;

  const composite = Math.min(1, Math.max(0,
    0.20 * passiveScore +
    0.30 * randomScore +
    0.30 * territoryScore +
    0.20 * eliminationScore
  ));

  return {
    vsRandom: randomScore,
    vsPassive: passiveScore,
    territoryScore,
    eliminationScore,
    composite,
  };
}

// ─── Full evaluation ───

function evaluateModel(model: NNModel, modelPath: string, gamesPerMatchup: number): EvalReport {
  const opponents: OpponentType[] = ["passive", "random"];
  const matchups: MatchupResult[] = [];

  for (const opp of opponents) {
    console.log(`  vs ${opp} (${gamesPerMatchup} games)...`);
    const startTime = Date.now();
    const result = evaluateMatchup(model, opp, gamesPerMatchup);
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

    console.log(
      `    W:${result.wins} L:${result.losses} D:${result.draws} ` +
      `(${(result.winRate * 100).toFixed(0)}% win) ` +
      `nodes=${(result.avgNodeControl * 100).toFixed(0)}% ` +
      `elim=${result.avgEnemiesDefeated.toFixed(1)}/2 ` +
      `${elapsed}s`
    );

    matchups.push(result);
  }

  const scores = computeCompositeScore(matchups);

  return {
    modelPath,
    matchups,
    compositeScore: scores.composite,
    breakdown: {
      vsRandom: scores.vsRandom,
      vsPassive: scores.vsPassive,
      territoryScore: scores.territoryScore,
      eliminationScore: scores.eliminationScore,
    },
  };
}

function printReport(report: EvalReport): void {
  console.log(`\n${"═".repeat(55)}`);
  console.log(`  Model: ${report.modelPath}`);
  console.log(`  Composite Score: ${(report.compositeScore * 100).toFixed(1)} / 100`);
  console.log(`${"═".repeat(55)}`);
  console.log(`  vs Passive:   ${(report.breakdown.vsPassive * 100).toFixed(0)}% win`);
  console.log(`  vs Random:    ${(report.breakdown.vsRandom * 100).toFixed(0)}% win`);
  console.log(`  Territory:    ${(report.breakdown.territoryScore * 100).toFixed(0)}%`);
  console.log(`  Elimination:  ${(report.breakdown.eliminationScore * 100).toFixed(0)}%`);
  console.log(`${"═".repeat(55)}`);
}

// ─── CLI ───

function parseArgs() {
  const args = process.argv.slice(2);
  let modelPath = "public/model";
  let comparePath: string | null = null;
  let gamesPerMatchup = 12;

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "--model":
        modelPath = args[++i];
        break;
      case "--compare":
        comparePath = args[++i];
        break;
      case "--games":
        gamesPerMatchup = parseInt(args[++i]);
        break;
    }
  }

  return {modelPath, comparePath, gamesPerMatchup};
}

async function loadModel(modelPath: string): Promise<NNModel> {
  const model = new NNModel();
  const absPath = path.resolve(modelPath);
  await model.load(nodeFileSystem(absPath));
  return model;
}

async function main() {
  await setupBackend();

  const {modelPath, comparePath, gamesPerMatchup} = parseArgs();

  console.log("=== AI Strength Evaluation ===\n");

  console.log(`Evaluating: ${modelPath}`);
  const model = await loadModel(modelPath);
  const report = evaluateModel(model, modelPath, gamesPerMatchup);
  printReport(report);
  model.dispose();

  if (comparePath) {
    console.log(`\nEvaluating: ${comparePath}`);
    try {
      const model2 = await loadModel(comparePath);
      const report2 = evaluateModel(model2, comparePath, gamesPerMatchup);
      printReport(report2);
      model2.dispose();

      const diff = report.compositeScore - report2.compositeScore;
      console.log(`\n${"═".repeat(55)}`);
      console.log(`  COMPARISON`);
      console.log(`${"═".repeat(55)}`);
      console.log(`  ${modelPath}: ${(report.compositeScore * 100).toFixed(1)}`);
      console.log(`  ${comparePath}: ${(report2.compositeScore * 100).toFixed(1)}`);
      console.log(`  Delta: ${diff > 0 ? "+" : ""}${(diff * 100).toFixed(1)} points`);
      console.log(`  Winner: ${diff > 0 ? modelPath : diff < 0 ? comparePath : "TIE"}`);
      console.log(`${"═".repeat(55)}`);
    } catch (e) {
      console.error(`Failed to load comparison model: ${e}`);
    }
  }
}

main().catch(console.error);
