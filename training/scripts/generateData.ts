/**
 * Dataset generator: the only place simulation data is produced. Datasets are
 * persisted to the trajectory store (training/data/store/<name>/) and consumed
 * by the training phases; regenerate one only when its simulation inputs
 * change (game rules, encoders, runners, or the models that played).
 *
 * Usage:
 *   npx tsx training/scripts/generateData.ts vs-random --out <name> [--games 500]
 *       [--model phase1] [--temperature 1.0] [--epsilon 0.1]
 *     NN (recorded) vs 2 Random seats, rotating seat; phase 2 training data.
 *
 *   npx tsx training/scripts/generateData.ts mixed --out <name> [--games-opp 100]
 *       [--games-self 100] [--model phase2] [--temperature 1.0] [--epsilon 0.1]
 *     Part A: NN vs passive/phase1/phase2 rotation; Part B: 3-way self-play;
 *     phase 3 training data.
 *
 *   npx tsx training/scripts/generateData.ts imitation --out <name>
 *       [--passive 10] [--random 70] [--greedy 10] [--dagger 10]
 *     Greedy-labeled imitation samples (finished samples, no consumer-side
 *     transform); phase 1 training data.
 *
 *   --force overwrites an existing dataset of the same name.
 */
import {setupBackend} from "../src/setupBackend";
import {NNModel} from "../../src/AI/nn/NNModel";
import {nodeFileSystem} from "../src/nodeIO";
import GameSystem from "../../src/lib/GameSystem";
import {encodeState} from "../../src/AI/nn/StateEncoder";
import {executeNNTurn} from "../../src/AI/TurnExecutor";
import {passiveTurn, randomTurn} from "../src/Opponents";
import {greedyTurn, daggerTurn, scorePlayer} from "../src/GreedyAI";
import {emptySample, terminalValue} from "../src/SampleTypes";
import type {Sample} from "../src/SampleTypes";
import {nnVsRandomGame, nnVsOpponentGame, nnSelfPlayGame} from "../src/SelfPlay";
import type {GameResult, OpponentFn} from "../src/SelfPlay";
import {
  GAME_KIND, datasetPath, createTrajectoryWriter, writeManifest,
} from "../src/TrajectoryStore";
import type {DatasetManifest, PersistedGameMeta} from "../src/TrajectoryStore";
import {
  MAX_TURNS, initLog, log, logRaw, countNodes, createSampleWriter, createRandomizedGame, weightsMd5,
} from "../src/trainUtils";
import {execSync} from "child_process";
import * as fs from "fs";
import * as path from "path";

// ─── CLI ───

function usage(): never {
  console.error([
    "Usage:",
    "  generateData.ts vs-random --out <name> [--games 500] [--model phase1] [--temperature 1.0] [--epsilon 0.1]",
    "  generateData.ts mixed --out <name> [--games-opp 100] [--games-self 100] [--model phase2] [--temperature 1.0] [--epsilon 0.1]",
    "  generateData.ts imitation --out <name> [--passive 10] [--random 70] [--greedy 10] [--dagger 10]",
    "  --force to overwrite an existing dataset",
  ].join("\n"));
  process.exit(1);
}

function parseArgs(argv: string[]): {type: string; opts: Map<string, string>; force: boolean} {
  const type = argv[2];
  if (!type) usage();
  const opts = new Map<string, string>();
  let force = false;
  for (let i = 3; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith("--")) usage();
    if (arg === "--force") { force = true; continue; }
    const value = argv[++i];
    if (value === undefined) usage();
    opts.set(arg.slice(2), value);
  }
  return {type, opts, force};
}

function numOpt(opts: Map<string, string>, key: string, dflt: number): number {
  const raw = opts.get(key);
  if (raw === undefined) return dflt;
  const n = Number(raw);
  if (!Number.isFinite(n) || n < 0) throw new Error(`Invalid --${key}: ${raw}`);
  return n;
}

function gitRev(): string {
  try {
    return execSync("git rev-parse --short HEAD", {stdio: ["ignore", "pipe", "ignore"]}).toString().trim();
  } catch {
    return "unknown";
  }
}

function winnerIndex(game: GameSystem): number {
  const winner = game.winner;
  if (!winner) return -1;
  return game.players.findIndex(p => p.name === winner.name);
}

function logGame(prefix: string, g: number, total: number, game: GameSystem, recordCount: number, elapsed: string): void {
  if ((g + 1) % 10 !== 0) return;
  const winner = game.winner?.name ?? "DRAW";
  const scores = [0, 1, 2].map(i => scorePlayer(game, i));
  const nodes = game.players.map(p => countNodes(game, p));
  logRaw(
    `  ${prefix} ${String(g + 1).padStart(4)}/${total} ${winner.padEnd(5)} T${String(game.turnCount).padEnd(3)} ` +
    `scores=[${scores.map(s => String(s.toFixed(0)).padStart(4)).join(",")}] ` +
    `nodes=[${nodes.map(n => String(n).padStart(2)).join(",")}] ` +
    `${String(recordCount).padStart(5)}r ${elapsed}s`
  );
}

// ─── Shared generation loop ───

interface GameStats {wins: number; losses: number; draws: number; turnsSum: number}

function recordOutcome(stats: GameStats, winnerIdx: number, nnIdx: number, selfPlay: boolean, turnCount: number): void {
  if (winnerIdx < 0) stats.draws++;
  else if (selfPlay || winnerIdx === nnIdx) stats.wins++;
  else stats.losses++;
  stats.turnsSum += turnCount;
}

async function loadModel(name: string): Promise<{model: NNModel; dir: string}> {
  const dir = path.resolve("training/model", name);
  if (!fs.existsSync(path.join(dir, "model.json"))) {
    throw new Error(`Model ${name} not found at ${dir}`);
  }
  const model = new NNModel();
  await model.load(nodeFileSystem(dir));
  return {model, dir};
}

// ─── vs-random ───

async function generateVsRandom(name: string, dir: string, opts: Map<string, string>): Promise<DatasetManifest> {
  const modelName = opts.get("model") ?? "phase1";
  const games = numOpt(opts, "games", 500);
  const temperature = numOpt(opts, "temperature", 1.0);
  const epsilon = numOpt(opts, "epsilon", 0.1);

  const {model, dir: modelDir} = await loadModel(modelName);
  const writer = createTrajectoryWriter(path.join(dir, "trajectories.bin"));
  const stats: GameStats = {wins: 0, losses: 0, draws: 0, turnsSum: 0};

  log(`\nGenerating ${games} NN(${modelName})-vs-Random games (T=${temperature}, ε=${epsilon})`);
  for (let g = 0; g < games; g++) {
    const game = createRandomizedGame();
    const nnIdx = g % 3;
    const t0 = Date.now();
    const result = nnVsRandomGame(game, model, nnIdx, MAX_TURNS, temperature, epsilon);
    writeGameFrame(writer, game, result, {kind: GAME_KIND.vsRandom, nnIdx, opponentId: 0});
    recordOutcome(stats, winnerIndex(game), nnIdx, false, game.turnCount);
    logGame("[vsRand]", g, games, game, result.records.length, ((Date.now() - t0) / 1000).toFixed(1));
  }
  model.dispose();

  const w = writer.stats;
  writer.close();
  return {
    name, type: "vs-random", format: "traj", createdAt: new Date().toISOString(), simRev: gitRev(),
    params: {games, temperature, epsilon, maxTurns: MAX_TURNS},
    model: {name: modelName, weightsMd5: weightsMd5(modelDir)},
    opponents: [{name: "random"}],
    stateDim: encodeState(createRandomizedGame(), 0).length,
    stats: {games: w.games, records: w.records, snapshots: w.snapshots, ...finishStats(stats, w.games)},
  };
}

// ─── mixed (phase 3 diet) ───

async function generateMixed(name: string, dir: string, opts: Map<string, string>): Promise<DatasetManifest> {
  const modelName = opts.get("model") ?? "phase2";
  const gamesOpp = numOpt(opts, "games-opp", 100);
  const gamesSelf = numOpt(opts, "games-self", 100);
  const temperature = numOpt(opts, "temperature", 1.0);
  const epsilon = numOpt(opts, "epsilon", 0.1);

  const {model, dir: modelDir} = await loadModel(modelName);
  const {model: phase1Opp, dir: phase1Dir} = await loadModel("phase1");
  const {model: phase2Opp, dir: phase2Dir} = await loadModel("phase2");

  const opponents: {name: string; fn: OpponentFn; weightsMd5?: string}[] = [
    {name: "passive", fn: passiveTurn},
    {name: "phase1", fn: (game) => executeNNTurn(game, phase1Opp), weightsMd5: weightsMd5(phase1Dir)},
    {name: "phase2", fn: (game) => executeNNTurn(game, phase2Opp), weightsMd5: weightsMd5(phase2Dir)},
  ];

  const writer = createTrajectoryWriter(path.join(dir, "trajectories.bin"));
  const stats: GameStats = {wins: 0, losses: 0, draws: 0, turnsSum: 0};

  log(`\nPart A: ${gamesOpp} NN(${modelName}) games vs opponent rotation (T=${temperature}, ε=${epsilon})`);
  for (let g = 0; g < gamesOpp; g++) {
    const game = createRandomizedGame();
    const nnIdx = g % 3;
    // Rotate the opponent every 3 games so every seat faces every opponent
    const oppId = Math.floor(g / 3) % opponents.length;
    const t0 = Date.now();
    const result = nnVsOpponentGame(game, model, nnIdx, opponents[oppId].fn, MAX_TURNS, temperature, epsilon);
    writeGameFrame(writer, game, result, {kind: GAME_KIND.vsOpponent, nnIdx, opponentId: oppId});
    recordOutcome(stats, winnerIndex(game), nnIdx, false, game.turnCount);
    logGame(`[${opponents[oppId].name.padEnd(7)}]`, g, gamesOpp, game, result.records.length, ((Date.now() - t0) / 1000).toFixed(1));
  }

  log(`\nPart B: ${gamesSelf} self-play games`);
  for (let g = 0; g < gamesSelf; g++) {
    const game = createRandomizedGame();
    const t0 = Date.now();
    const result = nnSelfPlayGame(game, model, MAX_TURNS, temperature, epsilon);
    writeGameFrame(writer, game, result, {kind: GAME_KIND.selfPlay, nnIdx: -1, opponentId: 0});
    recordOutcome(stats, winnerIndex(game), -1, true, game.turnCount);
    logGame("[self   ]", g, gamesSelf, game, result.records.length, ((Date.now() - t0) / 1000).toFixed(1));
  }

  model.dispose();
  phase1Opp.dispose();
  phase2Opp.dispose();

  const w = writer.stats;
  writer.close();
  return {
    name, type: "mixed", format: "traj", createdAt: new Date().toISOString(), simRev: gitRev(),
    params: {gamesOpp, gamesSelf, temperature, epsilon, maxTurns: MAX_TURNS},
    model: {name: modelName, weightsMd5: weightsMd5(modelDir)},
    opponents: opponents.map(o => (o.weightsMd5 ? {name: o.name, weightsMd5: o.weightsMd5} : {name: o.name})),
    stateDim: encodeState(createRandomizedGame(), 0).length,
    stats: {games: w.games, records: w.records, snapshots: w.snapshots, ...finishStats(stats, w.games)},
  };
}

function writeGameFrame(
  writer: ReturnType<typeof createTrajectoryWriter>, game: GameSystem, result: GameResult,
  meta: Omit<PersistedGameMeta, "winnerIdx" | "turnCount">,
): void {
  const terminalStates = [0, 1, 2].map(pi => encodeState(game, pi));
  writer.writeGame({...meta, winnerIdx: winnerIndex(game), turnCount: game.turnCount}, result, terminalStates);
}

// ─── imitation (phase 1 diet) ───

async function generateImitation(name: string, dir: string, opts: Map<string, string>): Promise<DatasetManifest> {
  const counts = {
    passive: numOpt(opts, "passive", 10),
    random: numOpt(opts, "random", 70),
    greedy: numOpt(opts, "greedy", 10),
    dagger: numOpt(opts, "dagger", 10),
  };
  type OpponentType = keyof typeof counts;
  const schedule: OpponentType[] = [];
  for (const type of Object.keys(counts) as OpponentType[]) {
    for (let i = 0; i < counts[type]; i++) schedule.push(type);
  }

  const writer = createSampleWriter(path.join(dir, "samples.bin"));
  const stats: GameStats = {wins: 0, losses: 0, draws: 0, turnsSum: 0};
  // DAgger uses a fresh random-weight model, matching the phase 1 bootstrap start
  const daggerModel = new NNModel();
  daggerModel.buildNew();

  log(`\nGenerating imitation data: ${schedule.length} games (${counts.passive} vs passive, ${counts.random} vs random, ${counts.greedy} vs greedy, ${counts.dagger} DAgger)`);
  for (let g = 0; g < schedule.length; g++) {
    const opType = schedule[g];
    const game = createRandomizedGame();
    const greedyIdx = g % 3;
    const gameSamples: Sample[] = [];
    const t0 = Date.now();

    const playTurn = (): void => {
      if (opType === "dagger") {
        if (game.currentPlayerIndex === greedyIdx) daggerTurn(game, daggerModel, gameSamples);
        else randomTurn(game);
      } else if (opType === "greedy") {
        if (game.currentPlayerIndex === greedyIdx) greedyTurn(game, gameSamples);
        else greedyTurn(game);
      } else {
        if (game.currentPlayerIndex === greedyIdx) greedyTurn(game, gameSamples);
        else if (opType === "passive") passiveTurn(game);
        else randomTurn(game);
      }
    };
    for (let turn = 0; turn < MAX_TURNS * 3 && !game.gameOver; turn++) playTurn();

    const winner = game.winner?.name ?? "draw";
    // Overwrite value with the game outcome from each sample's own perspective
    // (battle samples include defender-perspective states)
    for (const s of gameSamples) {
      s.value = terminalValue(winner, game.players[s.playerIdx]);
    }
    for (let pi = 0; pi < 3; pi++) {
      const vs = emptySample(pi);
      vs.state = encodeState(game, pi);
      vs.value = terminalValue(winner, game.players[pi]);
      gameSamples.push(vs);
    }
    writer.writeBatch(gameSamples);

    recordOutcome(stats, winnerIndex(game), greedyIdx, false, game.turnCount);
    logGame(`[${opType.padEnd(7)}]`, g, schedule.length, game, gameSamples.length, ((Date.now() - t0) / 1000).toFixed(1));
  }
  daggerModel.dispose();

  const samples = writer.count;
  writer.close();
  return {
    name, type: "imitation", format: "sample", createdAt: new Date().toISOString(), simRev: gitRev(),
    params: {...counts, maxTurns: MAX_TURNS},
    model: null,
    opponents: [{name: "passive"}, {name: "random"}, {name: "greedy"}],
    stateDim: encodeState(createRandomizedGame(), 0).length,
    stats: {games: schedule.length, records: 0, snapshots: 0, samples, ...finishStats(stats, schedule.length)},
  };
}

function finishStats(stats: GameStats, games: number): {wins: number; losses: number; draws: number; avgTurns: number} {
  return {
    wins: stats.wins, losses: stats.losses, draws: stats.draws,
    avgTurns: games > 0 ? Number((stats.turnsSum / games).toFixed(1)) : 0,
  };
}

// ─── Main ───

async function main() {
  const {type, opts, force} = parseArgs(process.argv);
  const name = opts.get("out");
  if (!name || !/^[\w.-]+$/.test(name)) usage();

  await setupBackend();
  initLog("generate.log");

  const dir = datasetPath(name);
  if (fs.existsSync(path.join(dir, "manifest.json"))) {
    if (!force) {
      log(`ERROR: dataset ${name} already exists at ${dir}; pass --force to overwrite.`);
      process.exitCode = 1;
      return;
    }
    log(`Overwriting existing dataset ${name} (--force)`);
    fs.rmSync(dir, {recursive: true, force: true});
  }
  fs.mkdirSync(dir, {recursive: true});

  log(`=== Generate dataset: ${name} (${type}) ===`);
  const t0 = Date.now();
  let manifest: DatasetManifest;
  if (type === "vs-random") manifest = await generateVsRandom(name, dir, opts);
  else if (type === "mixed") manifest = await generateMixed(name, dir, opts);
  else if (type === "imitation") manifest = await generateImitation(name, dir, opts);
  else usage();

  writeManifest(dir, manifest);
  const s = manifest.stats;
  const sizeMB = fs.readdirSync(dir)
    .reduce((sum, f) => sum + fs.statSync(path.join(dir, f)).size, 0) / (1024 * 1024);
  log(`\nDataset ${name} written: ${s.games} games, W/L/D ${s.wins}/${s.losses}/${s.draws}, avg turns ${s.avgTurns}, ` +
    `${s.records} records, ${s.snapshots} snapshots${s.samples !== undefined ? `, ${s.samples} samples` : ""}, ` +
    `${sizeMB.toFixed(0)} MB, ${((Date.now() - t0) / 60000).toFixed(1)} min`);
  log(`=== Done: ${dir} ===`);
}

main().catch(e => { console.error(e); process.exit(1); });
