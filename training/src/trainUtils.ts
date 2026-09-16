/**
 * Shared utilities for training phases.
 */
import {NNModel} from "../../src/AI/nn/NNModel";
import {nodeFileSystem} from "./nodeIO";
import GameSystem from "../../src/lib/GameSystem";
import Graph from "../../src/lib/Graph";
import Player from "../../src/lib/Player";
import Config from "../../src/lib/data/Config";
import type {GameConfig} from "../../src/lib/Config";
import {UNIT_TYPES} from "../../src/AI/nn/StateEncoder";
import {executeNNTurn} from "../../src/AI/TurnExecutor";
import {randomTurn} from "./Opponents";
import {scorePlayer} from "./GreedyAI";
import type {Sample} from "./SampleTypes";
import {SAMPLE_FLOATS, sampleToFloats} from "./SampleTypes";
import {spawn} from "child_process";
import * as crypto from "crypto";
import * as fs from "fs";
import * as path from "path";

// ─── Paths ───

export const MODEL_DIR_PHASE1 = path.resolve("training/model/phase1");
export const MODEL_DIR_PHASE2 = path.resolve("training/model/phase2");
export const MODEL_DIR_PHASE3 = path.resolve("training/model/phase3");
export const LOG_DIR = path.resolve("training/log");
export const DATA_DIR = path.resolve("training/data");

// ─── Config ───

export const MAX_TURNS = 100;

// ─── Config variance ───

function randomBetween(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

function randomizeStat(base: number, fraction: number, min: number = 1): number {
  const lo = Math.max(min, Math.round(base * (1 - fraction)));
  const hi = Math.round(base * (1 + fraction));
  return lo + Math.floor(Math.random() * (hi - lo + 1));
}

export function createRandomizedConfig(): GameConfig {
  const V = 0.25;

  // Unit stats
  const newStats: Record<string, {attack: number; defend: number; health: number; range: number; speed: number; cost: number}> = {};
  for (const typeName of UNIT_TYPES) {
    const orig = Config.unitStatsMap[typeName];
    newStats[typeName] = {
      attack: randomizeStat(orig.attack, V),
      defend: randomizeStat(orig.defend, V),
      health: randomizeStat(orig.health, V, 5),
      range: orig.range,
      speed: orig.speed,
      cost: randomizeStat(orig.cost, V),
    };
  }

  // Node income
  const graphJSON = Config.gameMap.toJSON();
  for (const node of graphJSON.nodes) {
    node.data.income = randomizeStat(node.data.income, V);
  }

  // Players
  const baseMoney = 100;
  const players = Config.players.map(p =>
    new Player(randomizeStat(baseMoney, V), p.name, p.homeLocation)
  );

  return {
    ...Config,
    gameMap: Graph.fromJSON(graphJSON),
    unitStatsMap: newStats,
    interestRate: randomBetween(Config.interestRate * (1 - V), Config.interestRate * (1 + V)),
    upkeepRate: randomBetween(Config.upkeepRate * (1 - V), Config.upkeepRate * (1 + V)),
    players,
    neutralGarrison: { unitType: "Infantry", unitStats: newStats.Infantry },
  };
}

export function createRandomizedGame(): GameSystem {
  return new GameSystem(createRandomizedConfig());
}
export const BATCH_SIZE = 64;
export const LEARNING_RATE = 0.001;
export const TEST_GAMES = 9;

// ─── Logging ───

let logFile = "";

/** Empty filename = console-only (tests and ad-hoc evals keep the log dir clean). */
export function initLog(filename: string) {
  if (!filename) {
    logFile = "";
    return;
  }
  logFile = path.resolve(LOG_DIR, filename);
  if (!fs.existsSync(LOG_DIR)) fs.mkdirSync(LOG_DIR, {recursive: true});
  fs.writeFileSync(logFile, "");
}

export function log(msg: string) {
  const d = new Date();
  const ts = `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`;
  const trimmed = msg.replace(/^\n/, "");
  const prefix = msg.startsWith("\n") ? "\n" : "";
  console.log(msg);
  if (logFile) fs.appendFileSync(logFile, `${prefix}[${ts}] ${trimmed}\n`);
}

export function logRaw(msg: string) {
  console.log(msg);
  if (logFile) fs.appendFileSync(logFile, msg + "\n");
}

// ─── Export samples ───

export function exportSamples(samples: Sample[], outPath: string): void {
  const writer = createSampleWriter(outPath);
  writer.writeBatch(samples);
  writer.close();
}

export function createSampleWriter(outPath: string) {
  const fd = fs.openSync(outPath, "w");
  // Write placeholder header (count=0), will be updated on close
  const headerBuf = Buffer.alloc(4);
  fs.writeSync(fd, headerBuf);
  let totalCount = 0;
  const singleBuf = Buffer.alloc(SAMPLE_FLOATS * 4);

  return {
    writeBatch(samples: Sample[]): void {
      for (const s of samples) {
        const floats = sampleToFloats(s);
        Buffer.from(floats.buffer).copy(singleBuf);
        fs.writeSync(fd, singleBuf);
      }
      totalCount += samples.length;
    },
    get count() { return totalCount; },
    close(): void {
      // Rewrite header with final count
      headerBuf.writeUInt32LE(totalCount, 0);
      fs.writeSync(fd, headerBuf, 0, 4, 0);
      fs.closeSync(fd);
    },
  };
}

// ─── Python training ───

export async function trainWithPython(
  dataFile: string, modelDir: string, epochs: number, numSamples: number, fresh: boolean,
): Promise<boolean> {
  const pythonDir = path.resolve("training/python");
  const trainArgs = [
    "run", "python", "-m", "app.scripts.train",
    "--data", dataFile, "--model", modelDir,
    "--epochs", String(epochs), "--batch-size", String(BATCH_SIZE), "--lr", String(LEARNING_RATE),
    ...(fresh ? ["--fresh"] : []),
  ];
  log(`\nTraining (${numSamples} samples, ${epochs} epochs${fresh ? ", fresh" : ""}):`);
  return new Promise<boolean>((resolve) => {
    const proc = spawn("uv", trainArgs, {cwd: pythonDir, stdio: ["ignore", "pipe", "pipe"]});
    let lastLoss = 0;
    let buf = "";
    proc.stdout.on("data", (chunk: Buffer) => {
      buf += chunk.toString();
      const lines = buf.split("\n");
      buf = lines.pop()!;
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith("epoch ") || trimmed.startsWith("ratio:") || trimmed.startsWith("Exported")) {
          logRaw(`  ${trimmed}`);
        } else if (trimmed.startsWith("battle:")) {
          logRaw(`                 ${trimmed}`);
        }
        const m = trimmed.match(/loss=(-?[\d.]+)/);
        if (m) lastLoss = parseFloat(m[1]);
      }
    });
    let stderrBuf = "";
    proc.stderr.on("data", (chunk: Buffer) => { stderrBuf += chunk.toString(); });
    proc.on("close", (code) => {
      if (buf.trim()) {
        const m = buf.trim().match(/loss=(-?[\d.]+)/);
        if (m) lastLoss = parseFloat(m[1]);
      }
      if (code !== 0) {
        log(`  Training failed (code ${code}):\n${stderrBuf}`);
        resolve(false);
      } else {
        log(`  Training done, final loss=${lastLoss.toFixed(4)}`);
        resolve(true);
      }
    });
  });
}

// ─── Shared game stats ───

export function countNodes(game: GameSystem, player: Player): number {
  return [...game.nodeOwnership.values()].filter(o => o === player).length;
}

export function copyModelDir(src: string, dest: string): void {
  fs.mkdirSync(dest, {recursive: true});
  for (const file of fs.readdirSync(src)) {
    fs.copyFileSync(path.join(src, file), path.join(dest, file));
  }
}

export function weightsMd5(modelDir: string): string {
  const data = fs.readFileSync(path.join(modelDir, "weights.bin"));
  return crypto.createHash("md5").update(data).digest("hex");
}

// ─── Test NN vs random ───

export interface EvalMetrics {
  wins: number;
  losses: number;
  draws: number;
  /** Average turnCount of won games; Infinity when there are no wins. */
  avgWinTurn: number;
}

export async function testNNvsRandom(modelDir: string, games: number = TEST_GAMES): Promise<EvalMetrics> {
  log(`\nTesting NN vs random (${games} games)...`);
  const testModel = new NNModel();
  await testModel.load(nodeFileSystem(modelDir));

  let wins = 0, losses = 0, draws = 0, winTurnSum = 0;
  for (let g = 0; g < games; g++) {
    const game = createRandomizedGame();
    const nnIdx = g % 3;
    for (let turn = 0; turn < MAX_TURNS * 3 && !game.gameOver; turn++) {
      if (game.currentPlayerIndex === nnIdx) executeNNTurn(game, testModel);
      else randomTurn(game);
    }
    const scores = [0, 1, 2].map(i => scorePlayer(game, i));
    const units = game.players.map(p => p.armies.reduce((s, a) => s + a.units.length, 0));
    const nodes = game.players.map(p => countNodes(game, p));
    const winner = game.winner?.name ?? "draw";
    const nnName = ["Blue", "Red", "Green"][nnIdx];
    if (winner === "draw") draws++;
    else if (winner === nnName) { wins++; winTurnSum += game.turnCount; }
    else losses++;
    const posName = ["B", "R", "G"][nnIdx];
    logRaw(
      `  Test ${g + 1}/${games} (${posName}): ${winner.padEnd(5)} T${String(game.turnCount).padEnd(3)} ` +
      `scores=[${scores.map(s => String(s.toFixed(0)).padStart(4)).join(",")}] ` +
      `nodes=[${nodes.map(n => String(n).padStart(2)).join(",")}] ` +
      `units=[${units.map(u => String(u).padStart(3)).join(",")}]`
    );
  }

  testModel.dispose();
  const avgWinTurn = wins > 0 ? winTurnSum / wins : Infinity;
  log(`  Eval: ${wins}W ${losses}L ${draws}D, avg win turn ${wins > 0 ? avgWinTurn.toFixed(1) : "-"}`);
  return {wins, losses, draws, avgWinTurn};
}

// ─── Unified baseline ───

export const BASELINE_GAMES = 81;

interface CachedEval extends EvalMetrics {
  weightsMd5: string;
  games: number;
  createdAt: string;
}

/**
 * The 81-game baseline of a model, measured once and cached in
 * <modelDir>/eval81.json keyed by the weights md5, so every training run and
 * sweep cell is judged against the same number instead of a fresh noisy
 * 27-game read (win counts of the same model fluctuated 21W to 27W at 27
 * games). copyModelDir carries the cache along with the weights.
 */
export async function baselineEval(modelDir: string): Promise<EvalMetrics> {
  const cachePath = path.join(modelDir, "eval81.json");
  const md5 = weightsMd5(modelDir);
  if (fs.existsSync(cachePath)) {
    const cached = JSON.parse(fs.readFileSync(cachePath, "utf8")) as CachedEval;
    if (cached.weightsMd5 === md5 && cached.games === BASELINE_GAMES) {
      log(`  Baseline (cached, ${cached.games} games, ${cached.createdAt}): ` +
        `${cached.wins}W ${cached.losses}L ${cached.draws}D, avg win turn ${cached.wins > 0 ? cached.avgWinTurn.toFixed(2) : "-"}`);
      return {wins: cached.wins, losses: cached.losses, draws: cached.draws, avgWinTurn: cached.avgWinTurn};
    }
  }
  const m = await testNNvsRandom(modelDir, BASELINE_GAMES);
  const cached: CachedEval = {...m, weightsMd5: md5, games: BASELINE_GAMES, createdAt: new Date().toISOString()};
  fs.writeFileSync(cachePath, JSON.stringify(cached, null, 2) + "\n");
  return m;
}

// ─── Bootstrap ───

export async function bootstrapModel(modelDir: string): Promise<void> {
  if (!fs.existsSync(modelDir)) fs.mkdirSync(modelDir, {recursive: true});
  log("Bootstrapping fresh model...");
  const model = new NNModel();
  model.buildNew();
  await model.save(nodeFileSystem(modelDir));
  model.dispose();
  log("Bootstrap done.");
}
