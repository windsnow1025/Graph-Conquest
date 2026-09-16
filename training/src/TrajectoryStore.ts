/**
 * Persistent trajectory store: the expensive simulation output is written once
 * at generation time (scripts/generateData.ts) and materialized into trainer
 * samples at consumption time (scripts/trainPhase*.ts), so datasets are reused
 * across training runs and re-weighted (λ) without re-simulation. Regenerate a
 * dataset only when its simulation inputs change: game rules, encoders, the
 * runners, or the weights of any model that played in it (all tracked in the
 * manifest).
 *
 * Layout: training/data/store/<name>/
 *   manifest.json      provenance and stats (DatasetManifest)
 *   trajectories.bin   per-game trajectories, format "traj" (vs-random, mixed)
 *   samples.bin        finished trainer samples, format "sample" (imitation,
 *                      which has no consumer-side transform)
 *
 * Formats are not versioned: when the layout changes, the magic changes and
 * stale datasets fail loudly and get regenerated.
 *
 * traj, little-endian, every block a multiple of 4 bytes so state vectors
 * can be viewed as Float32Array without copying:
 *   header (16 B): magic "STRATRAJ", stateDim u32, gameCount u32 (patched on
 *                  close)
 *   per game:
 *     meta (28 B): kind u8, nnIdx i8, opponentId u8, winnerIdx i8,
 *                  turnCount u16, pad u16, snapshotCount u32, recordCount u32,
 *                  outcomes f32[3]
 *     snapshots (12 + 4·stateDim each): playerIdx u8, pad[3], v f32,
 *                  recordFrom u32, state f32[stateDim]
 *     records (12 + 4·stateDim each): playerIdx u8, explored u8,
 *                  decisionType u8, pad, a f32, b f32, state f32[stateDim]
 *     terminal states: 3 × f32[stateDim] (their value labels are the outcomes)
 */
import type {RawRecord, TurnSnapshot, GameResult} from "./SelfPlay";
import {assignAdvantages, recordsToSamples, snapshotsToValueSamples} from "./SelfPlay";
import {emptySample} from "./SampleTypes";
import {STATE_SIZE} from "../../src/AI/nn/StateEncoder";
import {DATA_DIR, createSampleWriter} from "./trainUtils";
import * as fs from "fs";
import * as path from "path";

export const STORE_DIR = path.join(DATA_DIR, "store");

export const GAME_KIND = {vsRandom: 0, vsOpponent: 1, selfPlay: 2} as const;

const DECISION_TYPES: readonly string[] = [
  "army", "moveTarget", "recruit", "battleTarget", "battleSelect", "battleAllocate", "battleRetreat",
];

const MAGIC = "STRATRAJ";
const HEADER_SIZE = 16;
const META_SIZE = 28;
const ENTRY_SIZE = 12 + 4 * STATE_SIZE;

// Per decision type, the one or two numbers that recordsToSamples reads from
// the action object (see SelfPlay.recordsToSamples).
function encodeAction(decisionType: string, action: Record<string, number>): {a: number; b: number} {
  switch (decisionType) {
    case "army": return {a: action.actionType ?? 0, b: action.fraction ?? 0};
    case "moveTarget": return {a: action.moveTarget ?? -1, b: 0};
    case "recruit": return {a: action.recruitFraction ?? 0, b: 0};
    case "battleTarget": return {a: action.battleTarget ?? -1, b: 0};
    case "battleSelect": return {a: action.battleSelect ?? 0, b: 0};
    case "battleAllocate": return {a: action.killFraction ?? 0, b: 0};
    case "battleRetreat": return {a: action.battleRetreat ?? 0, b: 0};
    default: throw new Error(`Unknown decisionType: ${decisionType}`);
  }
}

function decodeAction(decisionType: string, a: number, b: number): Record<string, number> {
  switch (decisionType) {
    case "army": return {actionType: a, fraction: b};
    case "moveTarget": return {moveTarget: a};
    case "recruit": return {recruitFraction: a};
    case "battleTarget": return {battleTarget: a};
    case "battleSelect": return {battleSelect: a};
    case "battleAllocate": return {killFraction: a};
    case "battleRetreat": return {battleRetreat: a};
    default: throw new Error(`Unknown decisionType: ${decisionType}`);
  }
}

// ─── Manifest ───

export interface DatasetManifest {
  name: string;
  type: "vs-random" | "mixed" | "imitation";
  format: "traj" | "sample";
  createdAt: string;
  /** git HEAD at generation time; provenance only, not used for validation */
  simRev: string;
  params: Record<string, number>;
  /** model whose decisions were recorded; null when the generator is code (imitation) */
  model: {name: string; weightsMd5: string} | null;
  /** opponentId in each game frame indexes this list */
  opponents: {name: string; weightsMd5?: string}[];
  stateDim: number;
  stats: {
    games: number; records: number; snapshots: number; samples?: number;
    wins: number; losses: number; draws: number; avgTurns: number;
  };
}

export function datasetPath(name: string): string {
  return path.join(STORE_DIR, name);
}

export function writeManifest(dir: string, manifest: DatasetManifest): void {
  fs.writeFileSync(path.join(dir, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
}

export function readManifest(dir: string): DatasetManifest {
  const p = path.join(dir, "manifest.json");
  if (!fs.existsSync(p)) throw new Error(`No dataset manifest at ${p}`);
  return JSON.parse(fs.readFileSync(p, "utf8")) as DatasetManifest;
}

export function listDatasets(): string[] {
  if (!fs.existsSync(STORE_DIR)) return [];
  return fs.readdirSync(STORE_DIR).filter(n => fs.existsSync(path.join(STORE_DIR, n, "manifest.json")));
}

// ─── Writer ───

export interface PersistedGameMeta {
  kind: number;       // GAME_KIND
  nnIdx: number;      // recorded seat; -1 = all seats (self-play)
  opponentId: number; // index into manifest.opponents (self-play frames: 0, unused)
  winnerIdx: number;  // -1 = draw
  turnCount: number;
}

export function createTrajectoryWriter(filePath: string) {
  fs.mkdirSync(path.dirname(filePath), {recursive: true});
  const fd = fs.openSync(filePath, "w");
  const header = Buffer.alloc(HEADER_SIZE);
  header.write(MAGIC, 0, "ascii");
  header.writeUInt32LE(STATE_SIZE, 8);
  fs.writeSync(fd, header);
  let games = 0, records = 0, snapshots = 0;

  const writeState = (buf: Buffer, off: number, state: Float32Array): void => {
    if (state.length !== STATE_SIZE) throw new Error(`state length ${state.length} != ${STATE_SIZE}`);
    Buffer.from(state.buffer, state.byteOffset, 4 * STATE_SIZE).copy(buf, off);
  };

  return {
    writeGame(meta: PersistedGameMeta, result: GameResult, terminalStates: Float32Array[]): void {
      const {records: recs, snapshots: snaps, outcomes} = result;
      const buf = Buffer.alloc(META_SIZE + (snaps.length + recs.length) * ENTRY_SIZE + terminalStates.length * 4 * STATE_SIZE);
      buf.writeUInt8(meta.kind, 0);
      buf.writeInt8(meta.nnIdx, 1);
      buf.writeUInt8(meta.opponentId, 2);
      buf.writeInt8(meta.winnerIdx, 3);
      buf.writeUInt16LE(meta.turnCount, 4);
      buf.writeUInt32LE(snaps.length, 8);
      buf.writeUInt32LE(recs.length, 12);
      for (let p = 0; p < 3; p++) buf.writeFloatLE(outcomes[p], 16 + 4 * p);
      let off = META_SIZE;
      for (const s of snaps) {
        buf.writeUInt8(s.playerIdx, off);
        buf.writeFloatLE(s.v, off + 4);
        buf.writeUInt32LE(s.recordFrom, off + 8);
        writeState(buf, off + 12, s.state);
        off += ENTRY_SIZE;
      }
      for (const r of recs) {
        const dt = DECISION_TYPES.indexOf(r.decisionType);
        if (dt < 0) throw new Error(`Unknown decisionType: ${r.decisionType}`);
        const {a, b} = encodeAction(r.decisionType, r.action);
        buf.writeUInt8(r.playerIdx, off);
        buf.writeUInt8(r.explored ? 1 : 0, off + 1);
        buf.writeUInt8(dt, off + 2);
        buf.writeFloatLE(a, off + 4);
        buf.writeFloatLE(b, off + 8);
        writeState(buf, off + 12, r.state);
        off += ENTRY_SIZE;
      }
      for (const t of terminalStates) {
        writeState(buf, off, t);
        off += 4 * STATE_SIZE;
      }
      fs.writeSync(fd, buf);
      games++;
      records += recs.length;
      snapshots += snaps.length;
    },
    get stats() {
      return {games, records, snapshots};
    },
    close(): void {
      const count = Buffer.alloc(4);
      count.writeUInt32LE(games, 0);
      fs.writeSync(fd, count, 0, 4, 12);
      fs.closeSync(fd);
    },
  };
}

// ─── Reader ───

export interface PersistedGame extends PersistedGameMeta {
  outcomes: number[];
  snapshots: TurnSnapshot[];
  records: RawRecord[];
  terminalStates: Float32Array[];
}

/** Stream games one at a time; memory stays bounded by the largest game. */
export function readTrajectories(filePath: string, onGame: (game: PersistedGame, index: number) => void): {stateDim: number; gameCount: number} {
  const fd = fs.openSync(filePath, "r");
  try {
    const header = Buffer.alloc(HEADER_SIZE);
    if (fs.readSync(fd, header, 0, HEADER_SIZE, null) !== HEADER_SIZE) throw new Error(`${filePath}: truncated header`);
    if (header.toString("ascii", 0, 8) !== MAGIC) throw new Error(`${filePath}: bad magic (stale or foreign file; regenerate the dataset)`);
    const stateDim = header.readUInt32LE(8);
    if (stateDim !== STATE_SIZE) {
      throw new Error(`${filePath}: stateDim ${stateDim} != current STATE_SIZE ${STATE_SIZE} (encoder changed; regenerate the dataset)`);
    }
    const gameCount = header.readUInt32LE(12);
    const meta = Buffer.alloc(META_SIZE);

    for (let g = 0; g < gameCount; g++) {
      if (fs.readSync(fd, meta, 0, META_SIZE, null) !== META_SIZE) throw new Error(`${filePath}: truncated at game ${g}`);
      const snapCount = meta.readUInt32LE(8);
      const recCount = meta.readUInt32LE(12);
      const bodySize = (snapCount + recCount) * ENTRY_SIZE + 3 * 4 * STATE_SIZE;
      // allocUnsafeSlow: own ArrayBuffer at byteOffset 0, so 4-aligned offsets
      // can be viewed as Float32Array directly
      const body = Buffer.allocUnsafeSlow(bodySize);
      let got = 0;
      while (got < bodySize) {
        const n = fs.readSync(fd, body, got, bodySize - got, null);
        if (n <= 0) throw new Error(`${filePath}: truncated at game ${g} body`);
        got += n;
      }

      let off = 0;
      const snapshotsArr: TurnSnapshot[] = new Array(snapCount);
      for (let i = 0; i < snapCount; i++) {
        snapshotsArr[i] = {
          playerIdx: body.readUInt8(off),
          v: body.readFloatLE(off + 4),
          recordFrom: body.readUInt32LE(off + 8),
          state: new Float32Array(body.buffer, off + 12, STATE_SIZE),
        };
        off += ENTRY_SIZE;
      }
      const recordsArr: RawRecord[] = new Array(recCount);
      for (let i = 0; i < recCount; i++) {
        const decisionType = DECISION_TYPES[body.readUInt8(off + 2)];
        if (decisionType === undefined) throw new Error(`${filePath}: unknown decisionType id at game ${g}`);
        recordsArr[i] = {
          playerIdx: body.readUInt8(off),
          explored: body.readUInt8(off + 1) === 1,
          decisionType,
          action: decodeAction(decisionType, body.readFloatLE(off + 4), body.readFloatLE(off + 8)),
          advantage: 0,
          state: new Float32Array(body.buffer, off + 12, STATE_SIZE),
        };
        off += ENTRY_SIZE;
      }
      const terminalStates: Float32Array[] = [];
      for (let p = 0; p < 3; p++) {
        terminalStates.push(new Float32Array(body.buffer, off, STATE_SIZE));
        off += 4 * STATE_SIZE;
      }

      onGame({
        kind: meta.readUInt8(0),
        nnIdx: meta.readInt8(1),
        opponentId: meta.readUInt8(2),
        winnerIdx: meta.readInt8(3),
        turnCount: meta.readUInt16LE(4),
        outcomes: [meta.readFloatLE(16), meta.readFloatLE(20), meta.readFloatLE(24)],
        snapshots: snapshotsArr,
        records: recordsArr,
        terminalStates,
      }, g);
    }
    return {stateDim, gameCount};
  } finally {
    fs.closeSync(fd);
  }
}

// ─── Materialization ───

export interface MaterializeStats {
  games: number;
  raw: number;
  kept: number;
  posAdv: number;
  negAdv: number;
  wins: number;
  losses: number;
  draws: number;
}

/**
 * Convert a trajectory dataset into a trainer sample file, assigning TD(λ)
 * advantages at read time. Semantics are identical to the old inline pipeline:
 * policy records weighted by the non-negative advantage clamp, snapshot and
 * terminal value samples, ε-explored records value-only, then the w > 0 keep
 * filter. Decisive self-play games count as wins in the returned stats.
 */
export function materializeSamples(dir: string, outPath: string, tdLambda: number): MaterializeStats {
  const manifest = readManifest(dir);
  if (manifest.format !== "traj") {
    throw new Error(`Dataset ${manifest.name} has format ${manifest.format}; only traj datasets can be materialized (regenerate stale ones)`);
  }
  const writer = createSampleWriter(outPath);
  const stats: MaterializeStats = {games: 0, raw: 0, kept: 0, posAdv: 0, negAdv: 0, wins: 0, losses: 0, draws: 0};

  readTrajectories(path.join(dir, "trajectories.bin"), (game) => {
    assignAdvantages(game.records, game.snapshots, game.outcomes, tdLambda);
    const samples = recordsToSamples(game.records, game.outcomes);
    samples.push(...snapshotsToValueSamples(game.snapshots, game.outcomes));
    for (let p = 0; p < 3; p++) {
      const vs = emptySample(p);
      vs.state = game.terminalStates[p];
      vs.value = game.outcomes[p];
      samples.push(vs);
    }
    for (const r of game.records) {
      if (r.explored) continue;
      if (r.advantage > 0) stats.posAdv++;
      else if (r.advantage < 0) stats.negAdv++;
    }
    const kept = samples.filter(s => s.policyWeight > 0);
    writer.writeBatch(kept);
    stats.games++;
    stats.raw += samples.length;
    stats.kept += kept.length;
    if (game.winnerIdx < 0) stats.draws++;
    else if (game.kind === GAME_KIND.selfPlay || game.winnerIdx === game.nnIdx) stats.wins++;
    else stats.losses++;
  });

  writer.close();
  return stats;
}

/**
 * Materialization is deterministic in (dataset, λ), so repeated trainings on
 * the same pair reuse the sample file via a sidecar meta file.
 */
export function materializeCached(dir: string, outPath: string, tdLambda: number): {stats: MaterializeStats; cached: boolean} {
  const manifest = readManifest(dir);
  const metaPath = `${outPath}.meta.json`;
  if (fs.existsSync(metaPath) && fs.existsSync(outPath)) {
    const meta = JSON.parse(fs.readFileSync(metaPath, "utf8")) as
      {dataset: string; createdAt: string; tdLambda: number; stats: MaterializeStats};
    if (meta.dataset === manifest.name && meta.createdAt === manifest.createdAt && meta.tdLambda === tdLambda) {
      return {stats: meta.stats, cached: true};
    }
  }
  const stats = materializeSamples(dir, outPath, tdLambda);
  fs.writeFileSync(metaPath, JSON.stringify(
    {dataset: manifest.name, createdAt: manifest.createdAt, tdLambda, stats}, null, 2) + "\n");
  return {stats, cached: false};
}
