/**
 * End-to-end pipeline smoke test:
 *   1. Fresh model → NN-vs-random games with exploration → RL sample recording
 *   2. Greedy + DAgger games → imitation sample recording
 *   3. Validate sample invariants (finiteness, class indices within masks)
 *   4. Export binary → Python train 2 epochs → export TF.js
 *   5. Reload exported model in TS → play a full game
 *
 * Usage: npx tsx training/scripts/test/testPipeline.ts
 */
import {setupBackend} from "../../src/setupBackend";
import {NNModel, BATTLE_TARGET_DIM} from "../../../src/AI/nn/NNModel";
import {NUM_NODES} from "../../../src/AI/nn/StateEncoder";
import {nodeFileSystem} from "../../src/nodeIO";
import {executeNNTurn} from "../../../src/AI/TurnExecutor";
import {greedyTurn, daggerTurn} from "../../src/GreedyAI";
import {randomTurn} from "../../src/Opponents";
import {nnVsRandomGame, recordsToSamples, snapshotsToValueSamples, assignAdvantages} from "../../src/SelfPlay";
import {
  GAME_KIND, datasetPath, createTrajectoryWriter, writeManifest, materializeSamples,
} from "../../src/TrajectoryStore";
import {encodeState, STATE_SIZE} from "../../../src/AI/nn/StateEncoder";
import type {Sample} from "../../src/SampleTypes";
import {sampleToFloats, emptySample, SAMPLE_FLOATS} from "../../src/SampleTypes";
import {
  initLog, log, exportSamples, trainWithPython, createRandomizedGame, DATA_DIR,
} from "../../src/trainUtils";
import * as path from "path";
import * as fs from "fs";

const SMOKE_MODEL_DIR = path.resolve("training/model/smoke");

function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error(`ASSERT FAILED: ${msg}`);
}

function validateSamples(samples: Sample[], label: string): void {
  const counts = {
    action: 0, move: 0, btgt: 0, bsel: 0, kfr: 0, ret: 0, rec: 0, valueOnly: 0,
  };
  for (const s of samples) {
    const floats = sampleToFloats(s);
    for (let i = 0; i < floats.length; i++) {
      assert(Number.isFinite(floats[i]), `${label}: non-finite float at ${i}`);
    }
    if (s.actionTypeTarget >= 0) counts.action++;
    if (s.moveTargetIdx >= 0) {
      counts.move++;
      assert(s.moveTargetIdx < NUM_NODES, `${label}: moveTargetIdx ${s.moveTargetIdx} out of range`);
      assert(s.moveMask[s.moveTargetIdx] === 1, `${label}: moveTargetIdx ${s.moveTargetIdx} not in legal mask`);
    }
    if (s.battleTargetIdx >= 0) {
      counts.btgt++;
      assert(s.battleTargetIdx < BATTLE_TARGET_DIM, `${label}: battleTargetIdx ${s.battleTargetIdx} out of range`);
      assert(s.battleTargetMask[s.battleTargetIdx] === 1, `${label}: battleTargetIdx ${s.battleTargetIdx} not in mask`);
    }
    if (s.battleSelectMask === 1) counts.bsel++;
    if (s.killFracMask === 1) counts.kfr++;
    if (s.retreatMask === 1) counts.ret++;
    if (s.recruitMask === 1) counts.rec++;
    if (s.actionTypeTarget < 0 && s.moveTargetIdx < 0 && s.battleTargetIdx < 0
      && !s.battleSelectMask && !s.killFracMask && !s.retreatMask && !s.recruitMask
      && !s.splitMask && !s.disbandMask) counts.valueOnly++;
  }
  log(`  ${label}: ${samples.length} samples ` +
    `(act=${counts.action} mov=${counts.move} btgt=${counts.btgt} bsel=${counts.bsel} ` +
    `kfr=${counts.kfr} ret=${counts.ret} rec=${counts.rec} value-only=${counts.valueOnly})`);
}

async function main() {
  await setupBackend();
  initLog(""); // console-only: test output does not belong in the log dir
  log("=== Pipeline smoke test ===");

  const model = new NNModel();
  model.buildNew();

  const allSamples: Sample[] = [];

  // ── 1. NN vs random with exploration ──
  log("\n1. NN-vs-random recording (2 games, 12 rounds cap):");
  for (let g = 0; g < 2; g++) {
    const game = createRandomizedGame();
    const {records, outcomes} = nnVsRandomGame(game, model, g % 3, 12, 1.0, 0.2);
    const samples = recordsToSamples(records, outcomes);
    validateSamples(samples, `RL game ${g + 1}`);
    allSamples.push(...samples);
  }

  // ── 1b. Trajectory store roundtrip: persisted + materialized must equal in-memory ──
  log("\n1b. Trajectory store roundtrip (2 games, λ=0.8):");
  {
    const storeDir = datasetPath("pipeline-test");
    fs.rmSync(storeDir, {recursive: true, force: true});
    fs.mkdirSync(storeDir, {recursive: true});
    const writer = createTrajectoryWriter(path.join(storeDir, "trajectories.bin"));
    const reference: Sample[] = [];

    for (let g = 0; g < 2; g++) {
      const game = createRandomizedGame();
      const result = nnVsRandomGame(game, model, g % 3, 12, 1.0, 0.2);
      const winner = game.winner;
      const winnerIdx = winner ? game.players.findIndex(p => p.name === winner.name) : -1;
      const terminalStates = [0, 1, 2].map(pi => encodeState(game, pi));
      writer.writeGame(
        {kind: GAME_KIND.vsRandom, nnIdx: g % 3, opponentId: 0, winnerIdx, turnCount: game.turnCount},
        result, terminalStates,
      );

      assignAdvantages(result.records, result.snapshots, result.outcomes, 0.8);
      const samples = recordsToSamples(result.records, result.outcomes);
      samples.push(...snapshotsToValueSamples(result.snapshots, result.outcomes));
      for (let p = 0; p < 3; p++) {
        const vs = emptySample(p);
        vs.state = terminalStates[p];
        vs.value = result.outcomes[p];
        samples.push(vs);
      }
      reference.push(...samples.filter(s => s.policyWeight > 0));
    }
    const written = writer.stats;
    writer.close();

    writeManifest(storeDir, {
      name: "pipeline-test", type: "vs-random", format: "traj",
      createdAt: new Date().toISOString(), simRev: "test",
      params: {games: 2}, model: null, opponents: [{name: "random"}], stateDim: STATE_SIZE,
      stats: {games: written.games, records: written.records, snapshots: written.snapshots, wins: 0, losses: 0, draws: 0, avgTurns: 0},
    });

    const matFile = path.join(DATA_DIR, "pipeline-mat.bin");
    const stats = materializeSamples(storeDir, matFile, 0.8);
    assert(stats.kept === reference.length, `materialized ${stats.kept} != in-memory ${reference.length}`);

    const buf = fs.readFileSync(matFile);
    assert(buf.readUInt32LE(0) === reference.length, "materialized file count mismatch");
    for (let i = 0; i < reference.length; i++) {
      const ref = sampleToFloats(reference[i]);
      for (let j = 0; j < SAMPLE_FLOATS; j++) {
        const got = buf.readFloatLE(4 + (i * SAMPLE_FLOATS + j) * 4);
        assert(got === ref[j], `materialized sample ${i} float ${j}: ${got} != ${ref[j]}`);
      }
    }
    log(`  Roundtrip exact: ${stats.kept} samples (pos ${stats.posAdv} / neg ${stats.negAdv})`);
    fs.rmSync(storeDir, {recursive: true, force: true});
    fs.rmSync(matFile, {force: true});
  }

  // ── 2. Greedy recording (3 rounds cap) ──
  log("\n2. Greedy recording (1 game, 3 rounds cap):");
  {
    const game = createRandomizedGame();
    const samples: Sample[] = [];
    for (let t = 0; t < 9 && !game.gameOver; t++) {
      if (game.currentPlayerIndex === 0) greedyTurn(game, samples);
      else randomTurn(game);
    }
    validateSamples(samples, "Greedy");
    assert(samples.some(s => s.moveTargetIdx >= 0), "greedy produced no moveTarget samples");
    allSamples.push(...samples);
  }

  // ── 3. DAgger recording (2 rounds cap) ──
  log("\n3. DAgger recording (1 game, 2 rounds cap):");
  {
    const game = createRandomizedGame();
    const samples: Sample[] = [];
    for (let t = 0; t < 6 && !game.gameOver; t++) {
      if (game.currentPlayerIndex === 0) daggerTurn(game, model, samples);
      else randomTurn(game);
    }
    validateSamples(samples, "DAgger");
    allSamples.push(...samples);
  }

  model.dispose();

  // ── 4. Export + Python train ──
  for (const s of allSamples) s.policyWeight = 1; // format test, not a learning test
  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, {recursive: true});
  const dataFile = path.join(DATA_DIR, "pipeline.bin");
  exportSamples(allSamples, dataFile);
  log(`\n4. Exported ${allSamples.length} samples → ${dataFile}`);

  const ok = await trainWithPython(dataFile, SMOKE_MODEL_DIR, 2, allSamples.length, true);
  assert(ok, "python training failed");

  // ── 5. Reload exported model, play a full game ──
  log("\n5. Reload exported model and play one game:");
  const m2 = new NNModel();
  await m2.load(nodeFileSystem(SMOKE_MODEL_DIR));

  const game = createRandomizedGame();
  for (let turn = 0; turn < 40 * 3 && !game.gameOver; turn++) {
    if (game.currentPlayerIndex === 0) executeNNTurn(game, m2);
    else randomTurn(game);
  }
  log(`  Game finished: winner=${game.winner?.name ?? "none (turn cap)"} T${game.turnCount}`);

  m2.dispose();
  log("\n=== PASS ===");
}

main().catch(e => { console.error(e); process.exit(1); });
