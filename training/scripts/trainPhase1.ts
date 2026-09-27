/**
 * Phase 1 — Imitation learning on persisted imitation datasets (greedy-labeled
 * samples, see generateData.ts). This script does not simulate.
 *
 * Bootstraps a fresh model and trains it from scratch on the datasets'
 * samples.bin files (already in trainer format; imitation has no
 * consumer-side transform).
 * Output: training/model/phase1/
 *
 * Usage: npx tsx training/scripts/trainPhase1.ts <dataset> [dataset...]
 *   <dataset>: an imitation dataset from the trajectory store (a greedy one,
 *   optionally with stalemate ones)
 * Env: EPOCHS (default 50), ACTION_BALANCE (default 1: inverse class frequency
 *   power on the action-type loss, 0 = off)
 */
import {setupBackend} from "../src/setupBackend";
import * as fs from "fs";
import * as path from "path";
import {datasetPath, readManifest, listDatasets} from "../src/TrajectoryStore";
import {
  MODEL_DIR_PHASE1, LEARNING_RATE,
  initLog, log,
  trainWithPython, testNNvsRandom, bootstrapModel,
} from "../src/trainUtils";

// ─── Config ───

const EPOCHS = Number(process.env.EPOCHS ?? "50");
if (!Number.isFinite(EPOCHS) || EPOCHS < 1) {
  throw new Error(`Invalid EPOCHS env value: ${process.env.EPOCHS}`);
}
// MOVE is about 1 army label in 6 and DISBAND rarer still; unweighted training
// under-learns MOVE (recall 0.37 on the training set) and the model then never
// occupies the nodes it clears
const ACTION_BALANCE = Number(process.env.ACTION_BALANCE ?? "1");
if (!Number.isFinite(ACTION_BALANCE) || ACTION_BALANCE < 0) {
  throw new Error(`Invalid ACTION_BALANCE env value: ${process.env.ACTION_BALANCE}`);
}

// ─── Main ───

async function main() {
  await setupBackend();
  initLog("phase1.log");

  const datasetNames = process.argv.slice(2);
  if (datasetNames.length === 0) {
    log("Usage: npx tsx training/scripts/trainPhase1.ts <dataset> [dataset...]");
    log(`Available datasets: ${listDatasets().join(", ") || "(none; run generateData.ts first)"}`);
    return;
  }
  const dataFiles: string[] = [];
  let samples = 0;
  const lines: string[] = [];
  for (const datasetName of datasetNames) {
    const dataDir = datasetPath(datasetName);
    const manifest = readManifest(dataDir);
    if (manifest.type !== "imitation") {
      log(`ERROR: dataset ${datasetName} has type ${manifest.type}; phase 1 consumes imitation datasets.`);
      return;
    }
    const dataFile = path.join(dataDir, "samples.bin");
    if (!fs.existsSync(dataFile)) {
      log(`ERROR: ${dataFile} not found.`);
      return;
    }
    dataFiles.push(dataFile);
    samples += manifest.stats.samples ?? 0;
    const ds = manifest.stats;
    lines.push(`Dataset ${datasetName}: ${ds.games} games, W/L/D ${ds.wins}/${ds.losses}/${ds.draws}, avg turns ${ds.avgTurns}, ${ds.samples ?? 0} samples, simRev ${manifest.simRev}`);
  }

  await bootstrapModel(MODEL_DIR_PHASE1);

  log(`\n=== Phase 1: Imitation (datasets ${datasetNames.join(" + ")}, ${EPOCHS} epochs) ===`);
  for (const line of lines) log(line);

  log("\nBaseline (before training):");
  await testNNvsRandom(MODEL_DIR_PHASE1);

  const ok = await trainWithPython("imitation", dataFiles, MODEL_DIR_PHASE1, EPOCHS, samples, true, ACTION_BALANCE, LEARNING_RATE, null);
  if (!ok) {
    log("Training failed.");
    return;
  }

  log("\nAfter imitation:");
  await testNNvsRandom(MODEL_DIR_PHASE1);

  log("\n=== Phase 1 Done ===");
}

main().catch(console.error);
