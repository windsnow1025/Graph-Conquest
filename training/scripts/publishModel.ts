/**
 * Publish a trained model to public/model/ for the web UI.
 * The training pipeline only writes training/model/*; this is the single
 * place that writes the web model location: the TF.js layout (model.json +
 * weights.bin) for the Node scripts that read the published model, and
 * web.json for the page (see src/AI/nn/WebModel.ts).
 *
 * Usage: npx tsx training/scripts/publishModel.ts <phase1|phase2|phase3|modelDir>
 */
import * as fs from "fs";
import * as path from "path";
import {WEB_MODEL_FILE} from "../../src/AI/nn/WebModel";
import type {WebModelFile} from "../../src/AI/nn/WebModel";

const WEB_MODEL_DIR = path.resolve("public/model");

/** The parts of a TF.js layers-model model.json that the web model file carries. */
interface LayersModelJson {
  modelTopology: object;
  weightsManifest: {paths: string[]; weights: WebModelFile["weightSpecs"]}[];
}

function webModelFile(sourceDir: string): WebModelFile {
  const modelJson = JSON.parse(fs.readFileSync(path.join(sourceDir, "model.json"), "utf8")) as LayersModelJson;
  const groups = modelJson.weightsManifest;
  return {
    modelTopology: modelJson.modelTopology,
    weightSpecs: groups.flatMap(group => group.weights),
    weightData: Buffer.concat(groups.flatMap(group => group.paths.map(file => fs.readFileSync(path.join(sourceDir, file))))).toString("base64"),
  };
}

function main() {
  const arg = process.argv[2];
  if (!arg) {
    console.error("Usage: npx tsx training/scripts/publishModel.ts <phase1|phase2|phase3|modelDir>");
    process.exit(1);
  }
  const sourceDir = /^phase[123]$/.test(arg) ? path.resolve("training/model", arg) : path.resolve(arg);
  if (!fs.existsSync(path.join(sourceDir, "model.json"))) {
    console.error(`No model found at ${sourceDir}`);
    process.exit(1);
  }
  fs.mkdirSync(WEB_MODEL_DIR, {recursive: true});
  for (const file of fs.readdirSync(sourceDir)) {
    fs.copyFileSync(path.join(sourceDir, file), path.join(WEB_MODEL_DIR, file));
  }
  fs.writeFileSync(path.join(WEB_MODEL_DIR, WEB_MODEL_FILE), JSON.stringify(webModelFile(sourceDir)));
  console.log(`Published ${sourceDir} -> ${WEB_MODEL_DIR} (+ ${WEB_MODEL_FILE})`);
}

main();
