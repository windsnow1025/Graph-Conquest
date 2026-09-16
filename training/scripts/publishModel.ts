/**
 * Publish a trained model to public/model/ for the web UI.
 * The training pipeline only writes training/model/*; this is the single
 * place that writes the web model location.
 *
 * Usage: npx tsx training/scripts/publishModel.ts <phase1|phase2|phase3|modelDir>
 */
import * as fs from "fs";
import * as path from "path";

const WEB_MODEL_DIR = path.resolve("public/model");

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
  console.log(`Published ${sourceDir} -> ${WEB_MODEL_DIR}`);
}

main();
