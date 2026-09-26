/**
 * Measure (or read back) the unified 81-game baseline of a model directory:
 * the instrument of the phase gates (trainUtils.baselineEval, argmax play vs
 * Random on randomized configs), cached in <modelDir>/eval81.json by weights md5.
 *
 * Usage: npx tsx training/scripts/test/evalBaseline.ts <phase1|phase2|phase3|modelDir>
 */
import {setupBackend} from "../../src/setupBackend";
import {initLog, baselineEval} from "../../src/trainUtils";
import * as path from "path";

async function main() {
  const arg = process.argv[2];
  if (!arg) {
    console.error("Usage: npx tsx training/scripts/test/evalBaseline.ts <phase1|phase2|phase3|modelDir>");
    process.exit(1);
  }
  const modelDir = /^phase[123]$/.test(arg) ? path.resolve("training/model", arg) : path.resolve(arg);
  await setupBackend();
  initLog(""); // console-only
  await baselineEval(modelDir);
}

main().catch(e => { console.error(e); process.exit(1); });
