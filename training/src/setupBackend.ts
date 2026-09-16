/**
 * Initialize TF.js with WASM backend for Node.js training.
 *
 * WASM backend is ~2-10x faster than the default JS backend and
 * doesn't require native compilation like @tensorflow/tfjs-node.
 *
 * Must be called (and awaited) before any TF.js operations.
 */
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-wasm";
import {setWasmPaths} from "@tensorflow/tfjs-backend-wasm";
import * as path from "path";

let initialized = false;

export async function setupBackend(): Promise<void> {
  if (initialized) return;

  // Point to the WASM files in node_modules
  const wasmDir = path.resolve("node_modules/@tensorflow/tfjs-backend-wasm/dist/");
  setWasmPaths(wasmDir + "/");

  await tf.setBackend("wasm");
  await tf.ready();
  initialized = true;

  console.log(`TF.js backend: ${tf.getBackend()}`);
}
