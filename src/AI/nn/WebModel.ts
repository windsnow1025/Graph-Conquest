/**
 * The web model file, public/model/web.json: a TF.js layers model in one JSON
 * document with its weights as base64, so the page loads the model with a
 * single JSON request and never requests a .bin file (a download manager on
 * the user's machine took over the weights.bin response and left the page an
 * empty body). publishModel.ts writes it beside the TF.js layout (model.json +
 * weights.bin) that the Node scripts read.
 */
import * as tf from "@tensorflow/tfjs";

export const WEB_MODEL_FILE = "web.json";

export interface WebModelFile {
  modelTopology: object;
  weightSpecs: tf.io.WeightsManifestEntry[];
  /** The weight buffers of the manifest, concatenated in order, base64 */
  weightData: string;
}

/** An IO handler serving the file's model to NNModel.load. */
export function webModelHandler(file: WebModelFile): tf.io.IOHandler {
  return tf.io.fromMemory({
    modelTopology: file.modelTopology,
    weightSpecs: file.weightSpecs,
    weightData: decodeBase64(file.weightData),
  });
}

function decodeBase64(text: string): ArrayBuffer {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}
