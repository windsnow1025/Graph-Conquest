/**
 * Custom TF.js IO handler for Node.js file system (without @tensorflow/tfjs-node).
 * Provides save/load functionality using fs/path.
 */
import * as tf from "@tensorflow/tfjs";
import * as fs from "fs";
import * as path from "path";

/**
 * Create an IO handler that saves/loads model to/from a directory.
 */
export function nodeFileSystem(dirPath: string): tf.io.IOHandler {
  return {
    async save(modelArtifacts: tf.io.ModelArtifacts): Promise<tf.io.SaveResult> {
      if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, {recursive: true});
      }

      // Save weights as a single binary file
      const weightsPath = path.join(dirPath, "weights.bin");
      if (modelArtifacts.weightData) {
        // weightData can be ArrayBuffer or ArrayBuffer[]
        let buffer: Buffer;
        if (Array.isArray(modelArtifacts.weightData)) {
          // Concatenate multiple ArrayBuffers
          const totalLength = modelArtifacts.weightData.reduce((sum, ab) => sum + ab.byteLength, 0);
          buffer = Buffer.alloc(totalLength);
          let offset = 0;
          for (const ab of modelArtifacts.weightData) {
            Buffer.from(ab).copy(buffer, offset);
            offset += ab.byteLength;
          }
        } else {
          buffer = Buffer.from(modelArtifacts.weightData);
        }
        fs.writeFileSync(weightsPath, buffer);
      }

      // Build model.json
      const modelJSON: Record<string, unknown> = {
        modelTopology: modelArtifacts.modelTopology,
        format: modelArtifacts.format,
        generatedBy: modelArtifacts.generatedBy,
        convertedBy: modelArtifacts.convertedBy,
      };

      if (modelArtifacts.weightSpecs) {
        modelJSON.weightsManifest = [{
          paths: ["weights.bin"],
          weights: modelArtifacts.weightSpecs,
        }];
      }

      if (modelArtifacts.trainingConfig) {
        modelJSON.trainingConfig = modelArtifacts.trainingConfig;
      }

      const modelJsonPath = path.join(dirPath, "model.json");
      fs.writeFileSync(modelJsonPath, JSON.stringify(modelJSON, null, 2));

      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: "JSON",
        },
      };
    },

    async load(): Promise<tf.io.ModelArtifacts> {
      const modelJsonPath = path.join(dirPath, "model.json");
      const modelJSON = JSON.parse(fs.readFileSync(modelJsonPath, "utf-8"));

      const artifacts: tf.io.ModelArtifacts = {
        modelTopology: modelJSON.modelTopology,
        format: modelJSON.format,
        generatedBy: modelJSON.generatedBy,
        convertedBy: modelJSON.convertedBy,
        trainingConfig: modelJSON.trainingConfig,
      };

      if (modelJSON.weightsManifest) {
        const manifest = modelJSON.weightsManifest;
        const weightSpecs: tf.io.WeightsManifestEntry[] = [];
        const weightBuffers: ArrayBuffer[] = [];

        for (const group of manifest) {
          weightSpecs.push(...group.weights);
          for (const weightsFile of group.paths) {
            const weightsPath = path.join(dirPath, weightsFile);
            const buf = fs.readFileSync(weightsPath);
            weightBuffers.push(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
          }
        }

        artifacts.weightSpecs = weightSpecs;

        // Concatenate all weight buffers
        const totalLength = weightBuffers.reduce((sum, ab) => sum + ab.byteLength, 0);
        const combined = new ArrayBuffer(totalLength);
        const view = new Uint8Array(combined);
        let offset = 0;
        for (const ab of weightBuffers) {
          view.set(new Uint8Array(ab), offset);
          offset += ab.byteLength;
        }
        artifacts.weightData = combined;
      }

      return artifacts;
    },
  };
}
