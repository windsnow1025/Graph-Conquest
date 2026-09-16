/**
 * Neural network model v9: context-free trunk with per-head context shortcut.
 *
 * Architecture:
 *   state_core[924] (encoding without the context block) → Dense(1024, ReLU) → Dense(256, ReLU) = trunk
 *   Each head gets: concat(trunk[256], decision_type[7], own_context[N])
 *   → Dense(64, ReLU) → Dense(units, activation)
 *
 * The trunk, and therefore the value head, never sees the decision context:
 * V(s) is a pure state value, so TD differences between consecutive states
 * are not polluted by context switches. Policy heads receive their own
 * context (plus the decision type) via the shortcut inputs.
 *
 * Categorical heads (masked softmax at inference):
 *   action_type[5], move_target[16] (destination node), battle_target[17] (node or stop)
 */
import * as tf from "@tensorflow/tfjs";
import {NUM_NODES} from "./StateEncoder";
import {NUM_ACTION_TYPES} from "./ActionSpace";

export const BATTLE_TARGET_DIM = NUM_NODES + 1; // 16 nodes + stop
export const BATTLE_TARGET_STOP = NUM_NODES;    // index of the stop option

export interface NNPrediction {
  value: number;
  actionTypeLogits: Float32Array;
  splitFraction: number;
  disbandFraction: number;
  recruitFraction: number;
  moveTargetLogits: Float32Array;   // [16] destination node logits
  battleTargetLogits: Float32Array; // [17] node logits + stop logit
  battleSelect: number;             // score of one option (army or done), argmax across options
  killFraction: number;
  battleRetreat: number;
}

const HEAD_HIDDEN = 64;

// Context layout within state[924..1148]:
// offset 924: decision_type[7]
// offset 931: recruit[20]
// offset 951: army[28]
// offset 979: moveTarget[39]
// offset 1018: battleTarget[16]
// offset 1034: battleSelect[51]
// offset 1085: battleAllocate[46]
// offset 1131: battleRetreat[17]

export const CTX_BASE = 924;
export const CTX_DT_OFF = 0;       export const CTX_DT_LEN = 7;
export const CTX_REC_OFF = 7;      export const CTX_REC_LEN = 20;
export const CTX_ARMY_OFF = 27;    export const CTX_ARMY_LEN = 28;
export const CTX_MOV_OFF = 55;     export const CTX_MOV_LEN = 39;
export const CTX_BTGT_OFF = 94;    export const CTX_BTGT_LEN = 16;
export const CTX_BSEL_OFF = 110;   export const CTX_BSEL_LEN = 51;
export const CTX_BALLOC_OFF = 161; export const CTX_BALLOC_LEN = 46;
export const CTX_BRET_OFF = 207;   export const CTX_BRET_LEN = 17;

export class NNModel {
  private model: tf.LayersModel | null = null;

  buildNew(): void {
    const stateInput = tf.input({shape: [CTX_BASE], name: "state_input"});

    // Context segment inputs
    const ctxDtInput = tf.input({shape: [CTX_DT_LEN], name: "ctx_dt"});
    const ctxRecInput = tf.input({shape: [CTX_REC_LEN], name: "ctx_rec"});
    const ctxArmyInput = tf.input({shape: [CTX_ARMY_LEN], name: "ctx_army"});
    const ctxMovInput = tf.input({shape: [CTX_MOV_LEN], name: "ctx_mov"});
    const ctxBtgtInput = tf.input({shape: [CTX_BTGT_LEN], name: "ctx_btgt"});
    const ctxBselInput = tf.input({shape: [CTX_BSEL_LEN], name: "ctx_bsel"});
    const ctxBallocInput = tf.input({shape: [CTX_BALLOC_LEN], name: "ctx_balloc"});
    const ctxBretInput = tf.input({shape: [CTX_BRET_LEN], name: "ctx_bret"});

    // Shared trunk
    const dense1 = tf.layers.dense({
      units: 1024, activation: "relu", name: "dense1",
    }).apply(stateInput) as tf.SymbolicTensor;

    const dense2 = tf.layers.dense({
      units: 256, activation: "relu", name: "dense2",
    }).apply(dense1) as tf.SymbolicTensor;

    // Per-head concat and layers
    function makeHead(
      name: string, units: number, activation: "sigmoid" | "linear",
      ctxInputs: tf.SymbolicTensor[],
    ): tf.SymbolicTensor {
      const headInput = ctxInputs.length > 0
        ? tf.layers.concatenate({name: `${name}_concat`}).apply([dense2, ...ctxInputs]) as tf.SymbolicTensor
        : dense2;
      const hidden = tf.layers.dense({
        units: HEAD_HIDDEN, activation: "relu", name: `${name}_hidden`,
      }).apply(headInput) as tf.SymbolicTensor;
      return tf.layers.dense({
        units, activation, name: `${name}_head`,
      }).apply(hidden) as tf.SymbolicTensor;
    }

    const valueOut           = makeHead("value", 1, "sigmoid", []);
    const actionTypeOut      = makeHead("action_type", NUM_ACTION_TYPES, "linear", [ctxDtInput, ctxArmyInput]);
    const splitFractionOut   = makeHead("split_fraction", 1, "sigmoid", [ctxDtInput, ctxArmyInput]);
    const disbandFractionOut = makeHead("disband_fraction", 1, "sigmoid", [ctxDtInput, ctxArmyInput]);
    const recruitFractionOut = makeHead("recruit_fraction", 1, "sigmoid", [ctxDtInput, ctxRecInput]);
    const moveTargetOut      = makeHead("move_target", NUM_NODES, "linear", [ctxDtInput, ctxMovInput]);
    const battleTargetOut    = makeHead("battle_target", BATTLE_TARGET_DIM, "linear", [ctxDtInput, ctxBtgtInput]);
    const battleSelectOut    = makeHead("battle_select", 1, "sigmoid", [ctxDtInput, ctxBselInput]);
    const killFractionOut    = makeHead("kill_fraction", 1, "sigmoid", [ctxDtInput, ctxBallocInput]);
    const battleRetreatOut   = makeHead("battle_retreat", 1, "sigmoid", [ctxDtInput, ctxBretInput]);

    this.model = tf.model({
      inputs: [stateInput, ctxDtInput, ctxRecInput, ctxArmyInput, ctxMovInput,
               ctxBtgtInput, ctxBselInput, ctxBallocInput, ctxBretInput],
      outputs: [
        valueOut, actionTypeOut, splitFractionOut, disbandFractionOut,
        recruitFractionOut, moveTargetOut, battleTargetOut, battleSelectOut,
        killFractionOut, battleRetreatOut,
      ],
      name: "graph_conquest_nn_v9",
    });
  }

  async load(pathOrHandler: string | tf.io.IOHandler): Promise<void> {
    this.model = await tf.loadLayersModel(pathOrHandler);
  }

  async save(pathOrHandler: string | tf.io.IOHandler): Promise<void> {
    if (!this.model) throw new Error("No model to save");
    await this.model.save(pathOrHandler);
  }

  predict(stateEncoding: Float32Array): NNPrediction {
    if (!this.model) throw new Error("Model not loaded");

    return tf.tidy(() => {
      const s = stateEncoding;
      const stateTensor = tf.tensor2d(s.subarray(0, CTX_BASE), [1, CTX_BASE]);
      const ctxDt = tf.tensor2d(s.subarray(CTX_BASE + CTX_DT_OFF, CTX_BASE + CTX_DT_OFF + CTX_DT_LEN), [1, CTX_DT_LEN]);
      const ctxRec = tf.tensor2d(s.subarray(CTX_BASE + CTX_REC_OFF, CTX_BASE + CTX_REC_OFF + CTX_REC_LEN), [1, CTX_REC_LEN]);
      const ctxArmy = tf.tensor2d(s.subarray(CTX_BASE + CTX_ARMY_OFF, CTX_BASE + CTX_ARMY_OFF + CTX_ARMY_LEN), [1, CTX_ARMY_LEN]);
      const ctxMov = tf.tensor2d(s.subarray(CTX_BASE + CTX_MOV_OFF, CTX_BASE + CTX_MOV_OFF + CTX_MOV_LEN), [1, CTX_MOV_LEN]);
      const ctxBtgt = tf.tensor2d(s.subarray(CTX_BASE + CTX_BTGT_OFF, CTX_BASE + CTX_BTGT_OFF + CTX_BTGT_LEN), [1, CTX_BTGT_LEN]);
      const ctxBsel = tf.tensor2d(s.subarray(CTX_BASE + CTX_BSEL_OFF, CTX_BASE + CTX_BSEL_OFF + CTX_BSEL_LEN), [1, CTX_BSEL_LEN]);
      const ctxBalloc = tf.tensor2d(s.subarray(CTX_BASE + CTX_BALLOC_OFF, CTX_BASE + CTX_BALLOC_OFF + CTX_BALLOC_LEN), [1, CTX_BALLOC_LEN]);
      const ctxBret = tf.tensor2d(s.subarray(CTX_BASE + CTX_BRET_OFF, CTX_BASE + CTX_BRET_OFF + CTX_BRET_LEN), [1, CTX_BRET_LEN]);

      const outputs = this.model!.predict([
        stateTensor, ctxDt, ctxRec, ctxArmy, ctxMov,
        ctxBtgt, ctxBsel, ctxBalloc, ctxBret,
      ]) as tf.Tensor[];

      return {
        value:              (outputs[0].dataSync() as Float32Array)[0],
        actionTypeLogits:   new Float32Array(outputs[1].dataSync()),
        splitFraction:      (outputs[2].dataSync() as Float32Array)[0],
        disbandFraction:    (outputs[3].dataSync() as Float32Array)[0],
        recruitFraction:    (outputs[4].dataSync() as Float32Array)[0],
        moveTargetLogits:   new Float32Array(outputs[5].dataSync()),
        battleTargetLogits: new Float32Array(outputs[6].dataSync()),
        battleSelect:       (outputs[7].dataSync() as Float32Array)[0],
        killFraction:       (outputs[8].dataSync() as Float32Array)[0],
        battleRetreat:      (outputs[9].dataSync() as Float32Array)[0],
      };
    });
  }

  getModel(): tf.LayersModel {
    if (!this.model) throw new Error("Model not loaded");
    return this.model;
  }

  isLoaded(): boolean {
    return this.model !== null;
  }

  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
  }
}

// ─── Utility functions ───

export function applyMaskAndSoftmax(logits: Float32Array, mask: Float32Array): Float32Array {
  const masked = new Float32Array(logits.length);
  let maxVal = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (mask[i] > 0) { masked[i] = logits[i]; if (logits[i] > maxVal) maxVal = logits[i]; }
    else { masked[i] = -Infinity; }
  }
  let sumExp = 0;
  const probs = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    if (masked[i] > -Infinity) { probs[i] = Math.exp(masked[i] - maxVal); sumExp += probs[i]; }
  }
  if (sumExp > 0) { for (let i = 0; i < logits.length; i++) probs[i] /= sumExp; }
  return probs;
}

export function argmax(arr: Float32Array): number {
  return arr.reduce((best, v, i) => (v > arr[best] ? i : best), 0);
}

export function sampleFromProbs(probs: Float32Array): number {
  const r = Math.random();
  let cumulative = 0;
  for (let i = 0; i < probs.length; i++) { cumulative += probs[i]; if (r < cumulative) return i; }
  // Float32 rounding can leave the cumulative sum slightly below 1; the fallback
  // must not pick a masked (zero-probability) option
  const lastLegal = probs.findLastIndex(p => p > 0);
  return lastLegal >= 0 ? lastLegal : probs.length - 1;
}
