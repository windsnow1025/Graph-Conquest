/**
 * Neural network model v11: context-free trunk with per-head context shortcut
 * and categorical fraction heads.
 *
 * Architecture:
 *   state_core[1068] (encoding without the context block) → Dense(1024, ReLU) → Dense(256, ReLU) = trunk
 *   Each head gets: concat(trunk[256], decision_type[8], own_context[N])
 *   → Dense(64, ReLU) → Dense(units, activation)
 *
 * The trunk, and therefore the value head, never sees the decision context:
 * V(s) is a pure state value, so TD differences between consecutive states
 * are not polluted by context switches. Policy heads receive their own
 * context (plus the decision type) via the shortcut inputs.
 *
 * Categorical heads (masked softmax at inference):
 *   action_type[3], move_target[16] (destination node), battle_target[17] (node or stop)
 * Fraction heads (FRACTION_BINS logits per unit group, softmax per group):
 *   move_fraction[4 groups], disband_fraction[6], commit_fraction[3], recruit_fraction[1], kill_fraction[1]
 */
import * as tf from "@tensorflow/tfjs";
import {NUM_NODES} from "./GameIndex";
import {
  CTX_BASE,
  CTX_DT_OFF, CTX_DT_LEN,
  CTX_REC_OFF, CTX_REC_LEN,
  CTX_ARMY_OFF, CTX_ARMY_LEN,
  CTX_MOV_OFF, CTX_MOV_LEN,
  CTX_MCNT_OFF, CTX_MCNT_LEN,
  CTX_BTGT_OFF, CTX_BTGT_LEN,
  CTX_BSEL_OFF, CTX_BSEL_LEN,
  CTX_BALLOC_OFF, CTX_BALLOC_LEN,
  CTX_BRET_OFF, CTX_BRET_LEN,
} from "./StateEncoder";
import {NUM_ACTION_TYPES} from "./ActionSpace";
import {NUM_MOVE_GROUPS, NUM_DISBAND_GROUPS, NUM_COMMIT_GROUPS} from "./UnitGroups";
import {FRACTION_BINS, binToFraction} from "./FractionBins";

export const BATTLE_TARGET_DIM = NUM_NODES + 1; // 16 nodes + stop
export const BATTLE_TARGET_STOP = NUM_NODES;    // index of the stop option

export interface NNPrediction {
  value: number;
  actionTypeLogits: Float32Array;      // [3]
  moveFractionLogits: Float32Array;    // [4 × FRACTION_BINS] per move group
  disbandFractionLogits: Float32Array; // [6 × FRACTION_BINS] per disband group
  recruitFractionLogits: Float32Array; // [FRACTION_BINS]
  moveTargetLogits: Float32Array;      // [16] destination node logits
  battleTargetLogits: Float32Array;    // [17] node logits + stop logit
  battleSelect: number;                // score of one option (army or done), argmax across options
  commitFractionLogits: Float32Array;  // [3 × FRACTION_BINS] per commit group of the option's army
  killFractionLogits: Float32Array;    // [FRACTION_BINS]
  battleRetreat: number;
}

const HEAD_HIDDEN = 64;

// Output sizes in head order; a loaded model must match them
const HEAD_SIZES = [
  1, NUM_ACTION_TYPES, NUM_MOVE_GROUPS * FRACTION_BINS, NUM_DISBAND_GROUPS * FRACTION_BINS, FRACTION_BINS,
  NUM_NODES, BATTLE_TARGET_DIM, 1, NUM_COMMIT_GROUPS * FRACTION_BINS, FRACTION_BINS, 1,
];

export class NNModel {
  private model: tf.LayersModel | null = null;

  buildNew(): void {
    const stateInput = tf.input({shape: [CTX_BASE], name: "state_input"});

    // Context segment inputs
    const ctxDtInput = tf.input({shape: [CTX_DT_LEN], name: "ctx_dt"});
    const ctxRecInput = tf.input({shape: [CTX_REC_LEN], name: "ctx_rec"});
    const ctxArmyInput = tf.input({shape: [CTX_ARMY_LEN], name: "ctx_army"});
    const ctxMovInput = tf.input({shape: [CTX_MOV_LEN], name: "ctx_mov"});
    const ctxMcntInput = tf.input({shape: [CTX_MCNT_LEN], name: "ctx_mcnt"});
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

    const valueOut           = makeHead("value", HEAD_SIZES[0], "sigmoid", []);
    const actionTypeOut      = makeHead("action_type", HEAD_SIZES[1], "linear", [ctxDtInput, ctxArmyInput]);
    const moveFractionOut    = makeHead("move_fraction", HEAD_SIZES[2], "linear", [ctxDtInput, ctxMcntInput]);
    const disbandFractionOut = makeHead("disband_fraction", HEAD_SIZES[3], "linear", [ctxDtInput, ctxArmyInput]);
    const recruitFractionOut = makeHead("recruit_fraction", HEAD_SIZES[4], "linear", [ctxDtInput, ctxRecInput]);
    const moveTargetOut      = makeHead("move_target", HEAD_SIZES[5], "linear", [ctxDtInput, ctxMovInput]);
    const battleTargetOut    = makeHead("battle_target", HEAD_SIZES[6], "linear", [ctxDtInput, ctxBtgtInput]);
    const battleSelectOut    = makeHead("battle_select", HEAD_SIZES[7], "sigmoid", [ctxDtInput, ctxBselInput]);
    const commitFractionOut  = makeHead("commit_fraction", HEAD_SIZES[8], "linear", [ctxDtInput, ctxBselInput]);
    const killFractionOut    = makeHead("kill_fraction", HEAD_SIZES[9], "linear", [ctxDtInput, ctxBallocInput]);
    const battleRetreatOut   = makeHead("battle_retreat", HEAD_SIZES[10], "sigmoid", [ctxDtInput, ctxBretInput]);

    this.model = tf.model({
      inputs: [stateInput, ctxDtInput, ctxRecInput, ctxArmyInput, ctxMovInput, ctxMcntInput,
               ctxBtgtInput, ctxBselInput, ctxBallocInput, ctxBretInput],
      outputs: [
        valueOut, actionTypeOut, moveFractionOut, disbandFractionOut,
        recruitFractionOut, moveTargetOut, battleTargetOut, battleSelectOut,
        commitFractionOut, killFractionOut, battleRetreatOut,
      ],
      name: "graph_conquest_nn_v11",
    });
  }

  async load(pathOrHandler: string | tf.io.IOHandler): Promise<void> {
    const model = await tf.loadLayersModel(pathOrHandler);
    const sizes = model.outputs.map(output => output.shape[1]);
    if (sizes.length !== HEAD_SIZES.length || sizes.some((size, i) => size !== HEAD_SIZES[i])) {
      model.dispose();
      throw new Error(`Model head sizes [${sizes.join(", ")}] do not match the current architecture [${HEAD_SIZES.join(", ")}]`);
    }
    this.model = model;
  }

  async save(pathOrHandler: string | tf.io.IOHandler): Promise<void> {
    if (!this.model) throw new Error("No model to save");
    await this.model.save(pathOrHandler);
  }

  predict(stateEncoding: Float32Array): NNPrediction {
    if (!this.model) throw new Error("Model not loaded");

    return tf.tidy(() => {
      const s = stateEncoding;
      const block = (off: number, len: number) => tf.tensor2d(s.subarray(CTX_BASE + off, CTX_BASE + off + len), [1, len]);
      const stateTensor = tf.tensor2d(s.subarray(0, CTX_BASE), [1, CTX_BASE]);

      const outputs = this.model!.predict([
        stateTensor,
        block(CTX_DT_OFF, CTX_DT_LEN), block(CTX_REC_OFF, CTX_REC_LEN), block(CTX_ARMY_OFF, CTX_ARMY_LEN),
        block(CTX_MOV_OFF, CTX_MOV_LEN), block(CTX_MCNT_OFF, CTX_MCNT_LEN), block(CTX_BTGT_OFF, CTX_BTGT_LEN),
        block(CTX_BSEL_OFF, CTX_BSEL_LEN), block(CTX_BALLOC_OFF, CTX_BALLOC_LEN), block(CTX_BRET_OFF, CTX_BRET_LEN),
      ]) as tf.Tensor[];

      return {
        value:                 (outputs[0].dataSync() as Float32Array)[0],
        actionTypeLogits:      new Float32Array(outputs[1].dataSync()),
        moveFractionLogits:    new Float32Array(outputs[2].dataSync()),
        disbandFractionLogits: new Float32Array(outputs[3].dataSync()),
        recruitFractionLogits: new Float32Array(outputs[4].dataSync()),
        moveTargetLogits:      new Float32Array(outputs[5].dataSync()),
        battleTargetLogits:    new Float32Array(outputs[6].dataSync()),
        battleSelect:          (outputs[7].dataSync() as Float32Array)[0],
        commitFractionLogits:  new Float32Array(outputs[8].dataSync()),
        killFractionLogits:    new Float32Array(outputs[9].dataSync()),
        battleRetreat:         (outputs[10].dataSync() as Float32Array)[0],
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

export function softmax(logits: Float32Array): Float32Array {
  let maxVal = -Infinity;
  for (const v of logits) if (v > maxVal) maxVal = v;
  const probs = Float32Array.from(logits, v => Math.exp(v - maxVal));
  let sumExp = 0;
  for (const p of probs) sumExp += p;
  for (let i = 0; i < probs.length; i++) probs[i] /= sumExp;
  return probs;
}

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

/** The logits of one group of a fraction head. */
export function groupLogits(logits: Float32Array, group: number): Float32Array {
  return logits.subarray(group * FRACTION_BINS, (group + 1) * FRACTION_BINS);
}

/** The most likely level of every group of a fraction head, as fractions. */
export function argmaxFractions(logits: Float32Array, groups: number): Float32Array {
  return Float32Array.from({length: groups}, (_, g) => binToFraction(argmax(groupLogits(logits, g))));
}
