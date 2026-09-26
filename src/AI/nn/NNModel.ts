/**
 * Neural network model v10: context-free trunk with per-head context shortcut.
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
 * Per-group fraction heads (sigmoid, one value per unit group):
 *   move_fraction[4], disband_fraction[6], commit_fraction[3]
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

export const BATTLE_TARGET_DIM = NUM_NODES + 1; // 16 nodes + stop
export const BATTLE_TARGET_STOP = NUM_NODES;    // index of the stop option

export interface NNPrediction {
  value: number;
  actionTypeLogits: Float32Array;   // [3]
  moveFraction: Float32Array;       // [4] fraction per move group
  disbandFraction: Float32Array;    // [6] fraction per disband group
  recruitFraction: number;
  moveTargetLogits: Float32Array;   // [16] destination node logits
  battleTargetLogits: Float32Array; // [17] node logits + stop logit
  battleSelect: number;             // score of one option (army or done), argmax across options
  commitFraction: Float32Array;     // [3] fraction per commit group of the option's army
  killFraction: number;
  battleRetreat: number;
}

const HEAD_HIDDEN = 64;

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

    const valueOut           = makeHead("value", 1, "sigmoid", []);
    const actionTypeOut      = makeHead("action_type", NUM_ACTION_TYPES, "linear", [ctxDtInput, ctxArmyInput]);
    const moveFractionOut    = makeHead("move_fraction", NUM_MOVE_GROUPS, "sigmoid", [ctxDtInput, ctxMcntInput]);
    const disbandFractionOut = makeHead("disband_fraction", NUM_DISBAND_GROUPS, "sigmoid", [ctxDtInput, ctxArmyInput]);
    const recruitFractionOut = makeHead("recruit_fraction", 1, "sigmoid", [ctxDtInput, ctxRecInput]);
    const moveTargetOut      = makeHead("move_target", NUM_NODES, "linear", [ctxDtInput, ctxMovInput]);
    const battleTargetOut    = makeHead("battle_target", BATTLE_TARGET_DIM, "linear", [ctxDtInput, ctxBtgtInput]);
    const battleSelectOut    = makeHead("battle_select", 1, "sigmoid", [ctxDtInput, ctxBselInput]);
    const commitFractionOut  = makeHead("commit_fraction", NUM_COMMIT_GROUPS, "sigmoid", [ctxDtInput, ctxBselInput]);
    const killFractionOut    = makeHead("kill_fraction", 1, "sigmoid", [ctxDtInput, ctxBallocInput]);
    const battleRetreatOut   = makeHead("battle_retreat", 1, "sigmoid", [ctxDtInput, ctxBretInput]);

    this.model = tf.model({
      inputs: [stateInput, ctxDtInput, ctxRecInput, ctxArmyInput, ctxMovInput, ctxMcntInput,
               ctxBtgtInput, ctxBselInput, ctxBallocInput, ctxBretInput],
      outputs: [
        valueOut, actionTypeOut, moveFractionOut, disbandFractionOut,
        recruitFractionOut, moveTargetOut, battleTargetOut, battleSelectOut,
        commitFractionOut, killFractionOut, battleRetreatOut,
      ],
      name: "graph_conquest_nn_v10",
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
      const block = (off: number, len: number) => tf.tensor2d(s.subarray(CTX_BASE + off, CTX_BASE + off + len), [1, len]);
      const stateTensor = tf.tensor2d(s.subarray(0, CTX_BASE), [1, CTX_BASE]);

      const outputs = this.model!.predict([
        stateTensor,
        block(CTX_DT_OFF, CTX_DT_LEN), block(CTX_REC_OFF, CTX_REC_LEN), block(CTX_ARMY_OFF, CTX_ARMY_LEN),
        block(CTX_MOV_OFF, CTX_MOV_LEN), block(CTX_MCNT_OFF, CTX_MCNT_LEN), block(CTX_BTGT_OFF, CTX_BTGT_LEN),
        block(CTX_BSEL_OFF, CTX_BSEL_LEN), block(CTX_BALLOC_OFF, CTX_BALLOC_LEN), block(CTX_BRET_OFF, CTX_BRET_LEN),
      ]) as tf.Tensor[];

      return {
        value:              (outputs[0].dataSync() as Float32Array)[0],
        actionTypeLogits:   new Float32Array(outputs[1].dataSync()),
        moveFraction:       new Float32Array(outputs[2].dataSync()),
        disbandFraction:    new Float32Array(outputs[3].dataSync()),
        recruitFraction:    (outputs[4].dataSync() as Float32Array)[0],
        moveTargetLogits:   new Float32Array(outputs[5].dataSync()),
        battleTargetLogits: new Float32Array(outputs[6].dataSync()),
        battleSelect:       (outputs[7].dataSync() as Float32Array)[0],
        commitFraction:     new Float32Array(outputs[8].dataSync()),
        killFraction:       (outputs[9].dataSync() as Float32Array)[0],
        battleRetreat:      (outputs[10].dataSync() as Float32Array)[0],
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
