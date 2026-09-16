/**
 * Training sample types shared between GreedyAI (recording) and train.ts (export).
 * Must match training/python/app/config.py SAMPLE_FLOATS = 1203.
 */
import {STATE_SIZE, NUM_NODES} from "../../src/AI/nn/StateEncoder";
import {NUM_ACTION_TYPES} from "../../src/AI/nn/ActionSpace";
import {BATTLE_TARGET_DIM} from "../../src/AI/nn/NNModel";

/** Terminal value of a draw. 1/3 = uniform prior over 3 players; win=1, loss=0. */
export const DRAW_VALUE = 1 / 3;

/**
 * Terminal value of a game for one player: win=1, loss=0, draw=DRAW_VALUE.
 * A player eliminated before a timeout draw scores 0, not DRAW_VALUE.
 */
export function terminalValue(winner: string, player: {name: string; defeated: boolean}): number {
  if (player.defeated) return 0;
  return winner === "draw" ? DRAW_VALUE : winner === player.name ? 1 : 0;
}

export const SAMPLE_FLOATS = STATE_SIZE + 2
  + (1 + NUM_ACTION_TYPES)      // actionTypeTarget + mask[5]
  + 2 * 3                       // split, disband, recruit (fraction + mask)
  + (1 + NUM_NODES)             // moveTargetIdx + legal mask[16]
  + (1 + BATTLE_TARGET_DIM)     // battleTargetIdx + option mask[17]
  + 2 * 3;                      // battleSelect, killFraction, battleRetreat (target + mask)
// = 1148 + 2 + 6 + 6 + 17 + 18 + 6 = 1203

export interface Sample {
  playerIdx: number;        // perspective the state was encoded from; not serialized
  state: Float32Array;
  value: number;            // -1 = no label
  policyWeight: number;     // weight for policy heads
  actionTypeTarget: number; // -1 = no label
  actionTypeMask: Float32Array;
  splitFraction: number; splitMask: number;
  disbandFraction: number; disbandMask: number;
  recruitFraction: number; recruitMask: number;
  moveTargetIdx: number;         // -1 = no label, else destination node index
  moveMask: Float32Array;        // [16] legal destinations
  battleTargetIdx: number;       // -1 = no label, 0-15 = node, 16 = stop
  battleTargetMask: Float32Array; // [17] attackable nodes + stop
  battleSelect: number; battleSelectMask: number;
  killFraction: number; killFracMask: number;
  battleRetreat: number; retreatMask: number;
}

export function emptySample(playerIdx: number): Sample {
  return {
    playerIdx,
    state: new Float32Array(STATE_SIZE),
    value: -1,
    policyWeight: 1,
    actionTypeTarget: -1,
    actionTypeMask: new Float32Array(NUM_ACTION_TYPES),
    splitFraction: 0, splitMask: 0,
    disbandFraction: 0, disbandMask: 0,
    recruitFraction: 0, recruitMask: 0,
    moveTargetIdx: -1,
    moveMask: new Float32Array(NUM_NODES),
    battleTargetIdx: -1,
    battleTargetMask: new Float32Array(BATTLE_TARGET_DIM),
    battleSelect: 0, battleSelectMask: 0,
    killFraction: 0, killFracMask: 0,
    battleRetreat: 0, retreatMask: 0,
  };
}

export function sampleToFloats(s: Sample): Float32Array {
  const buf = new Float32Array(SAMPLE_FLOATS);
  let off = 0;
  buf.set(s.state, off); off += STATE_SIZE;
  buf[off++] = s.value;
  buf[off++] = s.policyWeight;
  buf[off++] = s.actionTypeTarget;
  buf.set(s.actionTypeMask, off); off += NUM_ACTION_TYPES;
  buf[off++] = s.splitFraction; buf[off++] = s.splitMask;
  buf[off++] = s.disbandFraction; buf[off++] = s.disbandMask;
  buf[off++] = s.recruitFraction; buf[off++] = s.recruitMask;
  buf[off++] = s.moveTargetIdx;
  buf.set(s.moveMask, off); off += NUM_NODES;
  buf[off++] = s.battleTargetIdx;
  buf.set(s.battleTargetMask, off); off += BATTLE_TARGET_DIM;
  buf[off++] = s.battleSelect; buf[off++] = s.battleSelectMask;
  buf[off++] = s.killFraction; buf[off++] = s.killFracMask;
  buf[off++] = s.battleRetreat; buf[off++] = s.retreatMask;
  return buf;
}
