/**
 * RL game runners with recording: the pure simulation side of the pipeline.
 *
 * Uses TurnExecutor's onDecision callback to record what the NN decides and
 * takes a context-free critic snapshot at the start of every turn. Runners
 * return raw trajectories (records, snapshots, outcomes) with advantages
 * unassigned; credit assignment (assignAdvantages) and sample conversion run
 * at consumption time, so persisted trajectories can be re-weighted with any
 * λ without re-simulation (see TrajectoryStore).
 * The snapshots double as value-only training samples, so the critic keeps
 * learning on exactly the context-free encodings the TD differences read.
 * ε-explored decisions keep only their value label (no policy target).
 */
import type {NNModel} from "../../src/AI/nn/NNModel";
import {BATTLE_TARGET_DIM, BATTLE_TARGET_STOP} from "../../src/AI/nn/NNModel";
import {executeNNTurn} from "../../src/AI/TurnExecutor";
import type {DecisionAction, DecisionInfo, TurnOptions} from "../../src/AI/TurnExecutor";
import {ACTION_DISBAND, NUM_ACTION_TYPES} from "../../src/AI/nn/ActionSpace";
import {NUM_MOVE_GROUPS, NUM_DISBAND_GROUPS, NUM_COMMIT_GROUPS} from "../../src/AI/nn/UnitGroups";
import {NUM_NODES} from "../../src/AI/nn/GameIndex";
import {
  encodeState, CTX_BASE, CTX_ARMY_OFF, CTX_MOV_OFF, CTX_MCNT_OFF, CTX_BTGT_OFF, CTX_BSEL_OFF,
  ARMY_ACTION_MASK_OFF, ARMY_DISBAND_MASK_OFF, MOV_LEGAL_MASK_OFF, MCNT_GROUP_MASK_OFF, BSEL_COMMIT_MASK_OFF,
} from "../../src/AI/nn/StateEncoder";
import type {Sample} from "./SampleTypes";
import {emptySample, terminalValue} from "./SampleTypes";
import {randomTurn} from "./Opponents";
import type GameSystem from "../../src/lib/GameSystem";

export interface RawRecord {
  playerIdx: number;
  state: Float32Array;
  action: DecisionAction;
  explored: boolean;
  advantage: number;
}

export interface TurnSnapshot {
  playerIdx: number;
  state: Float32Array;  // context-free encoding at the player's turn start
  v: number;            // critic value of that state
  recordFrom: number;   // records.length when the turn began
}

export interface GameResult {
  records: RawRecord[];
  outcomes: number[];
  snapshots: TurnSnapshot[];
}

// Mask locations within the encoded state (features are centered, so test > 0)
const ACTION_MASK_OFFSET = CTX_BASE + CTX_ARMY_OFF + ARMY_ACTION_MASK_OFF;
const DISBAND_MASK_OFFSET = CTX_BASE + CTX_ARMY_OFF + ARMY_DISBAND_MASK_OFF;
const MOVE_MASK_OFFSET = CTX_BASE + CTX_MOV_OFF + MOV_LEGAL_MASK_OFF;
const MOVE_GROUP_MASK_OFFSET = CTX_BASE + CTX_MCNT_OFF + MCNT_GROUP_MASK_OFF;
const BTGT_MASK_OFFSET = CTX_BASE + CTX_BTGT_OFF;
const COMMIT_MASK_OFFSET = CTX_BASE + CTX_BSEL_OFF + BSEL_COMMIT_MASK_OFF;

function readMask(state: Float32Array, offset: number, length: number): Float32Array {
  return Float32Array.from({length}, (_, i) => (state[offset + i] > 0 ? 1 : 0));
}

function makeRecordingOpts(records: RawRecord[], temperature: number, epsilon: number): TurnOptions {
  return {
    temperature,
    epsilon,
    onDecision: (info: DecisionInfo) => {
      records.push({
        playerIdx: info.playerIdx,
        state: new Float32Array(info.state),
        action: info.action,
        explored: info.explored,
        advantage: 0,
      });
    },
  };
}

function takeSnapshot(game: GameSystem, model: NNModel, records: RawRecord[], snapshots: TurnSnapshot[]): void {
  const playerIdx = game.currentPlayerIndex;
  const state = encodeState(game, playerIdx);
  // fround: keep advantages bit-identical whether trajectories are consumed
  // in memory or after an f32 round-trip through the trajectory store
  snapshots.push({playerIdx, state, v: Math.fround(model.predict(state).value), recordFrom: records.length});
}

/**
 * TD(λ) advantages per turn. δ_k = V(next own-turn start) − V(own-turn start)
 * (the last interval bootstraps to the outcome); the assigned advantage is the
 * λ-return A_k = δ_k + λ·A_{k+1}, computed by backward recursion.
 *
 * λ = 0 is the pure turn-level TD used against Random (all-win data: local
 * V-improvement is always improvement toward the win). In mixed-outcome data
 * (draws/losses) pure TD reinforces locally V-raising turns inside globally
 * bad trajectories (the hoard-to-draw failure); λ > 0 propagates the terminal
 * truth backward so those turns end non-positive and count against their
 * actions in the trainer's clipped surrogate.
 *
 * Consumer-side: called at materialization time (TrajectoryStore), not by the
 * game runners, so stored trajectories serve any λ.
 */
export function assignAdvantages(records: RawRecord[], snapshots: TurnSnapshot[], outcomes: number[], tdLambda: number): void {
  for (let p = 0; p < outcomes.length; p++) {
    const own = snapshots.filter(s => s.playerIdx === p);
    if (own.length === 0) continue;

    const deltas = own.map((snap, k) => {
      const nextV = k + 1 < own.length ? own[k + 1].v : outcomes[p];
      return nextV - snap.v;
    });
    const advantages = new Array<number>(own.length);
    let acc = 0;
    for (let k = own.length - 1; k >= 0; k--) {
      acc = deltas[k] + tdLambda * acc;
      advantages[k] = acc;
    }

    for (let k = 0; k < own.length; k++) {
      const to = k + 1 < own.length ? own[k + 1].recordFrom : records.length;
      for (let i = own[k].recordFrom; i < to; i++) {
        if (records[i].playerIdx === p) records[i].advantage = advantages[k];
      }
    }
    // Records of p before p's first snapshot (defending before their first turn) keep advantage 0
  }
}

function finishGame(game: GameSystem, records: RawRecord[], snapshots: TurnSnapshot[]): GameResult {
  return {records, outcomes: computeOutcomes(game), snapshots};
}

/**
 * Play one NN-vs-Random game with exploration.
 * NN plays as nnIdx, other players use random.
 * Records only NN's decisions via TurnOptions.onDecision.
 */
export function nnVsRandomGame(
  game: GameSystem, model: NNModel, nnIdx: number, maxTurns: number,
  temperature: number, epsilon: number,
): GameResult {
  const records: RawRecord[] = [];
  const snapshots: TurnSnapshot[] = [];
  const opts = makeRecordingOpts(records, temperature, epsilon);

  for (let turn = 0; turn < maxTurns * 3 && !game.gameOver; turn++) {
    takeSnapshot(game, model, records, snapshots);
    if (game.currentPlayerIndex === nnIdx) {
      executeNNTurn(game, model, opts);
    } else {
      randomTurn(game);
    }
  }

  return finishGame(game, records, snapshots);
}

/**
 * Play one 3-NN self-play game with exploration.
 * All 3 players use the same NN model.
 * Records all players' decisions via TurnOptions.onDecision.
 */
export function nnSelfPlayGame(
  game: GameSystem, model: NNModel, maxTurns: number,
  temperature: number, epsilon: number,
): GameResult {
  const records: RawRecord[] = [];
  const snapshots: TurnSnapshot[] = [];
  const opts = makeRecordingOpts(records, temperature, epsilon);

  for (let turn = 0; turn < maxTurns * 3 && !game.gameOver; turn++) {
    takeSnapshot(game, model, records, snapshots);
    executeNNTurn(game, model, opts);
  }

  return finishGame(game, records, snapshots);
}

export type OpponentFn = (game: GameSystem) => void;

/**
 * Play one game: current model vs opponent function.
 * Current model plays as nnIdx, opponents use opponentFn.
 * Records only current model's decisions.
 */
export function nnVsOpponentGame(
  game: GameSystem, model: NNModel, nnIdx: number, opponentFn: OpponentFn,
  maxTurns: number, temperature: number, epsilon: number,
): GameResult {
  const records: RawRecord[] = [];
  const snapshots: TurnSnapshot[] = [];
  const opts = makeRecordingOpts(records, temperature, epsilon);

  for (let turn = 0; turn < maxTurns * 3 && !game.gameOver; turn++) {
    takeSnapshot(game, model, records, snapshots);
    if (game.currentPlayerIndex === nnIdx) {
      executeNNTurn(game, model, opts);
    } else {
      opponentFn(game);
    }
  }

  return finishGame(game, records, snapshots);
}

function computeOutcomes(game: GameSystem): number[] {
  const winner = game.winner?.name ?? "draw";
  // fround: see takeSnapshot
  return game.players.map(p => Math.fround(terminalValue(winner, p)));
}

/**
 * Convert raw records to training samples.
 *
 * value target = game outcome (win=1, loss=0, draw=1/3, eliminated=0)
 * policyWeight = the signed turn-level TD(λ) advantage. Only the trainer's
 * clipped surrogate (objective ppo) may read a negative weight: its push-away
 * stops at the clip. A CE/BCE loss with a negative weight is unbounded below
 * and collapsed the policy every time it was tried, and the trainer asserts
 * that its imitation objective never sees one.
 */
export function recordsToSamples(records: RawRecord[], outcomes: number[]): Sample[] {
  return records.map(rec => {
    const s = emptySample(rec.playerIdx);
    s.state = rec.state;
    s.value = outcomes[rec.playerIdx];
    s.policyWeight = rec.advantage;

    // ε-random actions are noise, not policy: train only the value head on them
    if (rec.explored) return s;

    // Legality masks are read back from the recorded state
    const action = rec.action;
    switch (action.type) {
      case "army":
        s.actionTypeTarget = action.actionType;
        s.actionTypeMask = readMask(rec.state, ACTION_MASK_OFFSET, NUM_ACTION_TYPES);
        if (action.actionType === ACTION_DISBAND) {
          s.disbandFraction = new Float32Array(action.disbandFraction);
          s.disbandMask = readMask(rec.state, DISBAND_MASK_OFFSET, NUM_DISBAND_GROUPS);
        }
        break;
      case "moveTarget":
        s.moveTargetIdx = action.moveTarget;
        s.moveMask = readMask(rec.state, MOVE_MASK_OFFSET, NUM_NODES);
        break;
      case "moveCount":
        s.moveFraction = new Float32Array(action.moveFraction);
        s.moveFractionMask = readMask(rec.state, MOVE_GROUP_MASK_OFFSET, NUM_MOVE_GROUPS);
        break;
      case "recruit":
        s.recruitFraction = action.recruitFraction;
        s.recruitMask = 1;
        break;
      case "battleTarget": {
        s.battleTargetIdx = action.battleTarget;
        const mask = new Float32Array(BATTLE_TARGET_DIM);
        mask.set(readMask(rec.state, BTGT_MASK_OFFSET, NUM_NODES));
        mask[BATTLE_TARGET_STOP] = 1;
        s.battleTargetMask = mask;
        break;
      }
      case "battleSelect":
        s.battleSelect = action.chosen;
        s.battleSelectMask = 1;
        // The chosen army's commitment was acted on; the "done" option has an all-zero mask
        if (action.chosen === 1) {
          s.commitFraction = new Float32Array(action.commitFraction);
          s.commitMask = readMask(rec.state, COMMIT_MASK_OFFSET, NUM_COMMIT_GROUPS);
        }
        break;
      case "battleAllocate":
        s.killFraction = action.killFraction;
        s.killFracMask = 1;
        break;
      case "battleRetreat":
        s.battleRetreat = action.battleRetreat;
        s.retreatMask = 1;
        break;
    }

    return s;
  });
}

/** Context-free turn-start states as value-only samples (the critic diet for TD). */
export function snapshotsToValueSamples(snapshots: TurnSnapshot[], outcomes: number[]): Sample[] {
  return snapshots.map(snap => {
    const s = emptySample(snap.playerIdx);
    s.state = snap.state;
    s.value = outcomes[snap.playerIdx];
    return s;
  });
}
