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
import {
  CTX_BASE, CTX_ARMY_OFF, CTX_MOV_OFF, CTX_BTGT_OFF,
  BATTLE_TARGET_DIM, BATTLE_TARGET_STOP,
} from "../../src/AI/nn/NNModel";
import {executeNNTurn} from "../../src/AI/TurnExecutor";
import type {DecisionInfo, TurnOptions} from "../../src/AI/TurnExecutor";
import {ACTION_SPLIT, ACTION_DISBAND, NUM_ACTION_TYPES} from "../../src/AI/nn/ActionSpace";
import {NUM_NODES, encodeState} from "../../src/AI/nn/StateEncoder";
import type {Sample} from "./SampleTypes";
import {emptySample, terminalValue} from "./SampleTypes";
import {randomTurn} from "./Opponents";
import type GameSystem from "../../src/lib/GameSystem";

export interface RawRecord {
  playerIdx: number;
  state: Float32Array;
  decisionType: string;
  action: Record<string, number>;
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
const ACTION_MASK_OFFSET = CTX_BASE + CTX_ARMY_OFF + 23; // army context: features(23) + mask(5)
const MOVE_MASK_OFFSET = CTX_BASE + CTX_MOV_OFF + 23;    // moveTarget context: army info(23) + legal mask(16)
const BTGT_MASK_OFFSET = CTX_BASE + CTX_BTGT_OFF;        // battleTarget context: attackable mask(16)

function makeRecordingOpts(records: RawRecord[], temperature: number, epsilon: number): TurnOptions {
  return {
    temperature,
    epsilon,
    onDecision: (info: DecisionInfo) => {
      records.push({
        playerIdx: info.playerIdx,
        state: new Float32Array(info.state),
        decisionType: info.decisionType,
        action: {...info.action},
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
 * truth backward so those turns end non-positive and are discarded by the
 * non-negative weight clamp.
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
 * policyWeight = turn-level TD advantage, clamped to be non-negative.
 *
 * DO NOT let negative policy weights through. They have been introduced and
 * removed several times in this project's history and collapsed the policy
 * every time: offline epochs flip CE/BCE into a push-away objective whose
 * gradient never saturates, so near-cancelling ± advantage noise nets out as
 * repulsion of everything the policy does and play degenerates to passivity
 * (all-draw evals). Reinforce good turns; ignore bad ones.
 */
export function recordsToSamples(records: RawRecord[], outcomes: number[]): Sample[] {
  return records.map(rec => {
    const s = emptySample(rec.playerIdx);
    s.state = rec.state;
    s.value = outcomes[rec.playerIdx];
    s.policyWeight = Math.max(0, rec.advantage);

    // ε-random actions are noise, not policy: train only the value head on them
    if (rec.explored) return s;

    const dt = rec.decisionType;
    const action = rec.action;

    if (dt === "army") {
      const actionType = action.actionType ?? 0;
      // Extract mask from state (features are centered, so legal = value > 0)
      s.actionTypeTarget = actionType;
      s.actionTypeMask = Float32Array.from({length: NUM_ACTION_TYPES},
        (_, i) => rec.state[ACTION_MASK_OFFSET + i] > 0 ? 1 : 0);
      if (actionType === ACTION_SPLIT) { s.splitFraction = action.fraction ?? 0; s.splitMask = 1; }
      if (actionType === ACTION_DISBAND) { s.disbandFraction = action.fraction ?? 0; s.disbandMask = 1; }
    } else if (dt === "moveTarget") {
      s.moveTargetIdx = action.moveTarget ?? -1;
      s.moveMask = Float32Array.from({length: NUM_NODES},
        (_, i) => rec.state[MOVE_MASK_OFFSET + i] > 0 ? 1 : 0);
    } else if (dt === "recruit") {
      s.recruitFraction = action.recruitFraction ?? 0;
      s.recruitMask = 1;
    } else if (dt === "battleTarget") {
      s.battleTargetIdx = action.battleTarget ?? -1;
      const mask = Float32Array.from({length: BATTLE_TARGET_DIM},
        (_, i) => i < NUM_NODES && rec.state[BTGT_MASK_OFFSET + i] > 0 ? 1 : 0);
      mask[BATTLE_TARGET_STOP] = 1;
      s.battleTargetMask = mask;
    } else if (dt === "battleSelect") {
      s.battleSelect = action.battleSelect ?? 0;
      s.battleSelectMask = 1;
    } else if (dt === "battleAllocate") {
      s.killFraction = action.killFraction ?? 0;
      s.killFracMask = 1;
    } else if (dt === "battleRetreat") {
      s.battleRetreat = action.battleRetreat ?? 0;
      s.retreatMask = 1;
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
