import type GameAPI from "../lib/GameAPI";
import type {BattleData} from "../lib/Battle";
import {executeNNTurnSteps, executeNNDefenderPhase} from "./TurnExecutor";
import {NNModel} from "./nn/NNModel";
import {WEB_MODEL_FILE, webModelHandler} from "./nn/WebModel";
import type {WebModelFile} from "./nn/WebModel";
import {greedyTurnSteps, greedyDefenderPhase} from "../../training/src/GreedyAI";

const WEB_MODEL_URL = `/model/${WEB_MODEL_FILE}`;

let modelLoading: Promise<NNModel> | null = null;

/** The published web model, loaded once; a failed load propagates to the caller and is retried on the next call. */
function loadModel(): Promise<NNModel> {
  if (!modelLoading) {
    modelLoading = loadWebModel().catch((e: unknown) => {
      modelLoading = null;
      throw e;
    });
  }
  return modelLoading;
}

async function loadWebModel(): Promise<NNModel> {
  const response = await fetch(WEB_MODEL_URL);
  if (!response.ok) throw new Error(`${WEB_MODEL_URL}: ${response.status} ${response.statusText}`);
  const model = new NNModel();
  await model.load(webModelHandler(await response.json() as WebModelFile));
  return model;
}

/** True when the current player cannot play: the game is over or the player is defeated. */
function cannotPlay(api: GameAPI): boolean {
  const state = api.getState();
  return api.isGameOver() || state.players[state.currentPlayerIndex].defeated;
}

export async function aiTurnSteps(api: GameAPI): Promise<Generator<void> | null> {
  if (cannotPlay(api)) {
    api.endTurn();
    return null;
  }
  return executeNNTurnSteps(api, await loadModel());
}

export function greedyTurnStepsUI(api: GameAPI): Generator<void> | null {
  if (cannotPlay(api)) {
    api.endTurn();
    return null;
  }
  return greedyTurnSteps(api);
}

export async function aiDefenderPhase(api: GameAPI, battle: BattleData, mode: string): Promise<void> {
  if (mode === "greedy") {
    greedyDefenderPhase(api, battle);
  } else {
    executeNNDefenderPhase(api, await loadModel(), battle);
  }
}
