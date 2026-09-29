import type GameSystem from "../lib/GameSystem";
import type Battle from "../lib/Battle";
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

export async function aiTurnSteps(game: GameSystem): Promise<Generator<void> | null> {
  if (game.gameOver || game.currentPlayer.defeated) {
    game.endTurn();
    return null;
  }
  return executeNNTurnSteps(game, await loadModel());
}

export function greedyTurnStepsUI(game: GameSystem): Generator<void> | null {
  if (game.gameOver || game.currentPlayer.defeated) {
    game.endTurn();
    return null;
  }
  return greedyTurnSteps(game);
}

export async function aiDefenderPhase(game: GameSystem, battle: Battle, mode: string): Promise<void> {
  if (mode === "greedy") {
    greedyDefenderPhase(game, battle);
  } else {
    executeNNDefenderPhase(game, await loadModel(), battle);
  }
}
