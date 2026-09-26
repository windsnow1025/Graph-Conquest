import type GameSystem from "../lib/GameSystem";
import type Battle from "../lib/Battle";
import {executeNNTurnSteps, executeNNDefenderPhase} from "./TurnExecutor";
import {NNModel} from "./nn/NNModel";
import {greedyTurnSteps, greedyDefenderPhase} from "../../training/src/GreedyAI";

let modelLoading: Promise<NNModel> | null = null;

/** The published web model, loaded once; a failed load propagates to the caller and is retried on the next call. */
function loadModel(): Promise<NNModel> {
  if (!modelLoading) {
    const model = new NNModel();
    modelLoading = model.load("/model/model.json").then(() => model, (e: unknown) => {
      modelLoading = null;
      throw e;
    });
  }
  return modelLoading;
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
