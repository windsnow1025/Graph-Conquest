import type GameSystem from "../lib/GameSystem";
import type Battle from "../lib/Battle";
import {executeNNTurn, executeNNTurnSteps, executeNNDefenderPhase} from "./TurnExecutor";
import {NNModel} from "./nn/NNModel";
import {greedyTurnSteps, greedyDefenderPhase} from "../../training/src/GreedyAI";

let nnModel: NNModel | null = null;
let modelLoadAttempted = false;

async function ensureModelLoaded(): Promise<void> {
  if (modelLoadAttempted) return;
  modelLoadAttempted = true;

  try {
    const model = new NNModel();
    await model.load("/model/model.json");
    nnModel = model;
    console.log("NN model loaded successfully");
  } catch (e) {
    console.warn("NN model not found, AI will skip turns:", e);
    nnModel = null;
  }
}

export async function aiTakeTurn(game: GameSystem): Promise<void> {
  if (game.gameOver || game.currentPlayer.defeated) {
    game.endTurn();
    return;
  }

  await ensureModelLoaded();

  if (nnModel?.isLoaded()) {
    executeNNTurn(game, nnModel);
  } else {
    game.endTurn();
  }
}

export async function aiTurnSteps(game: GameSystem): Promise<Generator<void> | null> {
  if (game.gameOver || game.currentPlayer.defeated) {
    game.endTurn();
    return null;
  }

  await ensureModelLoaded();

  if (nnModel?.isLoaded()) {
    return executeNNTurnSteps(game, nnModel);
  } else {
    game.endTurn();
    return null;
  }
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
    await ensureModelLoaded();
    if (nnModel?.isLoaded()) {
      executeNNDefenderPhase(game, nnModel, battle);
    } else {
      battle.executeNeutralDefenderTurn();
    }
  }
}
