/**
 * Conformance of GameAPI with the engine, grown with the API one step at a
 * time: each step adds the checks of what it added. Seeded games are played
 * on the engine as in traceCore.ts; at the start of every game, after every
 * turn and at every battle start, a GameAPI built from the engine's save is
 * compared with the engine. A first section checks the exit on the default
 * configuration.
 *
 * Step 1, the boundary: create, fromJSON, toJSON, getState.
 *
 * Usage: npx tsx training/scripts/test/testAPI.ts
 */
import {setupBackend} from "../../src/setupBackend";
import {NNModel} from "../../../src/AI/nn/NNModel";
import {nodeFileSystem} from "../../src/nodeIO";
import {executeNNTurn} from "../../../src/AI/TurnExecutor";
import GameSystem from "../../../src/lib/GameSystem";
import GameAPI from "../../../src/lib/GameAPI";
import Player from "../../../src/lib/Player";
import DefaultConfig from "../../../src/lib/data/DefaultConfig";
import {randomTurn} from "../../src/Opponents";
import {greedyTurn} from "../../src/GreedyAI";
import {createRandomizedGame, seededRandom} from "../../src/trainUtils";
import {isDeepStrictEqual} from "util";
import * as path from "path";

const SEED = 20261003;
const RANDOM_GAMES = 40;
const GREEDY_GAMES = 3;

const stats = {games: 0, stateChecks: 0};

function fail(message: string): never {
  throw new Error(`MISMATCH: ${message}`);
}

function check(label: string, ok: boolean): void {
  if (!ok) fail(label);
  console.log(`ok  ${label}`);
}

/** The exit, on the default configuration. */
function checkExit(): void {
  const api = GameAPI.create(DefaultConfig);
  const engine = new GameSystem(DefaultConfig);
  check("create builds the same game as the engine", isDeepStrictEqual(api.toJSON(), engine.toJSON()));
  check("save and load round-trip", isDeepStrictEqual(GameAPI.fromJSON(api.toJSON()).toJSON(), api.toJSON()));

  // The lines under @ts-expect-error check the types: each would compile if the exit exposed what it must not
  // @ts-expect-error: the engine is private
  check("the engine is not reachable through the API object", api.game === undefined);
  const state = api.getState();
  check("a state is plain data", Object.getPrototypeOf(state) === Object.prototype && !(state instanceof GameSystem));
  // @ts-expect-error: a state carries no engine method
  check("a state carries no engine method", state.endTurn === undefined);
  // @ts-expect-error: a state carries no engine getter
  check("a state carries no engine getter", state.currentPlayer === undefined);
  // @ts-expect-error: a player of a state carries no Player method
  check("a player of a state carries no Player method", state.players[0].canBuy === undefined && !(state.players[0] instanceof Player));
  check("Maps, Sets and shared references survive the exit",
    state.nodeOwnership instanceof Map && state.gameMap.nodes instanceof Map
    && state.gameMap.nodes.get("Blue Home")!.connections instanceof Set
    && state.nodeOwnership.get("Blue Home") === state.players[0]);

  state.players[0].money = 999999;
  state.neutralPlayer.armies.length = 0;
  state.nodeOwnership.clear();
  state.turnCount = 50;
  check("writing to a returned state changes nothing", isDeepStrictEqual(api.getState(), structuredClone(engine)));
}

/** getState() and toJSON() of an API built from the engine's save, against the engine. */
function checkState(game: GameSystem, after: string): void {
  stats.stateChecks++;
  const api = GameAPI.fromJSON(game.toJSON());
  const state = api.getState();
  if (state instanceof GameSystem || state.players[0] instanceof Player) fail(`getState returned a live object after ${after}`);
  if (!isDeepStrictEqual(state, structuredClone(game))) fail(`getState after ${after}`);
  if (!isDeepStrictEqual(api.toJSON(), game.toJSON())) fail(`toJSON after ${after}`);

  const battle = state.currentBattle;
  if (battle) {
    const participants = [...state.players, state.neutralPlayer];
    const shared = participants.includes(battle.attackerPlayer) && participants.includes(battle.defenderPlayer)
      && battle.attackerArmies.every(army => battle.attackerPlayer.armies.includes(army))
      && battle.defenderArmies.every(army => battle.defenderPlayer.armies.includes(army));
    if (!shared) fail(`the battle of a state does not share its objects with the players after ${after}`);
  }
}

/** Checks the state at the start of the game, after every turn and at every battle start. */
function watch(game: GameSystem): void {
  stats.games++;
  const start = game.startBattle.bind(game);
  const end = game.endTurn.bind(game);
  game.startBattle = (target, selections) => {
    const battle = start(target, selections);
    if (battle) checkState(game, "startBattle");
    return battle;
  };
  game.endTurn = () => {
    const expected = end();
    checkState(game, "endTurn");
    return expected;
  };
  // The wrappers are own properties of the engine; non-enumerable, they stay out of the engine's cloned state
  for (const key of ["startBattle", "endTurn"]) Object.defineProperty(game, key, {enumerable: false});
  checkState(game, "start");
}

async function main() {
  checkExit();

  await setupBackend();
  Math.random = seededRandom(SEED);
  for (let g = 0; g < RANDOM_GAMES; g++) {
    const game = createRandomizedGame();
    watch(game);
    while (!game.gameOver) randomTurn(game);
  }
  console.log(`random games: ${JSON.stringify(stats)}`);

  for (let g = 0; g < GREEDY_GAMES; g++) {
    const game = createRandomizedGame();
    watch(game);
    while (!game.gameOver) {
      if (game.currentPlayerIndex === g % 3) greedyTurn(game);
      else randomTurn(game);
    }
  }
  console.log(`greedy games: ${JSON.stringify(stats)}`);

  const model = new NNModel();
  await model.load(nodeFileSystem(path.resolve("public/model")));
  const game = new GameSystem(DefaultConfig);
  watch(game);
  while (!game.gameOver) executeNNTurn(game, model);
  model.dispose();
  console.log(`NN game:      ${JSON.stringify(stats)}`);
  console.log("=== PASS ===");
}

main().catch(e => { console.error(e); process.exit(1); });
