/**
 * Conformance of GameAPI with the engine, grown with the API one step at a
 * time: each step adds the checks of what it added. Seeded games are played
 * on the engine as in traceCore.ts. At the start of every game, after every
 * turn and at every battle start, a GameAPI built from the engine's save is
 * compared with the engine; every command of the engine that the API offers
 * is also issued on a GameAPI built from the engine's save just before the
 * engine executes it, and the verdicts and the saves are compared. A first
 * section checks the exit and the verdicts on illegal input on the default
 * configuration.
 *
 * Step 1, the boundary: create, fromJSON, toJSON, getState.
 * Step 2, the turn and recruitment: endTurn, recruit, getRecruitLocations,
 * canBuy, isGameOver, getWinner, getUpkeep.
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

const stats = {games: 0, commands: 0, rejected: 0, checks: 0, answers: 0};

function fail(message: string): never {
  throw new Error(`MISMATCH: ${message}`);
}

function check(label: string, ok: boolean): void {
  if (!ok) fail(label);
  console.log(`ok  ${label}`);
}

function same(label: string, actual: unknown, expected: unknown): void {
  stats.answers++;
  if (!isDeepStrictEqual(actual, expected)) {
    fail(`${label}\n  api:    ${JSON.stringify(actual)}\n  engine: ${JSON.stringify(expected)}`);
  }
}

/** The exit and the verdicts on illegal input, on the default configuration. */
function checkExitAndInput(): void {
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

  const before = JSON.stringify(api.toJSON());
  check("a fractional count is rejected", !api.recruit("Infantry", "Blue Home", 2.5));
  check("a count of zero or less is rejected", !api.recruit("Infantry", "Blue Home", 0) && !api.recruit("Infantry", "Blue Home", -1));
  check("an unknown unit type is rejected", !api.recruit("Dragon", "Blue Home", 1) && !api.recruit("toString", "Blue Home", 1));
  check("an unknown location is rejected", !api.recruit("Infantry", "Nowhere", 1));
  check("a location of another player is rejected", !api.recruit("Infantry", "Red Home", 1));
  check("rejected commands leave the state untouched", JSON.stringify(api.toJSON()) === before);
  check("canBuy answers false to malformed input", !api.canBuy("Dragon", 1) && !api.canBuy("Infantry", 1.5) && !api.canBuy("Infantry", 0));
  check("getUpkeep of an unknown player is undefined", api.getUpkeep("Nobody") === undefined);
  check("a legal recruit succeeds and shows in the state", api.recruit("Infantry", "Blue Home", 5) && api.getState().players[0].armies[0].units.length === 5);
  check("endTurn passes the turn", api.endTurn() && api.getState().currentPlayerIndex === 1);
}

/** The state and every query of an API built from the engine's save, against the engine. */
function checkAll(game: GameSystem, after: string): void {
  stats.checks++;
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

  same("getWinner", api.getWinner(), game.winner?.name ?? null);
  same("isGameOver", api.isGameOver(), game.gameOver);
  for (const player of game.allPlayers) same("getUpkeep", api.getUpkeep(player.name), game.getUpkeep(player));
  same("getRecruitLocations", api.getRecruitLocations(), game.recruitLocations);
  for (const unitType of Object.keys(game.unitStatsMap)) {
    for (const count of [1, 7, 1000]) same("canBuy", api.canBuy(unitType, count), game.currentPlayer.canBuy(unitType, count));
  }
}

/**
 * Issues a command on an API built from the engine's save, then runs it on the
 * engine, and compares the verdicts and the saves. Returns the engine's result.
 */
function mirror<T>(game: GameSystem, command: string, onApi: (api: GameAPI) => boolean, onEngine: () => T, verdictOf: (result: T) => boolean): T {
  stats.commands++;
  const api = GameAPI.fromJSON(game.toJSON());
  const actual = onApi(api);
  const result = onEngine();
  const expected = verdictOf(result);
  if (!expected) stats.rejected++;
  if (actual !== expected) fail(`verdict of ${command}: engine ${expected}, api ${actual}`);
  if (JSON.stringify(api.toJSON()) !== JSON.stringify(game.toJSON())) fail(`state after ${command}`);
  return result;
}

/** The wrappers of watch() are own properties of the engine; non-enumerable, they stay out of the engine's cloned state. */
function hide(target: object): void {
  for (const [key, value] of Object.entries(target)) {
    if (typeof value === "function") Object.defineProperty(target, key, {enumerable: false});
  }
}

/** Mirrors the commands the API offers and checks the state and the queries at every check point. */
function watch(game: GameSystem): void {
  stats.games++;
  const recruit = game.recruitPlayerArmy.bind(game);
  const start = game.startBattle.bind(game);
  const end = game.endTurn.bind(game);
  game.recruitPlayerArmy = (unitType, location, count) =>
    mirror(game, "recruit", api => api.recruit(unitType, location, count), () => recruit(unitType, location, count), army => army !== null);
  game.startBattle = (target, selections) => {
    const battle = start(target, selections);
    if (battle) checkAll(game, "startBattle");
    return battle;
  };
  game.endTurn = () => {
    const expected = mirror(game, "endTurn", api => api.endTurn(), end, verdict => verdict);
    checkAll(game, "endTurn");
    return expected;
  };
  hide(game);
  checkAll(game, "start");
}

async function main() {
  checkExitAndInput();

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
