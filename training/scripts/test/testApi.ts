/**
 * Conformance of GameApi with the engine. The consumers that drive the engine
 * directly (random opponents, the greedy labeler, the NN executor) play seeded
 * games; every command they issue is replayed on a GameApi in its identifier
 * form (an army by owner, location and unit type; units by state and count),
 * and the verdict and the saved state are compared after each one. After
 * every turn and at every battle start, getState() and every query are
 * compared with the engine as well. A first section checks the exit (no live
 * object, no shared state) and the rejection of illegal input.
 *
 * Usage: npx tsx training/scripts/test/testApi.ts
 */
import {setupBackend} from "../../src/setupBackend";
import {NNModel} from "../../../src/AI/nn/NNModel";
import {nodeFileSystem} from "../../src/nodeIO";
import {executeNNTurn} from "../../../src/AI/TurnExecutor";
import GameSystem from "../../../src/lib/GameSystem";
import GameApi from "../../../src/lib/GameApi";
import type {ArmyId, UnitSelection} from "../../../src/lib/GameApi";
import Player from "../../../src/lib/Player";
import type Army from "../../../src/lib/Army";
import type Unit from "../../../src/lib/Unit";
import type Battle from "../../../src/lib/Battle";
import Config from "../../../src/lib/data/Config";
import {randomTurn} from "../../src/Opponents";
import {greedyTurn} from "../../src/GreedyAI";
import {createRandomizedGame, seededRandom} from "../../src/trainUtils";
import {isDeepStrictEqual} from "util";
import * as path from "path";

const SEED = 20261003;
const RANDOM_GAMES = 40;
const GREEDY_GAMES = 3;

const stats = {games: 0, commands: 0, rejected: 0, battles: 0, fullChecks: 0, answers: 0};

function fail(message: string): never {
  throw new Error(`MISMATCH: ${message}`);
}

function same(label: string, actual: unknown, expected: unknown): void {
  stats.answers++;
  if (!isDeepStrictEqual(actual, expected)) {
    fail(`${label}\n  api:    ${JSON.stringify(actual)}\n  engine: ${JSON.stringify(expected)}`);
  }
}

function idOf(game: GameSystem, army: Army): ArmyId {
  const owner = game.allPlayers.find(player => player.armies.includes(army))!;
  return {owner: owner.name, location: army.location, unitType: army.unitType};
}

/** The unit list a consumer passes to the engine, as states and counts in the same order. */
function selectionOf(units: Unit[]): UnitSelection[] {
  const entries: UnitSelection[] = [];
  for (const unit of units) {
    const last = entries.at(-1);
    if (last && last.currentHealth === unit.currentHealth && last.remainingMoves === unit.remainingMoves && last.canAttack === unit.canAttack) {
      last.count++;
    } else {
      entries.push({currentHealth: unit.currentHealth, remainingMoves: unit.remainingMoves, canAttack: unit.canAttack, count: 1});
    }
  }
  return entries;
}

/** getState() and every query of the API against the engine. */
function checkAll(game: GameSystem, api: GameApi, after: string): void {
  stats.fullChecks++;
  const state = api.getState();
  if (state instanceof GameSystem || state.players[0] instanceof Player) fail(`getState returned a live object after ${after}`);
  if (!isDeepStrictEqual(state, structuredClone(game))) fail(`getState after ${after}`);

  same("getWinner", api.getWinner(), game.winner?.name ?? null);
  same("isGameOver", api.isGameOver(), game.gameOver);
  for (const player of game.allPlayers) same("getUpkeep", api.getUpkeep(player.name), game.getUpkeep(player));
  same("getRecruitLocations", api.getRecruitLocations(), game.recruitLocations);
  same("getAttackableLocations", api.getAttackableLocations(), game.attackableLocations);

  const enemyLocations = game.enemyLocations;
  for (const army of game.currentPlayer.armies) {
    const id = idOf(game, army);
    const movable = army.getMovableLocations(game.gameMap, enemyLocations);
    same("getMovableLocations", api.getMovableLocations(id), movable);
    for (const destination of movable) {
      same("getMoveCandidates", api.getMoveCandidates(id, destination), structuredClone(army.getMoveCandidates(destination, game.gameMap, enemyLocations)));
    }
    same("hasAttackTargets", api.hasAttackTargets(id), game.hasAttackTargets(army));
  }
  for (const target of game.attackableLocations) {
    same("getArmiesInRange", api.getArmiesInRange(target), game.getArmiesInRange(target).map(army => idOf(game, army)));
  }
  for (const unitType of Object.keys(game.unitStatsMap)) {
    for (const count of [1, 7, 1000]) same("canBuy", api.canBuy(unitType, count), game.currentPlayer.canBuy(unitType, count));
  }
  const nodes = [...game.gameMap.nodes.keys()];
  for (const from of nodes.slice(0, 3)) {
    for (const to of nodes) same("getDistance", api.getDistance(from, to), game.gameMap.getDistance(from, to));
  }

  const battle = game.currentBattle;
  if (!battle) return;
  for (const army of battle.allArmies) {
    const id = idOf(game, army);
    same("canAct", api.canAct(id), battle.canAct(army));
    same("getRemainingAttacks", api.getRemainingAttacks(id), battle.getRemainingAttacks(army));
    const targets = battle.getTargetsInRange(army);
    same("getTargetsInRange", api.getTargetsInRange(id), targets.map(target => idOf(game, target)));
    for (const target of targets) same("getUnitsNeeded", api.getUnitsNeeded(id, idOf(game, target)), battle.getUnitsNeeded(army, target));
  }
}

/** The wrappers of shadow() are own properties of engine objects; non-enumerable, they stay out of the engine's cloned state. */
function hide(target: object): void {
  for (const [key, value] of Object.entries(target)) {
    if (typeof value === "function") Object.defineProperty(target, key, {enumerable: false});
  }
}

/** Replays every command issued on the engine `game` onto a GameApi started from the same state. */
function shadow(game: GameSystem): void {
  stats.games++;
  const api = GameApi.fromJSON(game.toJSON());
  let depth = 0;

  const mirror = (command: string, expected: boolean, actual: boolean): void => {
    stats.commands++;
    if (!expected) stats.rejected++;
    if (expected !== actual) fail(`verdict of ${command}: engine ${expected}, api ${actual}`);
    if (JSON.stringify(game.toJSON()) !== JSON.stringify(api.toJSON())) fail(`state after ${command}`);
  };

  const wrapBattle = (battle: Battle): void => {
    stats.battles++;
    const allocate = battle.allocateAttack.bind(battle);
    const retreat = battle.retreat.bind(battle);
    battle.allocateAttack = (army, allocations) => {
      // The passive defender's turn runs inside the engine, and inside the API's engine as well
      if (depth > 0) return allocate(army, allocations);
      const id = idOf(game, army);
      const list = [...allocations].map(([target, count]) => ({target: idOf(game, target), count}));
      depth++;
      const expected = allocate(army, allocations);
      depth--;
      mirror("allocateAttack", expected, api.allocateAttack(id, list));
      return expected;
    };
    battle.retreat = () => {
      const expected = retreat();
      mirror("retreat", expected, api.retreat());
      return expected;
    };
    hide(battle);
  };

  const recruit = game.recruitPlayerArmy.bind(game);
  const move = game.movePlayerUnits.bind(game);
  const disband = game.disbandPlayerUnits.bind(game);
  const start = game.startBattle.bind(game);
  const resolve = game.resolveBattle.bind(game);
  const end = game.endTurn.bind(game);
  game.recruitPlayerArmy = (unitType, location, count) => {
    const army = recruit(unitType, location, count);
    mirror("recruit", army !== null, api.recruit(unitType, location, count));
    return army;
  };
  game.movePlayerUnits = (army, location, units) => {
    const id = idOf(game, army);
    const selection = selectionOf(units);
    const expected = move(army, location, units);
    mirror("moveUnits", expected, api.moveUnits(id, location, selection));
    return expected;
  };
  game.disbandPlayerUnits = (army, units) => {
    const id = idOf(game, army);
    const selection = selectionOf(units);
    const expected = disband(army, units);
    mirror("disbandUnits", expected, api.disbandUnits(id, selection));
    return expected;
  };
  game.startBattle = (target, selections) => {
    const list = [...selections].map(([army, units]) => ({army: idOf(game, army), units: selectionOf(units)}));
    const battle = start(target, selections);
    mirror("startBattle", battle !== null, api.startBattle(target, list));
    if (battle) {
      wrapBattle(battle);
      checkAll(game, api, "startBattle");
    }
    return battle;
  };
  game.resolveBattle = () => {
    const expected = resolve();
    mirror("resolveBattle", expected, api.resolveBattle());
    return expected;
  };
  game.endTurn = () => {
    const expected = end();
    mirror("endTurn", expected, api.endTurn());
    checkAll(game, api, "endTurn");
    return expected;
  };
  hide(game);
  checkAll(game, api, "start");
}

/** The exit of the API and its verdicts on illegal input. */
function checkExitAndInput(): void {
  const check = (label: string, ok: boolean): void => {
    if (!ok) fail(label);
    console.log(`ok  ${label}`);
  };
  const api = GameApi.create(Config);
  const blueInfantry: ArmyId = {owner: "Blue", location: "Blue Home", unitType: "Infantry"};
  const nowhere: ArmyId = {owner: "Blue", location: "Nowhere", unitType: "Infantry"};
  const oneUnit: UnitSelection[] = [{currentHealth: 10, remainingMoves: 1, canAttack: true, count: 1}];
  const recruited = (count: number): UnitSelection[] => [{currentHealth: 10, remainingMoves: 0, canAttack: false, count}];

  check("the engine is not reachable through the API object", (api as unknown as {game: unknown}).game === undefined);
  const copy = api.getState();
  copy.players[0].money = 999999;
  copy.players[0].armies.length = 0;
  copy.nodeOwnership.clear();
  copy.turnCount = 50;
  const fresh = api.getState();
  check("writing to a returned state changes nothing", fresh.players[0].money !== 999999 && fresh.nodeOwnership.size === 16 && fresh.turnCount === 1);

  const before = JSON.stringify(api.toJSON());
  check("a fractional count is rejected", !api.recruit("Infantry", "Blue Home", 2.5));
  check("a negative count is rejected", !api.recruit("Infantry", "Blue Home", -1));
  check("an unknown unit type is rejected", !api.recruit("Dragon", "Blue Home", 1) && !api.recruit("toString", "Blue Home", 1));
  check("an unknown location is rejected", !api.recruit("Infantry", "Nowhere", 1));
  check("a location of another player is rejected", !api.recruit("Infantry", "Red Home", 1));
  check("an unknown army is rejected", !api.moveUnits(nowhere, "Center", oneUnit) && !api.disbandUnits(nowhere, oneUnit));
  check("a battle with unknown armies is rejected", !api.startBattle("Blue to Center", [{army: nowhere, units: oneUnit}]) && !api.startBattle("Nowhere", []));
  check("battle commands outside a battle are rejected", !api.allocateAttack(blueInfantry, []) && !api.retreat() && !api.resolveBattle());
  check("rejected commands leave the state untouched", JSON.stringify(api.toJSON()) === before);
  check("queries about unknown things give the empty answer",
    api.getMovableLocations(nowhere).length === 0 && api.getMoveCandidates(nowhere, "Center").length === 0
    && api.getUpkeep("Nobody") === undefined && !api.canAct(nowhere) && api.getTargetsInRange(nowhere).length === 0
    && api.getUnitsNeeded(nowhere, nowhere) === undefined && api.getRemainingAttacks(nowhere) === undefined
    && !api.hasAttackTargets(nowhere) && api.getArmiesInRange("Nowhere").length === 0 && api.getDistance("Nowhere", "Center") === undefined
    && !api.canBuy("Dragon", 1) && !api.canBuy("Infantry", 1.5));

  check("a legal recruit succeeds", api.recruit("Infantry", "Blue Home", 5) && api.getState().players[0].armies[0].units.length === 5);
  check("more units than the army has in that state is rejected", !api.disbandUnits(blueInfantry, recruited(6)));
  check("a state no unit is in is rejected", !api.disbandUnits(blueInfantry, oneUnit));
  check("a legal disband by state and count succeeds", api.disbandUnits(blueInfantry, recruited(2)) && api.getState().players[0].armies[0].units.length === 3);

  const fork = api.fork();
  fork.endTurn();
  check("a fork is an API object on a separate game", fork instanceof GameApi && fork.getState().currentPlayerIndex === 1 && api.getState().currentPlayerIndex === 0);
  check("save and load round-trip", isDeepStrictEqual(GameApi.fromJSON(api.toJSON()).toJSON(), api.toJSON()));
}

async function main() {
  checkExitAndInput();

  await setupBackend();
  Math.random = seededRandom(SEED);
  for (let g = 0; g < RANDOM_GAMES; g++) {
    const game = createRandomizedGame();
    shadow(game);
    while (!game.gameOver) randomTurn(game);
  }
  console.log(`random games: ${JSON.stringify(stats)}`);

  for (let g = 0; g < GREEDY_GAMES; g++) {
    const game = createRandomizedGame();
    shadow(game);
    while (!game.gameOver) {
      if (game.currentPlayerIndex === g % 3) greedyTurn(game);
      else randomTurn(game);
    }
  }
  console.log(`greedy games: ${JSON.stringify(stats)}`);

  const model = new NNModel();
  await model.load(nodeFileSystem(path.resolve("public/model")));
  const game = new GameSystem(Config);
  shadow(game);
  while (!game.gameOver) executeNNTurn(game, model);
  model.dispose();
  console.log(`NN game:      ${JSON.stringify(stats)}`);
  console.log("=== PASS ===");
}

main().catch(e => { console.error(e); process.exit(1); });
