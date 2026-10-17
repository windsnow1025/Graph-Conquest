/**
 * Conformance of GameAPI with the engine, grown with the API one step at a
 * time: each step adds the checks of what it added. Seeded games are played
 * on the engine as in traceCore.ts. At the start of every game, after every
 * turn and at every battle start, a GameAPI built from the engine's save is
 * compared with the engine; every command of the engine that the API offers
 * is also issued on a GameAPI built from the engine's save just before the
 * engine executes it, and the verdicts and the saves are compared. A first
 * section checks the exit, the input errors and the verdicts on the default
 * configuration.
 *
 * Step 1, the boundary: create, fromJSON, toJSON, getState.
 * Step 2, the turn and recruitment: endTurn, recruit, getRecruitLocations,
 * canBuy, isGameOver, getWinner, getUpkeep.
 * Step 3, the army identifier and unit selection: moveUnits, disbandUnits,
 * getMovableLocations, getMoveCandidates, getDistance.
 * Step 4, starting a battle: startBattle, getAttackableLocations,
 * getArmiesInRange, hasAttackTargets.
 * Step 5, fighting: canAct, getTargetsInRange, getUnitsNeeded,
 * getRemainingAttacks, allocateAttack, retreat.
 * Step 6, the fork: fork.
 *
 * Usage: npx tsx training/scripts/test/testAPI.ts
 */
import {setupBackend} from "../../src/setupBackend";
import {NNModel} from "../../../src/AI/nn/NNModel";
import {nodeFileSystem} from "../../src/nodeIO";
import {executeNNTurn} from "../../../src/AI/TurnExecutor";
import GameSystem from "../../../src/lib/GameSystem";
import GameAPI, {InputError} from "../../../src/lib/GameAPI";
import type {ArmyId, ArmySelection, UnitSelection} from "../../../src/lib/GameAPI";
import Player from "../../../src/lib/Player";
import type Army from "../../../src/lib/Army";
import type Unit from "../../../src/lib/Unit";
import DefaultConfig from "../../../src/lib/data/DefaultConfig";
import {randomTurn} from "../../src/Opponents";
import {greedyTurn} from "../../src/GreedyAI";
import {createRandomizedGame, seededRandom} from "../../src/trainUtils";
import {selectionsOf} from "../../../src/AI/nn/UnitGroups";
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

/** True when the call fails with an InputError, false when it returns; any other error propagates. */
function inputError(call: () => unknown): boolean {
  try {
    call();
  } catch (e) {
    if (!(e instanceof InputError)) throw e;
    return true;
  }
  return false;
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

/** The army of the current player that holds the units a consumer moves or disbands. */
function armyOf(game: GameSystem, units: Unit[]): Army {
  return game.currentPlayer.getArmyOf(units[0])!;
}

/** The unit list a consumer passes to startBattle, as one army with its selections per army, in order of first appearance. */
function attackersOf(game: GameSystem, units: Unit[]): ArmySelection[] {
  const unitsByArmy = new Map<Army, Unit[]>();
  for (const unit of units) {
    const army = game.currentPlayer.getArmyOf(unit)!;
    const armyUnits = unitsByArmy.get(army);
    if (armyUnits) armyUnits.push(unit);
    else unitsByArmy.set(army, [unit]);
  }
  return [...unitsByArmy].map(([army, armyUnits]) => ({army: idOf(game, army), selections: selectionsOf(armyUnits)}));
}

/** The exit, the input errors and the verdicts, on the default configuration. */
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
  check("a fractional count is an input error", inputError(() => api.recruit("Infantry", "Blue Home", 2.5)));
  check("a count of zero is an input error", inputError(() => api.recruit("Infantry", "Blue Home", 0)));
  check("a negative count is an input error", inputError(() => api.recruit("Infantry", "Blue Home", -1)));
  check("an unknown unit type is an input error", inputError(() => api.recruit("Dragon", "Blue Home", 1)));
  check("an inherited name is not a unit type", inputError(() => api.recruit("toString", "Blue Home", 1)));
  check("an unknown location is an input error", inputError(() => api.recruit("Infantry", "Nowhere", 1)));
  check("a location of another player is the false verdict", !api.recruit("Infantry", "Red Home", 1));
  check("input errors and rejections leave the state untouched", JSON.stringify(api.toJSON()) === before);
  check("canBuy with an unknown unit type is an input error", inputError(() => api.canBuy("Dragon", 1)));
  check("canBuy with a fractional count is an input error", inputError(() => api.canBuy("Infantry", 1.5)));
  check("canBuy answers the rule", api.canBuy("Infantry", 1) && !api.canBuy("Infantry", 1000));
  check("getUpkeep of an unknown player is an input error", inputError(() => api.getUpkeep("Nobody")));
  check("a legal recruit succeeds and shows in the state", api.recruit("Infantry", "Blue Home", 5) && api.getState().players[0].armies[0].units.length === 5);

  const blueInfantry: ArmyId = {owner: "Blue", location: "Blue Home", unitType: "Infantry"};
  const neutralCenter: ArmyId = {owner: "Neutral", location: "Center", unitType: "Infantry"};
  const nowhere: ArmyId = {owner: "Blue", location: "Nowhere", unitType: "Infantry"};
  const recruited = (count: number): UnitSelection[] => [{currentHealth: 10, remainingMoves: 0, canAttack: false, count}];
  const ready = (count: number): UnitSelection[] => [{currentHealth: 10, remainingMoves: 1, canAttack: true, count}];
  const armed = JSON.stringify(api.toJSON());
  check("an unknown army is an input error", inputError(() => api.moveUnits(nowhere, recruited(1), "Blue to Center")));
  check("an unknown owner is an input error", inputError(() => api.disbandUnits({...blueInfantry, owner: "Nobody"}, recruited(1))));
  check("an unknown destination is an input error", inputError(() => api.moveUnits(blueInfantry, recruited(1), "Nowhere")));
  check("a selection with a fractional count is an input error", inputError(() => api.disbandUnits(blueInfantry, [{...recruited(1)[0], count: 1.5}])));
  check("a state no unit is in is an input error", inputError(() => api.disbandUnits(blueInfantry, ready(1))));
  check("more units than the army has in that state is an input error", inputError(() => api.disbandUnits(blueInfantry, recruited(6))));
  check("a state selected twice is an input error", inputError(() => api.disbandUnits(blueInfantry, [...recruited(1), ...recruited(1)])));
  check("an army of another player is an input error", inputError(() => api.moveUnits(neutralCenter, recruited(1), "Blue to Center")) && inputError(() => api.disbandUnits(neutralCenter, recruited(1))));
  check("an empty selection is the false verdict", !api.moveUnits(blueInfantry, [], "Blue to Center") && !api.disbandUnits(blueInfantry, []));
  check("units without moves cannot move", !api.moveUnits(blueInfantry, recruited(1), "Blue to Center"));
  check("input errors and rejections leave the state untouched", JSON.stringify(api.toJSON()) === armed);
  check("a movement query about an unknown army is an input error", inputError(() => api.getMovableLocations(nowhere)));
  check("a movement query with an unknown destination is an input error", inputError(() => api.getMoveCandidates(blueInfantry, "Nowhere")));
  check("movement queries about another player's army are input errors",
    inputError(() => api.getMovableLocations(neutralCenter)) && inputError(() => api.getMoveCandidates(neutralCenter, "Blue to Center")));
  check("getDistance answers the map", api.getDistance("Blue Home", "Center") === 2);
  check("getDistance with an unknown node is an input error", inputError(() => api.getDistance("Nowhere", "Center")));
  check("a legal disband by state and count succeeds", api.disbandUnits(blueInfantry, recruited(2)) && api.getState().players[0].armies[0].units.length === 3);
  check("endTurn passes the turn", api.endTurn() && api.getState().currentPlayerIndex === 1);

  check("after a round the turn is Blue's again", api.endTurn() && api.endTurn() && api.getState().currentPlayerIndex === 0);
  check("getAttackableLocations answers a Set of the enemy nodes in range",
    api.getAttackableLocations() instanceof Set && api.getAttackableLocations().has("Blue to Center") && !api.getAttackableLocations().has("Center"));
  check("getArmiesInRange names the armies", isDeepStrictEqual(api.getArmiesInRange("Blue to Center"), [blueInfantry]) && api.getArmiesInRange("Center").length === 0);
  check("hasAttackTargets answers the rule", api.hasAttackTargets(blueInfantry));
  check("battle queries with unknown or foreign input are input errors",
    inputError(() => api.getArmiesInRange("Nowhere")) && inputError(() => api.hasAttackTargets(nowhere)) && inputError(() => api.hasAttackTargets(neutralCenter)));
  const attackers = (count: number): ArmySelection[] => [{army: blueInfantry, selections: ready(count)}];
  const beforeBattle = JSON.stringify(api.toJSON());
  check("starting a battle with an unknown army is an input error", inputError(() => api.startBattle([{army: nowhere, selections: ready(1)}], "Blue to Center")));
  check("starting a battle at an unknown location is an input error", inputError(() => api.startBattle(attackers(1), "Nowhere")));
  check("an army listed twice is an input error", inputError(() => api.startBattle([...attackers(1), ...attackers(1)], "Blue to Center")));
  check("a battle without attackers is the false verdict", !api.startBattle([], "Blue to Center"));
  check("a battle at a node without an enemy is the false verdict", !api.startBattle(attackers(1), "Blue Home"));
  check("a battle out of range is the false verdict", !api.startBattle(attackers(1), "Center"));
  check("input errors and rejections leave the state untouched", JSON.stringify(api.toJSON()) === beforeBattle);
  check("a legal battle starts and shows in the state", api.startBattle(attackers(3), "Blue to Center") && api.getState().currentBattle?.attackerArmies.length === 1);
  const inBattle: UnitSelection[] = [{currentHealth: 10, remainingMoves: 1, canAttack: false, count: 1}];
  check("recruiting, moving and ending the turn are rejected during a battle",
    !api.recruit("Infantry", "Blue Home", 1) && !api.moveUnits(blueInfantry, inBattle, "B to R") && !api.endTurn());

  const defender: ArmyId = {owner: "Neutral", location: "Blue to Center", unitType: "Infantry"};
  check("battle queries answer the battle",
    api.canAct(blueInfantry) && api.getRemainingAttacks(blueInfantry) === 3
    && isDeepStrictEqual(api.getTargetsInRange(blueInfantry), [defender]) && api.getUnitsNeeded(blueInfantry, defender) === 4);
  check("battle queries about an army outside the battle are input errors",
    inputError(() => api.canAct(nowhere)) && inputError(() => api.getTargetsInRange(neutralCenter))
    && inputError(() => api.getUnitsNeeded(blueInfantry, neutralCenter)) && inputError(() => api.getRemainingAttacks(neutralCenter)));
  check("allocating to an army outside the battle is an input error", inputError(() => api.allocateAttack(blueInfantry, [{target: neutralCenter, count: 1}])));
  check("a fractional allocation is an input error", inputError(() => api.allocateAttack(blueInfantry, [{target: defender, count: 1.5}])));
  check("a target listed twice is an input error", inputError(() => api.allocateAttack(blueInfantry, [{target: defender, count: 1}, {target: defender, count: 1}])));
  check("an army that cannot act now gets the false verdict", !api.allocateAttack(defender, [{target: blueInfantry, count: 1}]));
  check("a legal allocation succeeds and the round advances",
    api.allocateAttack(blueInfantry, [{target: defender, count: 3}]) && api.getState().currentBattle?.round === 2 && api.getRemainingAttacks(blueInfantry) === 2);
  check("retreat ends the battle", api.retreat() && api.getState().currentBattle === null);
  check("battle queries and allocations without a battle are input errors",
    inputError(() => api.canAct(blueInfantry)) && inputError(() => api.allocateAttack(blueInfantry, [])));
  check("retreating without a battle is the false verdict", !api.retreat());
  check("the turn can end after the battle", api.endTurn());

  const fork = api.fork();
  check("a fork is an API object on a copy of the game", fork instanceof GameAPI && isDeepStrictEqual(fork.toJSON(), api.toJSON()));
  check("a fork plays on without the original", fork.endTurn() && fork.getState().currentPlayerIndex === 2 && api.getState().currentPlayerIndex === 1);
  // @ts-expect-error: the engine is private
  check("a fork hides its engine too", fork.game === undefined);
}

/** The state and every query of an API built from the engine's save, against the engine. */
function checkAll(game: GameSystem, after: string): void {
  stats.checks++;
  const api = GameAPI.fromJSON(game.toJSON());
  const state = api.getState();
  if (state instanceof GameSystem || state.players[0] instanceof Player) fail(`getState returned a live object after ${after}`);
  if (!isDeepStrictEqual(state, structuredClone(game))) fail(`getState after ${after}`);
  if (!isDeepStrictEqual(api.toJSON(), game.toJSON())) fail(`toJSON after ${after}`);
  if (!isDeepStrictEqual(api.fork().toJSON(), game.toJSON())) fail(`fork after ${after}`);

  const battleData = state.currentBattle;
  if (battleData) {
    const participants = [...state.players, state.neutralPlayer];
    const shared = participants.includes(battleData.attackerPlayer) && participants.includes(battleData.defenderPlayer)
      && battleData.attackerArmies.every(army => battleData.attackerPlayer.armies.includes(army))
      && battleData.defenderArmies.every(army => battleData.defenderPlayer.armies.includes(army));
    if (!shared) fail(`the battle of a state does not share its objects with the players after ${after}`);
  }

  same("getWinner", api.getWinner(), game.winner?.name ?? null);
  same("isGameOver", api.isGameOver(), game.gameOver);
  for (const player of game.allPlayers) same("getUpkeep", api.getUpkeep(player.name), game.getUpkeep(player));
  same("getRecruitLocations", api.getRecruitLocations(), game.recruitLocations);
  for (const unitType of Object.keys(game.unitStatsMap)) {
    for (const count of [1, 7, 1000]) same("canBuy", api.canBuy(unitType, count), game.currentPlayer.canBuy(unitType, count));
  }

  const enemyLocations = game.enemyLocations;
  for (const army of game.currentPlayer.armies) {
    const id = idOf(game, army);
    const movable = army.getMovableLocations(game.gameMap, enemyLocations);
    same("getMovableLocations", api.getMovableLocations(id), movable);
    same("hasAttackTargets", api.hasAttackTargets(id), game.hasAttackTargets(army));
    for (const destination of movable) {
      same("getMoveCandidates", api.getMoveCandidates(id, destination), structuredClone(army.getMoveCandidates(destination, game.gameMap, enemyLocations)));
    }
  }
  const other = game.allPlayers.find(player => player !== game.currentPlayer && player.armies.length > 0);
  if (other && !inputError(() => api.getMovableLocations(idOf(game, other.armies[0])))) fail("getMovableLocations of another player's army is not an input error");
  const nodes = [...game.gameMap.nodes.keys()];
  for (const from of nodes.slice(0, 3)) {
    for (const to of nodes) same("getDistance", api.getDistance(from, to), game.gameMap.getDistance(from, to));
  }
  same("getAttackableLocations", api.getAttackableLocations(), game.attackableLocations);
  for (const target of game.attackableLocations) {
    same("getArmiesInRange", api.getArmiesInRange(target), game.getArmiesInRange(target).map(army => idOf(game, army)));
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
  const move = game.movePlayerUnits.bind(game);
  const disband = game.disbandPlayerUnits.bind(game);
  const start = game.startBattle.bind(game);
  const allocate = game.allocateAttack.bind(game);
  const retreat = game.retreat.bind(game);
  const end = game.endTurn.bind(game);
  game.recruitPlayerArmy = (unitType, location, count) =>
    mirror(game, "recruit", api => api.recruit(unitType, location, count), () => recruit(unitType, location, count), army => army !== null);
  game.movePlayerUnits = (units, location) => {
    const id = idOf(game, armyOf(game, units));
    const selection = selectionsOf(units);
    return mirror(game, "moveUnits", api => api.moveUnits(id, selection, location), () => move(units, location), verdict => verdict);
  };
  game.disbandPlayerUnits = (units) => {
    const id = idOf(game, armyOf(game, units));
    const selection = selectionsOf(units);
    return mirror(game, "disbandUnits", api => api.disbandUnits(id, selection), () => disband(units), verdict => verdict);
  };
  game.startBattle = (units, target) => {
    const attackers = attackersOf(game, units);
    const battle = mirror(game, "startBattle", api => api.startBattle(attackers, target), () => start(units, target), battle => battle !== null);
    if (battle) checkAll(game, "startBattle");
    return battle;
  };
  game.allocateAttack = (army, allocations) => {
    const id = idOf(game, army);
    const list = [...allocations].map(([target, count]) => ({target: idOf(game, target), count}));
    return mirror(game, "allocateAttack", api => api.allocateAttack(id, list), () => allocate(army, allocations), verdict => verdict);
  };
  game.retreat = () => mirror(game, "retreat", api => api.retreat(), retreat, verdict => verdict);
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
