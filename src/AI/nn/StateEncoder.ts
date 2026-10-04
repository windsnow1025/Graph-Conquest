/**
 * State encoder v10.
 *
 * Encoding order:
 *   1. Game config (5)
 *   2. Unit type stats (18): 3 types × 6 stats
 *   3. Player stats (21): 3 players × 7
 *   4. Per-node (1024): 16 nodes × 64
 *   5. Context (292):
 *      - decision type one-hot (8)
 *      - recruit (20)
 *      - army (36): army info(27) + action type mask(3) + disband group mask(6)
 *      - moveTarget (43): army info(27) + legal destination mask(16)
 *      - moveCount (47): army info(27) + destination(16) + move group mask(4)
 *      - battleTarget (16): attackable node mask
 *      - battleSelect (59): army info(27) + target node(16) + selection state(12) + isDone(1) + commit group mask(3)
 *      - battleAllocate (46)
 *      - battleRetreat (17)
 *
 * Army info (27): location(16) + type(3) + units + avgHp + units per group(6).
 * Unit counts are scaled by cost/100 (units per 100 money).
 *
 * Total: 5 + 18 + 21 + 1024 + 292 = 1360 features.
 */
import type GameSystem from "../../lib/GameSystem";
import type Army from "../../lib/Army";
import type Unit from "../../lib/Unit";
import type Graph from "../../lib/Graph";
import type {DefaultUnitName} from "../../lib/data/DefaultUnitStatsMap.ts";
import {NODE_ORDER, NUM_NODES, UNIT_TYPES, NUM_UNIT_TYPES} from "./GameIndex";
import {NUM_UNIT_GROUPS, NUM_MOVE_GROUPS, NUM_COMMIT_GROUPS, NUM_DISBAND_GROUPS, groupCounts} from "./UnitGroups";
import {NUM_ACTION_TYPES} from "./ActionSpace";

export const DECISION_TYPES = [
  "recruit", "army", "moveTarget", "moveCount", "battleTarget", "battleSelect", "battleAllocate", "battleRetreat",
] as const;
export type DecisionType = typeof DECISION_TYPES[number];
const NUM_DECISION_TYPES = DECISION_TYPES.length;

// Feature dimensions
const GAME_CONFIG_FEATURES = 5;
const UNIT_STATS_FEATURES = 18;
const PLAYER_STATS_FEATURES = 21;
const PER_NODE_FEATURES = 2 + 4 + 4 * NUM_UNIT_TYPES * 2 + NUM_UNIT_TYPES * NUM_UNIT_GROUPS + NUM_NODES; // 64
const ARMY_INFO_FEATURES = NUM_NODES + NUM_UNIT_TYPES + 2 + NUM_UNIT_GROUPS; // 27
const SELECTION_STATE_FEATURES = 2 * NUM_UNIT_TYPES * 2; // selected and remaining, per type (units, avgHp)
const RECRUIT_CONTEXT = 20;
const ARMY_CONTEXT = ARMY_INFO_FEATURES + NUM_ACTION_TYPES + NUM_DISBAND_GROUPS;        // 36
const MOVE_TARGET_CONTEXT = ARMY_INFO_FEATURES + NUM_NODES;                             // 43
const MOVE_COUNT_CONTEXT = ARMY_INFO_FEATURES + NUM_NODES + NUM_MOVE_GROUPS;            // 47
const BATTLE_TARGET_CONTEXT = NUM_NODES;                                                // 16
const BATTLE_SELECT_CONTEXT = ARMY_INFO_FEATURES + NUM_NODES + SELECTION_STATE_FEATURES + 1 + NUM_COMMIT_GROUPS; // 59
const BATTLE_ALLOCATE_CONTEXT = 46;
const BATTLE_RETREAT_CONTEXT = 17;
const MAX_DISTANCE = 6; // diameter of the default map (home to opposite gate); unreachable encodes as 1.0 too

// Context layout: the trunk consumes state[0:CTX_BASE], each head reads its own block
export const CTX_BASE = GAME_CONFIG_FEATURES + UNIT_STATS_FEATURES + PLAYER_STATS_FEATURES + NUM_NODES * PER_NODE_FEATURES; // 1068
export const CTX_DT_OFF = 0;                                 export const CTX_DT_LEN = NUM_DECISION_TYPES;       // 8
export const CTX_REC_OFF = CTX_DT_OFF + CTX_DT_LEN;          export const CTX_REC_LEN = RECRUIT_CONTEXT;         // 20
export const CTX_ARMY_OFF = CTX_REC_OFF + CTX_REC_LEN;       export const CTX_ARMY_LEN = ARMY_CONTEXT;           // 36
export const CTX_MOV_OFF = CTX_ARMY_OFF + CTX_ARMY_LEN;      export const CTX_MOV_LEN = MOVE_TARGET_CONTEXT;     // 43
export const CTX_MCNT_OFF = CTX_MOV_OFF + CTX_MOV_LEN;       export const CTX_MCNT_LEN = MOVE_COUNT_CONTEXT;     // 47
export const CTX_BTGT_OFF = CTX_MCNT_OFF + CTX_MCNT_LEN;     export const CTX_BTGT_LEN = BATTLE_TARGET_CONTEXT;  // 16
export const CTX_BSEL_OFF = CTX_BTGT_OFF + CTX_BTGT_LEN;     export const CTX_BSEL_LEN = BATTLE_SELECT_CONTEXT;  // 59
export const CTX_BALLOC_OFF = CTX_BSEL_OFF + CTX_BSEL_LEN;   export const CTX_BALLOC_LEN = BATTLE_ALLOCATE_CONTEXT; // 46
export const CTX_BRET_OFF = CTX_BALLOC_OFF + CTX_BALLOC_LEN; export const CTX_BRET_LEN = BATTLE_RETREAT_CONTEXT; // 17
const CONTEXT_FEATURES = CTX_BRET_OFF + CTX_BRET_LEN; // 292

export const STATE_SIZE = CTX_BASE + CONTEXT_FEATURES; // 1360

// Mask positions inside their blocks (legality is read back from recorded states)
export const ARMY_ACTION_MASK_OFF = ARMY_INFO_FEATURES;
export const ARMY_DISBAND_MASK_OFF = ARMY_ACTION_MASK_OFF + NUM_ACTION_TYPES;
export const MOV_LEGAL_MASK_OFF = ARMY_INFO_FEATURES;
export const MCNT_GROUP_MASK_OFF = ARMY_INFO_FEATURES + NUM_NODES;
export const BSEL_COMMIT_MASK_OFF = ARMY_INFO_FEATURES + NUM_NODES + SELECTION_STATE_FEATURES + 1;

// ─── Precomputed distance matrix cache ───
let cachedDistMatrix: Float32Array | null = null;
let cachedGraph: Graph | null = null;

function getDistanceMatrix(graph: Graph): Float32Array {
  if (cachedGraph === graph && cachedDistMatrix) return cachedDistMatrix;
  const mat = new Float32Array(NUM_NODES * NUM_NODES);
  for (let i = 0; i < NUM_NODES; i++) {
    for (let j = 0; j < NUM_NODES; j++) {
      const d = graph.getDistance(NODE_ORDER[i], NODE_ORDER[j])!;
      mat[i * NUM_NODES + j] = (d === Infinity ? MAX_DISTANCE : d) / MAX_DISTANCE;
    }
  }
  cachedGraph = graph;
  cachedDistMatrix = mat;
  return mat;
}

// ─── Decision context types ───

export interface RecruitContext {
  type: "recruit";
  locationIdx: number;
  unitType: DefaultUnitName;
  affordableCount: number;
}

export interface ArmyContext {
  type: "army";
  army: Army;
  actionTypeMask: Float32Array; // [3]
  disbandMask: Float32Array;    // [6] 1 = non-empty disband group
}

export interface MoveTargetContext {
  type: "moveTarget";
  army: Army;
  legalMask: Float32Array; // [16] 1 = legal destination, choice is softmax over these
}

export interface MoveCountContext {
  type: "moveCount";
  army: Army;
  destinationIdx: number;
  groupMask: Float32Array; // [4] 1 = move group that can reach the destination
}

export interface BattleTargetContext {
  type: "battleTarget";
  attackableMask: Float32Array; // [16] 1 = attackable node, choice is softmax over these + stop
}

export interface BattleSelectContext {
  type: "battleSelect";
  army: Army | null;          // null for the "done" option
  targetNodeIdx: number;
  selectedPerType: number[];  // [6] = 3 × (committed units, avgHp)
  remainingPerType: number[]; // [6] = 3 × (attack-ready units of the remaining candidates, avgHp)
  isDone: boolean;            // true = this option means "stop adding armies"
  commitMask: Float32Array;   // [3] 1 = non-empty commit group; all 0 for the "done" option
}

export interface BattleAllocateContext {
  type: "battleAllocate";
  army: Army;
  remaining: number;
  enemyArmy: Army;
  attackProgress: number;
  isAttacker: boolean;
  unitsNeeded: number;
}

export interface BattleRetreatContext {
  type: "battleRetreat";
  targetNodeIdx: number;
  attackProgress: number;
}

export type DecisionContext =
  | RecruitContext
  | ArmyContext
  | MoveTargetContext
  | MoveCountContext
  | BattleTargetContext
  | BattleSelectContext
  | BattleAllocateContext
  | BattleRetreatContext;

// ─── Helper: encode army basics ───

function encodeArmyLocation(buf: Float32Array, offset: number, army: Army): number {
  const locIdx = NODE_ORDER.indexOf(army.location);
  if (locIdx >= 0) buf[offset + locIdx] = 1;
  return offset + 16;
}

function encodeArmyType(buf: Float32Array, offset: number, army: Army): number {
  const typeIdx = (UNIT_TYPES as readonly string[]).indexOf(army.unitType);
  if (typeIdx >= 0) buf[offset + typeIdx] = 1;
  return offset + 3;
}

function encodeUnitsHp(units: Unit[]): number {
  if (units.length === 0) return 0;
  let sum = 0;
  for (const u of units) sum += u.currentHealth / u.health;
  return sum / units.length;
}

/** Army info (27): location(16) + type(3) + units + avgHp + units per group(6). */
function encodeArmyInfo(buf: Float32Array, offset: number, game: GameSystem, army: Army): number {
  offset = encodeArmyLocation(buf, offset, army);
  offset = encodeArmyType(buf, offset, army);
  const armyCost = game.unitStatsMap[army.unitType].cost;
  buf[offset++] = army.units.length / (100 / armyCost);
  buf[offset++] = encodeUnitsHp(army.units);
  const counts = groupCounts(army.units);
  for (let g = 0; g < NUM_UNIT_GROUPS; g++) buf[offset++] = counts[g] / (100 / armyCost);
  return offset;
}

/** Per unit type, the unit count and average health ratio of the given (army, units) entries: [units, avgHp] × 3. */
export function perTypeState(entries: Iterable<readonly [Army, Unit[]]>): number[] {
  const state = new Array<number>(NUM_UNIT_TYPES * 2).fill(0);
  const hpSums = new Array<number>(NUM_UNIT_TYPES).fill(0);
  for (const [army, units] of entries) {
    const t = (UNIT_TYPES as readonly string[]).indexOf(army.unitType);
    state[t * 2] += units.length;
    for (const unit of units) hpSums[t] += unit.currentHealth / unit.health;
  }
  for (let t = 0; t < NUM_UNIT_TYPES; t++) {
    const units = state[t * 2];
    state[t * 2 + 1] = units > 0 ? hpSums[t] / units : 0;
  }
  return state;
}

// ─── Main encoder ───

export function encodeState(
  game: GameSystem,
  playerIdx: number,
  context?: DecisionContext,
): Float32Array {
  const buf = new Float32Array(STATE_SIZE);
  let offset = 0;

  const self = game.players[playerIdx];
  const opponents = game.players.filter((_, i) => i !== playerIdx);
  const perspPlayers = [self, opponents[0], opponents[1]];
  const neutral = game.neutralPlayer;
  const allFactions = [self, opponents[0], opponents[1], neutral];

  const distMatrix = getDistanceMatrix(game.gameMap);

  // ─── 1. Game config (5) ───
  buf[offset++] = game.interestRate / 0.10;
  buf[offset++] = game.upkeepRate / 0.20;
  buf[offset++] = game.turnCount / 100;
  buf[offset++] = game.maxTurns / 100;
  buf[offset++] = game.maxArmyAttacks / 20;

  // ─── 2. Unit type stats (3 × 6 = 18) ───
  for (const typeName of UNIT_TYPES) {
    const stats = game.unitStatsMap[typeName];
    buf[offset++] = stats.attack / 9;
    buf[offset++] = stats.defend / 3;
    buf[offset++] = stats.health / 20;
    buf[offset++] = stats.range / 2;
    buf[offset++] = stats.speed / 2;
    buf[offset++] = stats.cost / 2;
  }

  // ─── 3. Player stats (3 × 7 = 21) ───
  for (let i = 0; i < 3; i++) {
    const p = perspPlayers[i];

    buf[offset++] = p.money / 200;

    let nodeIncome = 0;
    for (const [node, nodeOwner] of game.nodeOwnership) {
      if (nodeOwner === p) {
        nodeIncome += game.gameMap.getNodeData(node)?.income ?? 0;
      }
    }
    buf[offset++] = nodeIncome / 68;

    buf[offset++] = (p.money * game.interestRate) / 10;

    buf[offset++] = game.getUpkeep(p) / 20;

    const totalUnits = p.armies.reduce((sum, a) => sum + a.units.length, 0);
    buf[offset++] = totalUnits / 200;

    let nodeCount = 0;
    for (const [, nodeOwner] of game.nodeOwnership) {
      if (nodeOwner === p) nodeCount++;
    }
    buf[offset++] = nodeCount / 16;

    buf[offset++] = p.defeated ? 1 : 0;
  }

  // ─── 4. Per-node features (16 × 64 = 1024) ───
  for (let ni = 0; ni < NUM_NODES; ni++) {
    const nodeName = NODE_ORDER[ni];
    const owner = game.nodeOwnership.get(nodeName) ?? null;
    const nodeData = game.gameMap.getNodeData(nodeName);

    buf[offset++] = (nodeData?.income ?? 0) / 10;
    buf[offset++] = nodeData?.canRecruit ? 1 : 0;

    if (owner === perspPlayers[0]) {
      buf[offset] = 1;
    } else if (owner === perspPlayers[1]) {
      buf[offset + 1] = 1;
    } else if (owner === perspPlayers[2]) {
      buf[offset + 2] = 1;
    } else {
      buf[offset + 3] = 1;
    }
    offset += 4;

    for (const faction of allFactions) {
      for (let t = 0; t < NUM_UNIT_TYPES; t++) {
        let unitCount = 0;
        let hpSum = 0;
        let hpCount = 0;
        for (const army of faction.armies) {
          if (army.location === nodeName && army.unitType === UNIT_TYPES[t]) {
            unitCount += army.units.length;
            for (const unit of army.units) {
              hpSum += unit.currentHealth / unit.health;
              hpCount++;
            }
          }
        }
        const unitCost = game.unitStatsMap[UNIT_TYPES[t]].cost;
        buf[offset++] = unitCount / (100 / unitCost);
        buf[offset++] = hpCount > 0 ? hpSum / hpCount : 0;
      }
    }

    // Own units per (remaining moves, can attack) group, per type
    for (let t = 0; t < NUM_UNIT_TYPES; t++) {
      const army = self.getArmy(nodeName, UNIT_TYPES[t]);
      const unitCost = game.unitStatsMap[UNIT_TYPES[t]].cost;
      const counts = army ? groupCounts(army.units) : new Array<number>(NUM_UNIT_GROUPS).fill(0);
      for (let g = 0; g < NUM_UNIT_GROUPS; g++) buf[offset++] = counts[g] / (100 / unitCost);
    }

    const rowBase = ni * NUM_NODES;
    for (let j = 0; j < NUM_NODES; j++) {
      buf[offset++] = distMatrix[rowBase + j];
    }
  }

  // ─── 5. Context (292) ───

  // Decision type one-hot (8)
  if (context) {
    buf[offset + DECISION_TYPES.indexOf(context.type)] = 1;
  }
  offset += NUM_DECISION_TYPES;

  // ── recruit (20): location[16] + type[3] + affordable[1] ──
  if (context?.type === "recruit") {
    if (context.locationIdx >= 0 && context.locationIdx < NUM_NODES) {
      buf[offset + context.locationIdx] = 1;
    }
    offset += 16;
    const typeIdx = UNIT_TYPES.indexOf(context.unitType);
    if (typeIdx >= 0) buf[offset + typeIdx] = 1;
    offset += 3;
    const recruitCost = game.unitStatsMap[context.unitType].cost;
    buf[offset++] = context.affordableCount / (200 / recruitCost);
  } else {
    offset += RECRUIT_CONTEXT;
  }

  // ── army (36): armyInfo(27) + actionTypeMask[3] + disbandMask[6] ──
  if (context?.type === "army") {
    offset = encodeArmyInfo(buf, offset, game, context.army);
    for (let i = 0; i < NUM_ACTION_TYPES; i++) buf[offset++] = context.actionTypeMask[i];
    for (let i = 0; i < NUM_DISBAND_GROUPS; i++) buf[offset++] = context.disbandMask[i];
  } else {
    offset += ARMY_CONTEXT;
  }

  // ── moveTarget (43): armyInfo(27) + legalDestinationMask[16] ──
  if (context?.type === "moveTarget") {
    offset = encodeArmyInfo(buf, offset, game, context.army);
    for (let i = 0; i < NUM_NODES; i++) {
      buf[offset + i] = context.legalMask[i] > 0 ? 1 : 0;
    }
    offset += 16;
  } else {
    offset += MOVE_TARGET_CONTEXT;
  }

  // ── moveCount (47): armyInfo(27) + destination[16] + moveGroupMask[4] ──
  if (context?.type === "moveCount") {
    offset = encodeArmyInfo(buf, offset, game, context.army);
    if (context.destinationIdx >= 0 && context.destinationIdx < NUM_NODES) {
      buf[offset + context.destinationIdx] = 1;
    }
    offset += 16;
    for (let i = 0; i < NUM_MOVE_GROUPS; i++) buf[offset++] = context.groupMask[i];
  } else {
    offset += MOVE_COUNT_CONTEXT;
  }

  // ── battleTarget (16): attackableNodeMask[16] ──
  if (context?.type === "battleTarget") {
    for (let i = 0; i < NUM_NODES; i++) {
      buf[offset + i] = context.attackableMask[i] > 0 ? 1 : 0;
    }
    offset += 16;
  } else {
    offset += BATTLE_TARGET_CONTEXT;
  }

  // ── battleSelect (59): armyInfo(27) + targetNode[16] + selectionState(12) + isDone(1) + commitMask[3] ──
  if (context?.type === "battleSelect") {
    const army = context.army;
    if (army) {
      offset = encodeArmyInfo(buf, offset, game, army);
    } else {
      offset += ARMY_INFO_FEATURES; // "done" option: army fields all zero
    }
    if (context.targetNodeIdx >= 0 && context.targetNodeIdx < NUM_NODES) {
      buf[offset + context.targetNodeIdx] = 1;
    }
    offset += 16;
    for (let t = 0; t < NUM_UNIT_TYPES; t++) {
      const typeCost = game.unitStatsMap[UNIT_TYPES[t]].cost;
      buf[offset++] = context.selectedPerType[t * 2] / (100 / typeCost);
      buf[offset++] = context.selectedPerType[t * 2 + 1];
    }
    for (let t = 0; t < NUM_UNIT_TYPES; t++) {
      const typeCost = game.unitStatsMap[UNIT_TYPES[t]].cost;
      buf[offset++] = context.remainingPerType[t * 2] / (100 / typeCost);
      buf[offset++] = context.remainingPerType[t * 2 + 1];
    }
    buf[offset++] = context.isDone ? 1 : 0;
    for (let i = 0; i < NUM_COMMIT_GROUPS; i++) buf[offset++] = context.commitMask[i];
  } else {
    offset += BATTLE_SELECT_CONTEXT;
  }

  // ── battleAllocate (46): myArmy(22) + enemyArmy(21) + battleState(2) + unitsNeeded(1) ──
  // Units and health are those of the battle contingents, not the whole armies
  if (context?.type === "battleAllocate") {
    const army = context.army;
    offset = encodeArmyLocation(buf, offset, army);
    offset = encodeArmyType(buf, offset, army);
    const armyCost = game.unitStatsMap[army.unitType].cost;
    const battleUnits = army.battleUnits;
    buf[offset++] = battleUnits.length / (100 / armyCost);
    buf[offset++] = context.remaining / (100 / armyCost);
    buf[offset++] = encodeUnitsHp(battleUnits);
    const enemy = context.enemyArmy;
    offset = encodeArmyLocation(buf, offset, enemy);
    offset = encodeArmyType(buf, offset, enemy);
    const enemyCost = game.unitStatsMap[enemy.unitType].cost;
    const enemyUnits = enemy.battleUnits;
    buf[offset++] = enemyUnits.length / (100 / enemyCost);
    buf[offset++] = encodeUnitsHp(enemyUnits);
    buf[offset++] = context.attackProgress;
    buf[offset++] = context.isAttacker ? 1 : 0;
    buf[offset++] = context.unitsNeeded / 500;
  } else {
    offset += BATTLE_ALLOCATE_CONTEXT;
  }

  // ── battleRetreat (17): targetNode[16] + attackProgress[1] ──
  if (context?.type === "battleRetreat") {
    if (context.targetNodeIdx >= 0 && context.targetNodeIdx < NUM_NODES) {
      buf[offset + context.targetNodeIdx] = 1;
    }
    offset += 16;
    buf[offset++] = context.attackProgress;
  } else {
    offset += BATTLE_RETREAT_CONTEXT;
  }

  // Center all features: [0,1] → [-0.5, 0.5]
  for (let i = 0; i < STATE_SIZE; i++) buf[i] -= 0.5;

  return buf;
}
