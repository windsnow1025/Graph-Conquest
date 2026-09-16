/**
 * State encoder v9.
 *
 * Encoding order:
 *   1. Game config (5)
 *   2. Unit type stats (18): 3 types × 6 stats
 *   3. Player stats (21): 3 players × 7
 *   4. Per-node (880): 16 nodes × 55
 *   5. Context (224):
 *      - decision type one-hot (7)
 *      - recruit (20)
 *      - army (28)
 *      - moveTarget (39): army info(23) + legal destination mask(16)
 *      - battleTarget (16): attackable node mask
 *      - battleSelect (51): army info(22) + target node(16) + selection state(12) + isDone(1)
 *      - battleAllocate (46)
 *      - battleRetreat (17)
 *
 * Total: 5 + 18 + 21 + 880 + 224 = 1148 features.
 */
import type GameSystem from "../../lib/GameSystem";
import type Army from "../../lib/Army";
import type Graph from "../../lib/Graph";
import type {DefaultUnitName} from "../../lib/data/DefaultUnitStatsMap.ts";

export const NODE_ORDER: string[] = [
  "Blue Home", "Blue to Center", "B to G", "B to R",
  "Red Home", "Red to Center", "R to B", "R to G",
  "Green Home", "Green to Center", "G to R", "G to B",
  "Gate RB", "Gate GB", "Gate RG", "Center",
];

export const NUM_NODES = 16;
export const UNIT_TYPES: DefaultUnitName[] = ["Infantry", "Archer", "Cavalry"];
export const NUM_UNIT_TYPES = 3;

// Feature dimensions
const GAME_CONFIG_FEATURES = 5;
const UNIT_STATS_FEATURES = 18;
const PLAYER_STATS_FEATURES = 21;
const PER_NODE_FEATURES = 55;
const RECRUIT_CONTEXT = 20;
const ARMY_CONTEXT = 28;          // features(23) + action_type_mask(5)
const MOVE_TARGET_CONTEXT = 39;   // army info(23) + legal destination mask(16)
const BATTLE_TARGET_CONTEXT = 16; // attackable node mask
const BATTLE_SELECT_CONTEXT = 51; // army info(22) + target node(16) + selection state(12) + isDone(1)
const BATTLE_ALLOCATE_CONTEXT = 46;
const BATTLE_RETREAT_CONTEXT = 17;
const CONTEXT_FEATURES = 7 + RECRUIT_CONTEXT + ARMY_CONTEXT + MOVE_TARGET_CONTEXT
  + BATTLE_TARGET_CONTEXT + BATTLE_SELECT_CONTEXT + BATTLE_ALLOCATE_CONTEXT
  + BATTLE_RETREAT_CONTEXT; // 224
const MAX_DISTANCE = 6; // diameter of the default map (home to opposite gate); unreachable encodes as 1.0 too

export const STATE_SIZE =
  GAME_CONFIG_FEATURES +
  UNIT_STATS_FEATURES +
  PLAYER_STATS_FEATURES +
  NUM_NODES * PER_NODE_FEATURES +
  CONTEXT_FEATURES; // 1148

// ─── Precomputed distance matrix cache ───
let cachedDistMatrix: Float32Array | null = null;
let cachedGraph: Graph | null = null;

function getDistanceMatrix(graph: Graph): Float32Array {
  if (cachedGraph === graph && cachedDistMatrix) return cachedDistMatrix;
  const mat = new Float32Array(NUM_NODES * NUM_NODES);
  for (let i = 0; i < NUM_NODES; i++) {
    for (let j = 0; j < NUM_NODES; j++) {
      const d = graph.getDistance(NODE_ORDER[i], NODE_ORDER[j]);
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
  actionTypeMask: Float32Array; // [5]
}

export interface MoveTargetContext {
  type: "moveTarget";
  army: Army;
  legalMask: Float32Array; // [16] 1 = legal destination, choice is softmax over these
}

export interface BattleTargetContext {
  type: "battleTarget";
  attackableMask: Float32Array; // [16] 1 = attackable node, choice is softmax over these + stop
}

export interface BattleSelectContext {
  type: "battleSelect";
  army: Army | null;          // null for the "done" option
  targetNodeIdx: number;
  selectedPerType: number[];  // [6] = 3 × (units, avgHp)
  remainingPerType: number[]; // [6] = 3 × (units, avgHp)
  isDone: boolean;            // true = this option means "stop adding armies"
}

export interface BattleAllocateContext {
  type: "battleAllocate";
  army: Army;
  remaining: number;
  enemyArmy: Army;
  roundProgress: number;
  isAttacker: boolean;
  unitsNeeded: number;
}

export interface BattleRetreatContext {
  type: "battleRetreat";
  targetNodeIdx: number;
  roundProgress: number;
}

export type DecisionContext =
  | RecruitContext
  | ArmyContext
  | MoveTargetContext
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

function encodeArmyHp(army: Army): number {
  if (army.units.length === 0) return 0;
  let sum = 0;
  for (const u of army.units) sum += u.currentHealth / u.health;
  return sum / army.units.length;
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
  buf[offset++] = game.maxBattleRounds / 20;

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

    buf[offset++] = p.getUpkeep(game.upkeepRate) / 20;

    const totalUnits = p.armies.reduce((sum, a) => sum + a.units.length, 0);
    buf[offset++] = totalUnits / 200;

    let nodeCount = 0;
    for (const [, nodeOwner] of game.nodeOwnership) {
      if (nodeOwner === p) nodeCount++;
    }
    buf[offset++] = nodeCount / 16;

    buf[offset++] = p.defeated ? 1 : 0;
  }

  // ─── 4. Per-node features (16 × 55 = 880) ───
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

    for (let t = 0; t < NUM_UNIT_TYPES; t++) {
      let armyCount = 0;
      for (const army of self.armies) {
        if (army.location === nodeName && army.unitType === UNIT_TYPES[t]) {
          armyCount++;
        }
      }
      buf[offset++] = armyCount / 4;
    }

    for (let t = 0; t < NUM_UNIT_TYPES; t++) {
      let maxMoves = 0;
      let canAttack = 0;
      for (const army of self.armies) {
        if (army.location === nodeName && army.unitType === UNIT_TYPES[t]) {
          if (army.remainingMoves > maxMoves) maxMoves = army.remainingMoves;
          if (army.canAttack) canAttack = 1;
        }
      }
      buf[offset++] = maxMoves / 2;
      buf[offset++] = canAttack;
    }

    const rowBase = ni * NUM_NODES;
    for (let j = 0; j < NUM_NODES; j++) {
      buf[offset++] = distMatrix[rowBase + j];
    }
  }

  // ─── 5. Context (224) ───

  // Decision type one-hot (7): recruit | army | moveTarget | battleTarget | battleSelect | battleAllocate | battleRetreat
  const decisionTypes = ["recruit", "army", "moveTarget", "battleTarget", "battleSelect", "battleAllocate", "battleRetreat"] as const;
  if (context) {
    const idx = decisionTypes.indexOf(context.type);
    if (idx >= 0) buf[offset + idx] = 1;
  }
  offset += 7;

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

  // ── army (28): features(23) + actionTypeMask[5] ──
  if (context?.type === "army") {
    const army = context.army;
    offset = encodeArmyLocation(buf, offset, army);
    offset = encodeArmyType(buf, offset, army);
    const armyCost = game.unitStatsMap[army.unitType].cost;
    buf[offset++] = army.units.length / (100 / armyCost);
    buf[offset++] = encodeArmyHp(army);
    buf[offset++] = army.remainingMoves / 2;
    buf[offset++] = army.canAttack ? 1 : 0;
    for (let i = 0; i < 5; i++) buf[offset++] = context.actionTypeMask[i];
  } else {
    offset += ARMY_CONTEXT;
  }

  // ── moveTarget (39): armyInfo(23) + legalDestinationMask[16] ──
  if (context?.type === "moveTarget") {
    const army = context.army;
    offset = encodeArmyLocation(buf, offset, army);
    offset = encodeArmyType(buf, offset, army);
    const armyCost = game.unitStatsMap[army.unitType].cost;
    buf[offset++] = army.units.length / (100 / armyCost);
    buf[offset++] = encodeArmyHp(army);
    buf[offset++] = army.remainingMoves / 2;
    buf[offset++] = army.canAttack ? 1 : 0;
    for (let i = 0; i < NUM_NODES; i++) {
      buf[offset + i] = context.legalMask[i] > 0 ? 1 : 0;
    }
    offset += 16;
  } else {
    offset += MOVE_TARGET_CONTEXT;
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

  // ── battleSelect (51): armyInfo(22) + targetNode[16] + selectionState(12) + isDone(1) ──
  if (context?.type === "battleSelect") {
    const army = context.army;
    if (army) {
      offset = encodeArmyLocation(buf, offset, army);
      offset = encodeArmyType(buf, offset, army);
      const armyCost = game.unitStatsMap[army.unitType].cost;
      buf[offset++] = army.units.length / (100 / armyCost);
      buf[offset++] = encodeArmyHp(army);
      buf[offset++] = army.remainingMoves / 2;
    } else {
      offset += 22; // "done" option: army fields all zero
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
  } else {
    offset += BATTLE_SELECT_CONTEXT;
  }

  // ── battleAllocate (46): myArmy(22) + enemyArmy(21) + battleState(2) + unitsNeeded(1) ──
  if (context?.type === "battleAllocate") {
    const army = context.army;
    offset = encodeArmyLocation(buf, offset, army);
    offset = encodeArmyType(buf, offset, army);
    const armyCost = game.unitStatsMap[army.unitType].cost;
    buf[offset++] = army.units.length / (100 / armyCost);
    buf[offset++] = context.remaining / (100 / armyCost);
    buf[offset++] = encodeArmyHp(army);
    const enemy = context.enemyArmy;
    offset = encodeArmyLocation(buf, offset, enemy);
    offset = encodeArmyType(buf, offset, enemy);
    const enemyCost = game.unitStatsMap[enemy.unitType].cost;
    buf[offset++] = enemy.units.length / (100 / enemyCost);
    buf[offset++] = encodeArmyHp(enemy);
    buf[offset++] = context.roundProgress;
    buf[offset++] = context.isAttacker ? 1 : 0;
    buf[offset++] = context.unitsNeeded / 500;
  } else {
    offset += BATTLE_ALLOCATE_CONTEXT;
  }

  // ── battleRetreat (17): targetNode[16] + roundProgress[1] ──
  if (context?.type === "battleRetreat") {
    if (context.targetNodeIdx >= 0 && context.targetNodeIdx < NUM_NODES) {
      buf[offset + context.targetNodeIdx] = 1;
    }
    offset += 16;
    buf[offset++] = context.roundProgress;
  } else {
    offset += BATTLE_RETREAT_CONTEXT;
  }

  // Center all features: [0,1] → [-0.5, 0.5]
  for (let i = 0; i < STATE_SIZE; i++) buf[i] -= 0.5;

  return buf;
}
