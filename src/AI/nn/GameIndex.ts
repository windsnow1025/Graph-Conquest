/**
 * Fixed index order of the map nodes and the unit types, shared by the state
 * encoder (feature positions), the categorical heads (class indices) and the
 * action space (target node lookup).
 */
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
