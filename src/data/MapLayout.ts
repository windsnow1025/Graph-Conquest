export type NodeType = "home" | "center" | "gate" | "path";

export interface NodeLayout {
  x: number;
  y: number;
  type: NodeType;
}

const mapLayout: Record<string, NodeLayout> = {
  // Vertices
  "Blue Home": {x: 400, y: 80, type: "home"},
  "Green Home": {x: 90, y: 590, type: "home"},
  "Red Home": {x: 710, y: 590, type: "home"},

  // Center
  "Center": {x: 400, y: 410, type: "center"},

  // Blue → Center → (midpoint)
  "Blue to Center": {x: 400, y: 230, type: "path"},

  // Green → Center → (midpoint)
  "Green to Center": {x: 245, y: 500, type: "path"},

  // Red → Center → (midpoint)
  "Red to Center": {x: 555, y: 500, type: "path"},

  // Blue–Red edge (25%, 50%, 75% from Blue to Red)
  "B to R": {x: 478, y: 185, type: "path"},
  "Gate RB": {x: 555, y: 320, type: "gate"},
  "R to B": {x: 633, y: 455, type: "path"},

  // Blue–Green edge (25%, 50%, 75% from Blue to Green)
  "B to G": {x: 322, y: 185, type: "path"},
  "Gate GB": {x: 245, y: 320, type: "gate"},
  "G to B": {x: 168, y: 455, type: "path"},

  // Green–Red edge (25%, 50%, 75% from Green to Red)
  "G to R": {x: 245, y: 590, type: "path"},
  "Gate RG": {x: 400, y: 590, type: "gate"},
  "R to G": {x: 555, y: 590, type: "path"},
};

export default mapLayout;
