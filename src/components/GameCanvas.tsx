import {useEffect, useRef, useState} from "react";
import {Application, Graphics, Text, Container, TextStyle, FederatedPointerEvent, Ticker, Rectangle} from "pixi.js";
import GameSystem from "../lib/GameSystem";
import Player from "../lib/Player";
import Army from "../lib/Army";
import mapLayout, {type NodeType} from "../data/MapLayout";
import {playerColors} from "../data/PlayerColors";

interface BattleAllocationDisplay {
  attacker: Army;
  target: Army;
  unitCount: number;
}

interface GameCanvasProps {
  game: GameSystem;
  version: number;
  selectedArmy: Army | null;
  selectedNode: string | null;
  highlightPulse: Set<Army>;
  highlightSolid: Set<Army>;
  targetPulse: Set<Army>;
  attackableLocations: Set<string>;
  battleAllocationDisplay: BattleAllocationDisplay[];
  battleRemaining: number;
  onNodeClick: (location: string) => void;
  onArmyClick: (army: Army, player: Player) => void;
  onClearSelection: () => void;
}

const Army_Radius = 14;
export const Base_Width = 800;
export const Base_Height = 720;

function GameCanvas({game, version, selectedArmy, selectedNode, highlightPulse, highlightSolid, targetPulse, attackableLocations, battleAllocationDisplay, battleRemaining, onNodeClick, onArmyClick, onClearSelection}: GameCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const appRef = useRef<Application | null>(null);
  const [ready, setReady] = useState(false);
  const [resizeCount, setResizeCount] = useState(0);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let cancelled = false;
    const app = new Application();

    app.init({
      width: Base_Width,
      height: Base_Height,
      background: 0x1a1a2e,
      antialias: true,
      resolution: window.devicePixelRatio || 1,
      autoDensity: true,
    }).then(() => {
      if (cancelled) {
        app.destroy(true);
        return;
      }
      appRef.current = app;
      container.appendChild(app.canvas);
      setReady(true);
    }).catch((err) => {
      console.error("Failed to initialize Pixi.js:", err);
    });

    const onResize = () => {
      if (container) {
        setResizeCount(c => c + 1);
      }
    };
    window.addEventListener("resize", onResize);

    return () => {
      cancelled = true;
      setReady(false);
      window.removeEventListener("resize", onResize);
      if (appRef.current) {
        appRef.current.destroy(true);
        appRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!ready) return;
    const app = appRef.current;
    if (!app || !app.stage) return;
    const container = containerRef.current;
    if (!container) return;

    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    const scale = Math.min(containerWidth / Base_Width, containerHeight / Base_Height);

    app.renderer.resize(containerWidth, containerHeight);
    for (const child of app.stage.removeChildren()) {
      child.destroy(true); // rebuilt from scratch each pass; free Text textures and Graphics contexts
    }
    app.stage.scale.set(scale);
    app.stage.position.set(
      (containerWidth - Base_Width * scale) / 2,
      (containerHeight - Base_Height * scale) / 2,
    );

    // Background hit area for clearing selection on empty-space click
    const bgHit = new Container();
    bgHit.eventMode = "static";
    bgHit.hitArea = new Rectangle(0, 0, Base_Width, Base_Height);
    bgHit.on("pointertap", () => onClearSelection());
    app.stage.addChild(bgHit);

    drawEdges(app);
    const nodePulse = drawNodes(app, game, selectedArmy, selectedNode, attackableLocations, onNodeClick);
    const { pulseGraphics: armyPulse, armyPositions } = drawArmies(app, game, selectedArmy, highlightPulse, highlightSolid, targetPulse, onArmyClick);
    if (battleAllocationDisplay.length > 0 || battleRemaining > 0) {
      drawBattleArrows(app, armyPositions, battleAllocationDisplay, battleRemaining, selectedArmy);
    }
    const pulseGraphics = [...nodePulse, ...armyPulse];

    // Pulse animation
    let elapsed = 0;
    const tickerFn = (ticker: Ticker) => {
      elapsed += ticker.deltaMS;
      const alpha = 0.3 + 0.7 * Math.abs(Math.sin(elapsed * 0.004));
      for (const g of pulseGraphics) {
        g.alpha = alpha;
      }
    };
    if (pulseGraphics.length > 0) {
      app.ticker.add(tickerFn);
    }

    return () => {
      if (pulseGraphics.length > 0 && app.ticker) {
        app.ticker.remove(tickerFn);
      }
    };
  }, [ready, version, resizeCount, selectedArmy, selectedNode, highlightPulse, highlightSolid, targetPulse, attackableLocations, battleAllocationDisplay, battleRemaining, game, onNodeClick, onArmyClick, onClearSelection]);

  return <div ref={containerRef} style={{width: "100%", height: "100%", overflow: "hidden"}}/>;
}

function drawEdges(app: Application) {
  const edges = new Graphics();
  const connections: [string, string][] = [
    ["Blue Home", "Blue to Center"],
    ["Blue Home", "B to R"],
    ["Blue Home", "B to G"],
    ["Blue to Center", "Center"],
    ["B to R", "Gate RB"],
    ["B to G", "Gate GB"],
    ["Red Home", "Red to Center"],
    ["Red Home", "R to B"],
    ["Red Home", "R to G"],
    ["Red to Center", "Center"],
    ["R to B", "Gate RB"],
    ["R to G", "Gate RG"],
    ["Green Home", "Green to Center"],
    ["Green Home", "G to B"],
    ["Green Home", "G to R"],
    ["Green to Center", "Center"],
    ["G to B", "Gate GB"],
    ["G to R", "Gate RG"],
  ];

  for (const [from, to] of connections) {
    const fromPos = mapLayout[from];
    const toPos = mapLayout[to];
    if (!fromPos || !toPos) continue;

    edges.moveTo(fromPos.x, fromPos.y);
    edges.lineTo(toPos.x, toPos.y);
  }
  edges.stroke({width: 2, color: 0x444466});

  app.stage.addChild(edges);
}

const nodeStyles: Record<NodeType, {
  radius: number;
  defaultFill: number;
  defaultStroke: number;
  strokeWidth: number;
}> = {
  home: {radius: 24, defaultFill: 0x334455, defaultStroke: 0x667788, strokeWidth: 3},
  center: {radius: 22, defaultFill: 0x443322, defaultStroke: 0xFFB300, strokeWidth: 3},
  gate: {radius: 18, defaultFill: 0x2a2a44, defaultStroke: 0x8888aa, strokeWidth: 2},
  path: {radius: 16, defaultFill: 0x334455, defaultStroke: 0x667788, strokeWidth: 1.5},
};

function drawNodeShape(g: Graphics, type: NodeType, radius: number) {
  switch (type) {
    case "home":
      for (let i = 0; i < 5; i++) {
        const angle = (i * 2 * Math.PI) / 5 - Math.PI / 2;
        const x = Math.cos(angle) * radius;
        const y = Math.sin(angle) * radius;
        if (i === 0) g.moveTo(x, y);
        else g.lineTo(x, y);
      }
      g.closePath();
      break;
    case "center":
      for (let i = 0; i < 6; i++) {
        const angle = (i * 2 * Math.PI) / 6 - Math.PI / 6;
        const x = Math.cos(angle) * radius;
        const y = Math.sin(angle) * radius;
        if (i === 0) g.moveTo(x, y);
        else g.lineTo(x, y);
      }
      g.closePath();
      break;
    case "gate":
      g.moveTo(0, -radius);
      g.lineTo(radius, 0);
      g.lineTo(0, radius);
      g.lineTo(-radius, 0);
      g.closePath();
      break;
    case "path":
      g.circle(0, 0, radius);
      break;
  }
}

function drawNodes(
  app: Application,
  game: GameSystem,
  selectedArmy: Army | null,
  selectedNode: string | null,
  attackableLocations: Set<string>,
  onNodeClick: (location: string) => void,
): Graphics[] {
  const pulseGraphics: Graphics[] = [];
  const movableLocations = selectedArmy && !game.currentBattle
    ? selectedArmy.getMovableLocations(game.gameMap, game.enemyLocations)
    : [];

  for (const [name, node] of Object.entries(mapLayout)) {
    const style = nodeStyles[node.type];
    const nodeContainer = new Container();
    nodeContainer.position.set(node.x, node.y);
    nodeContainer.eventMode = "static";
    nodeContainer.cursor = "pointer";
    nodeContainer.on("pointertap", () => {
      onNodeClick(name);
    });

    const shape = new Graphics();
    let fillColor = style.defaultFill;
    let strokeColor = style.defaultStroke;

    const owner = game.nodeOwnership.get(name);
    if (owner) {
      const ownerColor = playerColors[owner.name];
      if (ownerColor !== undefined) {
        const r = ((ownerColor >> 16) & 0xFF) * 0.35;
        const g = ((ownerColor >> 8) & 0xFF) * 0.35;
        const b = (ownerColor & 0xFF) * 0.35;
        fillColor = (Math.round(r) << 16) | (Math.round(g) << 8) | Math.round(b);
        strokeColor = ownerColor;
      }
    }

    drawNodeShape(shape, node.type, style.radius);
    shape.fill(fillColor);
    shape.stroke({width: style.strokeWidth, color: strokeColor});
    nodeContainer.addChild(shape);

    if (movableLocations.includes(name)) {
      const ring = new Graphics();
      drawNodeShape(ring, node.type, style.radius + 5);
      ring.stroke({width: 3, color: 0x4CAF50});
      nodeContainer.addChild(ring);
      pulseGraphics.push(ring);
    } else if (attackableLocations.has(name)) {
      const ring = new Graphics();
      drawNodeShape(ring, node.type, style.radius + 5);
      ring.stroke({width: 2.5, color: 0xFF5252});
      nodeContainer.addChild(ring);
      pulseGraphics.push(ring);
    } else if (selectedNode === name) {
      const ring = new Graphics();
      drawNodeShape(ring, node.type, style.radius + 5);
      ring.stroke({width: 2.5, color: 0x42A5F5});
      nodeContainer.addChild(ring);
    }

    const label = new Text({
      text: name,
      style: new TextStyle({
        fontSize: 10,
        fill: 0xcccccc,
        align: "center",
      }),
    });
    label.anchor.set(0.5);
    label.position.set(0, style.radius + 12);
    nodeContainer.addChild(label);

    app.stage.addChild(nodeContainer);
  }
  return pulseGraphics;
}

function drawArmies(
  app: Application,
  game: GameSystem,
  selectedArmy: Army | null,
  highlightPulse: Set<Army>,
  highlightSolid: Set<Army>,
  targetPulse: Set<Army>,
  onArmyClick: (army: Army, player: Player) => void,
): { pulseGraphics: Graphics[]; armyPositions: Map<Army, {x: number, y: number}> } {
  const pulseGraphics: Graphics[] = [];
  const armyPositions = new Map<Army, {x: number, y: number}>();

  const armiesByLocation = new Map<string, Array<{ army: Army; player: Player }>>();
  for (const player of game.allPlayers) {
    for (const army of player.armies) {
      if (army.units.length === 0) continue;
      if (!armiesByLocation.has(army.location)) {
        armiesByLocation.set(army.location, []);
      }
      armiesByLocation.get(army.location)!.push({army, player});
    }
  }

  const isCurrentPlayer = (player: Player) => player === game.currentPlayer;

  for (const [location, entries] of armiesByLocation) {
    const node = mapLayout[location];
    if (!node) continue;
    const nodeRadius = nodeStyles[node.type].radius;

    entries.forEach((entry, index) => {
      const {army, player} = entry;
      const offsetX = (index - (entries.length - 1) / 2) * (Army_Radius * 2.5);

      const armyX = node.x + offsetX;
      const armyY = node.y - nodeRadius - Army_Radius - 4;
      armyPositions.set(army, {x: armyX, y: armyY});

      const armyContainer = new Container();
      armyContainer.position.set(armyX, armyY);
      armyContainer.eventMode = "static";
      armyContainer.cursor = "pointer";
      armyContainer.on("pointertap", (e: FederatedPointerEvent) => {
        e.stopPropagation();
        onArmyClick(army, player);
      });

      const color = playerColors[player.name] ?? 0x888888;
      const canMove = army.remainingMoves > 0;
      const canAtk = army.canAttack;
      const isCurrent = isCurrentPlayer(player);
      const exhausted = isCurrent && !canMove && !canAtk && !game.currentBattle;

      // Green solid ring — selected for battle
      if (highlightSolid.has(army)) {
        const ring = new Graphics();
        ring.circle(0, 0, Army_Radius + 2);
        ring.stroke({width: 3, color: game.currentBattle ? 0xFF5252 : 0x4CAF50});
        armyContainer.addChild(ring);
      }
      // Green pulsing ring — available/unacted
      else if (highlightPulse.has(army)) {
        const ring = new Graphics();
        ring.circle(0, 0, Army_Radius + 2);
        ring.stroke({width: 2, color: 0x4CAF50});
        armyContainer.addChild(ring);
        pulseGraphics.push(ring);
      }

      // Red pulsing ring — targetable enemy
      if (targetPulse.has(army)) {
        const ring = new Graphics();
        ring.circle(0, 0, Army_Radius + 2);
        ring.stroke({width: 2.5, color: 0xFF5252});
        armyContainer.addChild(ring);
        pulseGraphics.push(ring);
      }

      // Selection highlight ring
      if (army === selectedArmy) {
        const ring = new Graphics();
        ring.circle(0, 0, Army_Radius + 3);
        ring.stroke({width: 2, color: 0xFFD740});
        armyContainer.addChild(ring);
      }

      // Main circle
      const bg = new Graphics();
      bg.circle(0, 0, Army_Radius);
      bg.fill(color);
      bg.stroke({width: 1, color: exhausted ? 0x666666 : 0xffffff});
      armyContainer.addChild(bg);

      if (exhausted) {
        armyContainer.alpha = 0.45;
      }

      // Status indicators (only outside battle)
      if (isCurrent && !exhausted && !game.currentBattle) {
        const indicators = new Graphics();
        if (canMove) {
          const hasMoveDest = army.getMovableLocations(game.gameMap, game.enemyLocations).length > 0;
          indicators.circle(-Army_Radius + 3, Army_Radius - 3, 3);
          indicators.fill(hasMoveDest ? 0x4CAF50 : 0x2E7D32);
          indicators.alpha = hasMoveDest ? 1 : 0.4;
        }
        if (canAtk) {
          const hasTarget = game.hasAttackTargets(army);
          indicators.circle(Army_Radius - 3, Army_Radius - 3, 3);
          indicators.fill(hasTarget ? 0xFF5252 : 0xB71C1C);
          indicators.alpha = hasTarget ? 1 : 0.4;
        }
        armyContainer.addChild(indicators);
      }

      // Unit count
      const countLabel = new Text({
        text: `${army.units.length}`,
        style: new TextStyle({
          fontSize: 11,
          fill: 0xffffff,
          fontWeight: "bold",
          align: "center",
        }),
      });
      countLabel.anchor.set(0.5);
      armyContainer.addChild(countLabel);

      // Unit type initial
      const typeLabel = new Text({
        text: army.unitType.charAt(0),
        style: new TextStyle({
          fontSize: 8,
          fill: 0xffffff,
          align: "center",
        }),
      });
      typeLabel.anchor.set(0.5);
      typeLabel.position.set(0, Army_Radius + 6);
      armyContainer.addChild(typeLabel);

      app.stage.addChild(armyContainer);
    });
  }
  return { pulseGraphics, armyPositions };
}

function drawBattleArrows(
  app: Application,
  armyPositions: Map<Army, {x: number, y: number}>,
  allocations: BattleAllocationDisplay[],
  remaining: number,
  attackerArmy: Army | null,
) {
  for (const alloc of allocations) {
    const from = armyPositions.get(alloc.attacker);
    const to = armyPositions.get(alloc.target);
    if (!from || !to) continue;

    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist === 0) continue;

    // Shorten arrow to not overlap army circles
    const nx = dx / dist;
    const ny = dy / dist;
    const startX = from.x + nx * (Army_Radius + 2);
    const startY = from.y + ny * (Army_Radius + 2);
    const endX = to.x - nx * (Army_Radius + 6);
    const endY = to.y - ny * (Army_Radius + 6);

    // Arrow line + arrowhead
    const arrow = new Graphics();
    arrow.moveTo(startX, startY);
    arrow.lineTo(endX, endY);
    const headSize = 8;
    const angle = Math.atan2(endY - startY, endX - startX);
    arrow.moveTo(endX, endY);
    arrow.lineTo(
      endX - headSize * Math.cos(angle - 0.4),
      endY - headSize * Math.sin(angle - 0.4),
    );
    arrow.moveTo(endX, endY);
    arrow.lineTo(
      endX - headSize * Math.cos(angle + 0.4),
      endY - headSize * Math.sin(angle + 0.4),
    );
    arrow.stroke({width: 2, color: 0xFF5252});
    arrow.alpha = 0.6;
    app.stage.addChild(arrow);

    // Unit count label at midpoint
    const midX = startX + (endX - startX) * 0.85;
    const midY = startY + (endY - startY) * 0.85;
    const label = new Text({
      text: `${alloc.unitCount}`,
      style: new TextStyle({
        fontSize: 11,
        fill: 0xFFD740,
        fontWeight: "bold",
      }),
    });
    label.anchor.set(0.5);
    label.alpha = 0.8;
    label.position.set(midX, midY - 8);
    app.stage.addChild(label);
  }

  // Show remaining count near attacker army
  if (attackerArmy && remaining > 0) {
    const pos = armyPositions.get(attackerArmy);
    if (pos) {
      const label = new Text({
        text: `${remaining} left`,
        style: new TextStyle({
          fontSize: 9,
          fill: 0xFFD740,
        }),
      });
      label.anchor.set(0.5);
      label.position.set(pos.x, pos.y - Army_Radius - 10);
      app.stage.addChild(label);
    }
  }
}

export default GameCanvas;
