import {useState, useRef, useCallback, useEffect, useMemo} from "react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Slider from "@mui/material/Slider";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import TextField from "@mui/material/TextField";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import Chip from "@mui/material/Chip";
import GameAPI from "./lib/GameAPI";
import type {ArmyId, ArmySelection, AttackAllocation, UnitSelection, GameData, GameSave, ArmyData} from "./lib/GameAPI";
import DefaultConfig from "./lib/data/DefaultConfig";
import {BattlePhase} from "./lib/GameAPI";
import type {DefaultUnitName} from "./lib/data/DefaultUnitStatsMap.ts";
import DefaultUnitStatsMap from "./lib/data/DefaultUnitStatsMap.ts";
import {armyIdOf, keyOfId, attackReady, battleUnits} from "./AI/nn/Armies";
import {sameState, selectionsOf} from "./AI/nn/UnitGroups";
import GameCanvas, {Base_Width, Base_Height} from "./components/GameCanvas";
import ModeSelect from "./components/ModeSelect";
import UnitPicker from "./components/UnitPicker";
import type {GameMode} from "./components/ModeSelect";
import mapLayout from "./data/MapLayout";
import {playerCssColors} from "./data/PlayerColors";
import {aiTurnSteps, greedyTurnStepsUI, aiDefenderPhase} from "./AI";

const VS_AI_PLAYERS = new Set(["Red", "Green"]);
const ALL_PLAYERS = new Set(["Blue", "Red", "Green"]);
const NO_PLAYERS = new Set<string>();

const overlayPaper = {opacity: 0.9, pointerEvents: "auto" as const};

/** The army a state holds under an identifier, or null when its units have all left. */
function findArmy(state: GameData, armyId: ArmyId): ArmyData | null {
  const player = [...state.players, state.neutralPlayer].find(p => p.name === armyId.owner);
  return player?.armies.find(army => army.location === armyId.location && army.unitType === armyId.unitType) ?? null;
}

function countOf(selections: UnitSelection[]): number {
  return selections.reduce((sum, selection) => sum + selection.count, 0);
}

function App() {
  const apiRef = useRef(GameAPI.create(DefaultConfig));
  const [version, setVersion] = useState(0);
  const [selectedArmyId, setSelectedArmyId] = useState<ArmyId | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [gameMode, setGameMode] = useState<GameMode>("menu");
  const [gameEpoch, setGameEpoch] = useState(0); // bumped when apiRef is replaced, so game-bound effects restart
  const [aiError, setAiError] = useState<string | null>(null); // an AI turn or defender phase that failed (the game cannot continue)
  const canvasContainerRef = useRef<HTMLDivElement>(null);

  // Recruit UI
  const [unitType, setUnitType] = useState<DefaultUnitName>("Infantry");
  const [recruitCount, setRecruitCount] = useState(1);

  // Army management UI
  const [selectedUnits, setSelectedUnits] = useState<UnitSelection[]>([]);

  // Pre-battle targeting
  const [battleTarget, setBattleTarget] = useState<string | null>(null);
  const [battlePicked, setBattlePicked] = useState<Map<string, ArmySelection>>(new Map());

  // In-battle attack allocation
  const [battleAttackArmyId, setBattleAttackArmyId] = useState<ArmyId | null>(null);
  const [battleAllocations, setBattleAllocations] = useState<AttackAllocation[]>([]);
  const [battleAllocTargetId, setBattleAllocTargetId] = useState<ArmyId | null>(null);
  const [battleAllocCount, setBattleAllocCount] = useState(1);

  // Save/Load
  const [saveLoadMode, setSaveLoadMode] = useState<"save" | "load" | null>(null);
  const [saveName, setSaveName] = useState("");

  const api = apiRef.current;
  // One state read per game change, so that the objects stay the same between changes, as the effects keyed on them expect
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const state = useMemo(() => api.getState(), [api, version]);
  const battle = state.currentBattle;
  const currentPlayer = state.players[state.currentPlayerIndex];
  const gameOver = api.isGameOver();

  const update = useCallback(() => {
    setVersion(v => v + 1);
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedArmyId(null);
    setSelectedNode(null);
    setSelectedUnits([]);
    setBattleTarget(null);
    setBattlePicked(new Map());
  }, []);

  const clearBattleAttack = useCallback(() => {
    setBattleAttackArmyId(null);
    setBattleAllocations([]);
    setBattleAllocTargetId(null);
    setBattleAllocCount(1);
  }, []);

  const selectDefaultNode = useCallback(() => {
    const current = apiRef.current.getState();
    const player = current.players[current.currentPlayerIndex];
    if (current.nodeOwnership.get(player.homeLocation) === player) {
      setSelectedNode(player.homeLocation);
      return;
    }
    const recruitLocs = apiRef.current.getRecruitLocations();
    if (recruitLocs.length === 1) {
      setSelectedNode(recruitLocs[0]);
    }
  }, []);

  // --- AI turn ---
  const aiPlayers = gameMode === "autoplay" ? ALL_PLAYERS
    : gameMode === "ai" || gameMode === "greedy" ? VS_AI_PLAYERS
    : NO_PLAYERS;
  const isAITurn = (gameMode === "ai" || gameMode === "greedy" || gameMode === "autoplay") && !gameOver && aiPlayers.has(currentPlayer.name);

  useEffect(() => {
    if (!isAITurn) return;
    let cancelled = false;
    const runAllAITurns = async () => {
      const a = apiRef.current;
      const aiSet = gameMode === "autoplay" ? ALL_PLAYERS : VS_AI_PLAYERS;
      const currentName = () => {
        const current = a.getState();
        return current.players[current.currentPlayerIndex].name;
      };
      while (!cancelled && !a.isGameOver() && aiSet.has(currentName())) {
        const gen = gameMode === "greedy"
          ? greedyTurnStepsUI(a)
          : await aiTurnSteps(a);
        if (cancelled) return;
        if (!gen) { update(); continue; }
        // A step runs in a timer callback: a throw there (an AI decision, the game engine)
        // must reject this promise to reach the catch below instead of dying uncaught
        await new Promise<void>((resolve, reject) => {
          const step = () => {
            if (cancelled) { resolve(); return; }
            try {
              const result = gen.next();
              update();
              if (!result.done) setTimeout(step, 0); else resolve();
            } catch (e) {
              reject(e);
            }
          };
          setTimeout(step, 0);
        });
      }
      if (!cancelled) { clearSelection(); selectDefaultNode(); update(); }
    };
    runAllAITurns().catch((e: unknown) => { if (!cancelled) setAiError(e instanceof Error ? e.message : String(e)); });
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAITurn, gameMode, gameEpoch]);

  // --- Battle auto-actions (skip during AI turns — AI handles battles itself) ---
  useEffect(() => {
    if (!battle || isAITurn) return;

    // AI-controlled defender uses their model (NN or greedy)
    if (battle.phase === BattlePhase.DefenderTurn && aiPlayers.has(battle.defenderPlayer.name)) {
      aiDefenderPhase(apiRef.current, battle, gameMode).then(update, (e: unknown) => setAiError(e instanceof Error ? e.message : String(e)));
      return;
    }

    // Auto-allocate the defender's forced move when both sides have exactly 1 army.
    // The attacker is never auto-played: acting would foreclose the retreat decision.
    if (battle.phase === BattlePhase.DefenderTurn
      && battle.attackerArmies.length === 1 && battle.defenderArmies.length === 1) {
      const armyId = armyIdOf(battle.defenderPlayer.name, battle.defenderArmies[0]);
      const targetId = armyIdOf(battle.attackerPlayer.name, battle.attackerArmies[0]);
      if (apiRef.current.canAct(armyId)) {
        apiRef.current.allocateAttack(armyId, [{targetId, count: battleUnits(battle.defenderArmies[0]).length}]);
        clearBattleAttack();
        update();
        return;
      }
    }
  }, [battle, isAITurn, aiPlayers, version, update, clearBattleAttack]);

  // --- Reset battle UI state on phase/round change ---
  useEffect(() => {
    clearBattleAttack();
  }, [battle?.phase, battle?.round, clearBattleAttack]);

  // --- Handlers ---
  const handleModeSelect = useCallback((mode: GameMode) => {
    apiRef.current = GameAPI.create(DefaultConfig);
    setVersion(0);
    clearSelection();
    clearBattleAttack();
    setAiError(null);
    setGameMode(mode);
    selectDefaultNode();
  }, [clearSelection, clearBattleAttack, selectDefaultNode]);

  const SAVE_PREFIX = "graph-conquest-save-";

  const getSaveList = (): string[] =>
    Object.keys(localStorage)
      .filter(key => key.startsWith(SAVE_PREFIX))
      .map(key => key.slice(SAVE_PREFIX.length))
      .toSorted();

  const handleSave = (name: string) => {
    if (!name.trim()) return;
    const save = apiRef.current.toJSON();
    localStorage.setItem(SAVE_PREFIX + name.trim(), JSON.stringify(save));
    setSaveLoadMode(null);
    setSaveName("");
  };

  const handleLoad = (name: string) => {
    const raw = localStorage.getItem(SAVE_PREFIX + name);
    if (!raw) return;
    const save: GameSave = JSON.parse(raw);
    apiRef.current = GameAPI.fromJSON(save);
    setGameEpoch(e => e + 1);
    clearSelection();
    clearBattleAttack();
    setAiError(null);
    setSaveLoadMode(null);
    update();
  };

  const handleDeleteSave = (name: string) => {
    localStorage.removeItem(SAVE_PREFIX + name);
    update();
  };

  const handleEndTurn = useCallback(() => {
    apiRef.current.endTurn();
    clearSelection();
    selectDefaultNode();
    update();
  }, [update, clearSelection, selectDefaultNode]);

  // --- Window resize → re-render so node-anchored overlays track the rescaled map ---
  const [, setResizeCount] = useState(0);
  useEffect(() => {
    const onResize = () => setResizeCount(c => c + 1);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // --- Node screen position (for positioning overlays at map locations) ---
  const getNodeScreenPos = (location: string) => {
    const el = canvasContainerRef.current;
    const node = mapLayout[location];
    if (!el || !node) return null;
    const scale = Math.min(el.clientWidth / Base_Width, el.clientHeight / Base_Height);
    const offsetX = (el.clientWidth - Base_Width * scale) / 2;
    const offsetY = (el.clientHeight - Base_Height * scale) / 2;
    return {
      left: node.x * scale + offsetX,
      top: node.y * scale + offsetY,
      scale,
    };
  };

  // --- Computed highlights ---
  const armiesInRange = new Set(battleTarget ? api.getArmiesInRange(battleTarget).map(keyOfId) : []);

  const battleUnacted = new Set<string>();
  if (battle) {
    const attacking = battle.phase === BattlePhase.AttackerTurn;
    const owner = (attacking ? battle.attackerPlayer : battle.defenderPlayer).name;
    for (const army of attacking ? battle.attackerArmies : battle.defenderArmies) {
      if (!battle.actedArmies.has(army)) battleUnacted.add(keyOfId(armyIdOf(owner, army)));
    }
  }
  // The army the player is allocating attacks from, while it can still act
  const attackingArmyId = battle && battleAttackArmyId && api.canAct(battleAttackArmyId) ? battleAttackArmyId : null;
  const allocatedTargets = new Set(battleAllocations.map(allocation => keyOfId(allocation.targetId)));
  const battleTargets = attackingArmyId && !battleAllocTargetId
    ? new Set(api.getTargetsInRange(attackingArmyId).map(keyOfId).filter(key => !allocatedTargets.has(key)))
    : new Set<string>();

  const battleRemaining = attackingArmyId
    ? battleUnits(findArmy(state, attackingArmyId)!).length - battleAllocations.reduce((sum, allocation) => sum + allocation.count, 0)
    : 0;

  // --- Click handlers ---
  const attackableLocations = !battle && !battleTarget && !selectedArmyId && !isAITurn ? api.getAttackableLocations() : new Set<string>();

  const handleNodeClick = useCallback((location: string) => {
    if (battle || isAITurn) return;

    // Cancel targeting mode
    if (battleTarget) {
      setBattleTarget(null);
      setBattlePicked(new Map());
      return;
    }

    // Move selected army to clicked node
    if (selectedArmyId && selectedArmyId.owner === currentPlayer.name) {
      apiRef.current.moveUnits(selectedArmyId, selectedUnits, location);
      clearSelection();
      update();
      return;
    }

    // Click attackable enemy node → enter battle targeting
    if (!selectedArmyId && apiRef.current.getAttackableLocations().has(location)) {
      setSelectedArmyId(null);
      setSelectedNode(null);
      setBattleTarget(location);
      setBattlePicked(new Map());
      return;
    }

    // Select node (for recruit UI etc.)
    setSelectedArmyId(null);
    setSelectedNode(location);
  }, [battle, isAITurn, selectedArmyId, currentPlayer, selectedUnits, battleTarget, update, clearSelection]);

  const handleArmyClick = useCallback((armyId: ArmyId) => {
    if (isAITurn) return;
    const a = apiRef.current;
    const current = a.getState();
    const key = keyOfId(armyId);

    if (current.currentBattle) {
      if (a.canAct(armyId)) {
        setBattleAttackArmyId(armyId);
        setBattleAllocations([]);
        setBattleAllocTargetId(null);
        setBattleAllocCount(1);
        return;
      }
      if (attackingArmyId && battleRemaining > 0) {
        const targets = a.getTargetsInRange(attackingArmyId);
        if (targets.some(targetId => keyOfId(targetId) === key) && !battleAllocations.some(allocation => keyOfId(allocation.targetId) === key)) {
          const needed = a.getUnitsNeeded(attackingArmyId, armyId);
          setBattleAllocTargetId(armyId);
          setBattleAllocCount(Math.min(needed, battleRemaining));
          return;
        }
      }
      return;
    }

    const army = findArmy(current, armyId);
    if (!army) return;
    const own = armyId.owner === current.players[current.currentPlayerIndex].name;
    if (battleTarget && own) {
      if (!a.getArmiesInRange(battleTarget).some(inRange => keyOfId(inRange) === key)) return;
      setBattlePicked((prev) => {
        const next = new Map(prev);
        if (next.has(key)) next.delete(key);
        else next.set(key, {armyId, selections: selectionsOf(attackReady(army))});
        return next;
      });
      return;
    }

    // Click enemy army → treat as clicking the node (enter targeting if attackable)
    if (!own) {
      if (!battleTarget && !selectedArmyId && a.getAttackableLocations().has(armyId.location)) {
        setSelectedArmyId(null);
        setSelectedNode(null);
        setBattleTarget(armyId.location);
        setBattlePicked(new Map());
      }
      return;
    }

    // Click own army → select it
    setBattleTarget(null);
    setBattlePicked(new Map());
    setSelectedArmyId(armyId);
    setSelectedUnits(selectionsOf(army.units));
    setSelectedNode(null);
  }, [isAITurn, battleTarget, selectedArmyId, attackingArmyId, battleRemaining, battleAllocations]);

  // --- Army actions ---
  const handleDisband = useCallback(() => {
    if (!selectedArmyId) return;
    apiRef.current.disbandUnits(selectedArmyId, selectedUnits);
    const army = findArmy(apiRef.current.getState(), selectedArmyId);
    if (!army) clearSelection();
    else setSelectedUnits(selectionsOf(army.units));
    update();
  }, [selectedArmyId, selectedUnits, update, clearSelection]);

  // --- Battle actions ---
  const handleStartBattle = useCallback(() => {
    if (!battleTarget || battlePicked.size === 0) return;
    apiRef.current.startBattle([...battlePicked.values()], battleTarget);
    clearSelection();
    clearBattleAttack();
    update();
  }, [battleTarget, battlePicked, update, clearSelection, clearBattleAttack]);

  const handleBattleAllocate = useCallback(() => {
    if (!battleAllocTargetId) return;
    setBattleAllocations(prev => [...prev, {targetId: battleAllocTargetId, count: battleAllocCount}]);
    setBattleAllocTargetId(null);
    setBattleAllocCount(1);
  }, [battleAllocTargetId, battleAllocCount]);

  const handleBattleConfirm = useCallback(() => {
    if (!battleAttackArmyId || !battle) return;
    apiRef.current.allocateAttack(battleAttackArmyId, battleAllocations);
    clearBattleAttack();
    update();
  }, [battle, battleAttackArmyId, battleAllocations, update, clearBattleAttack]);


  const handleBattleRetreat = useCallback(() => {
    if (!battle) return;
    apiRef.current.retreat();
    clearBattleAttack();
    update();
  }, [battle, update, clearBattleAttack]);

  // One-click full allocation for the 1v1 case (replaces the former auto-allocation)
  const handleBattleAttackAll = useCallback(() => {
    if (!battle) return;
    if (battle.attackerArmies.length !== 1 || battle.defenderArmies.length !== 1) return;
    const armyId = armyIdOf(battle.attackerPlayer.name, battle.attackerArmies[0]);
    const targetId = armyIdOf(battle.defenderPlayer.name, battle.defenderArmies[0]);
    if (apiRef.current.canAct(armyId)) {
      apiRef.current.allocateAttack(armyId, [{targetId, count: battleUnits(battle.attackerArmies[0]).length}]);
    }
    clearBattleAttack();
    update();
  }, [battle, update, clearBattleAttack]);


  // --- Render ---
  if (gameMode === "menu") {
    return <ModeSelect onSelect={handleModeSelect}/>;
  }

  const canvasSelectedArmyId = battle ? battleAttackArmyId : selectedArmyId;

  const canvasHighlightPulse = battle
    ? (battleAttackArmyId ? new Set<string>() : battleUnacted)
    : battleTarget ? armiesInRange : new Set<string>();
  const canvasHighlightSolid = battle
    ? new Set([
        ...allocatedTargets,
        ...(battleAllocTargetId ? [keyOfId(battleAllocTargetId)] : []),
      ])
    : new Set(battlePicked.keys());

  // Recruit conditions
  const canRecruitAtSelected = !battle && !isAITurn && selectedNode !== null
    && api.getRecruitLocations().includes(selectedNode);
  const maxRecruit = Math.max(1, Math.floor(currentPlayer.money / state.unitStatsMap[unitType].cost) || 1);
  const recruitCountClamped = Math.min(recruitCount, maxRecruit);

  // Own army selected (not in battle)
  const ownArmy = !battle && !isAITurn && selectedArmyId && selectedArmyId.owner === currentPlayer.name ? findArmy(state, selectedArmyId) : null;
  const selectedCount = countOf(selectedUnits);
  // Nodes every selected unit can reach: the units of one state share their remaining moves
  const moveDestinations = ownArmy && selectedArmyId && selectedCount > 0
    ? new Set(api.getMovableLocations(selectedArmyId).filter((location) => {
        const candidates = api.getMoveCandidates(selectedArmyId, location);
        return selectedUnits.every((selection) => candidates.some((unit) => sameState(unit, selection)));
      }))
    : new Set<string>();
  const moveHint = selectedCount === 0
    ? {text: "Select units to move or disband", warning: false}
    : moveDestinations.size > 0
      ? {text: `Click a highlighted node to move ${selectedCount} ${selectedCount === 1 ? "unit" : "units"}`, warning: false}
      : selectedUnits.some((selection) => selection.remainingMoves === 0)
        ? {text: "Some selected units have no moves left", warning: true}
        : {text: "No node within reach of the selected units", warning: true};

  const winner = api.getWinner();

  return (
    <Box className="local-scroll-root" sx={{flexDirection: "row"}}>
      <Box ref={canvasContainerRef} className="inflex-fill flex-center-nowrap" sx={{position: "relative"}}>
        <GameCanvas
          api={api}
          state={state}
          version={version}
          selectedArmyId={canvasSelectedArmyId}
          selectedNode={selectedNode}
          moveDestinations={moveDestinations}
          highlightPulse={canvasHighlightPulse}
          highlightSolid={canvasHighlightSolid}
          targetPulse={battleTargets}
          attackableLocations={attackableLocations}
          battleAllocationDisplay={battleAttackArmyId ? battleAllocations.map(({targetId, count}) => ({attackerId: battleAttackArmyId, targetId, unitCount: count})) : []}
          battleRemaining={battleRemaining}
          onNodeClick={handleNodeClick}
          onArmyClick={handleArmyClick}
          onClearSelection={battle ? clearBattleAttack : clearSelection}
        />

        {/* ===== TOP LEFT: Unit Stats ===== */}
        <Paper elevation={2} sx={{position: "absolute", top: 8, left: 8, p: 1, ...overlayPaper}}>
          <Typography variant="caption" sx={{fontWeight: "bold", mb: 0.5, display: "block"}}>
            Unit Stats
          </Typography>
          <Box component="table" sx={{borderCollapse: "collapse", "& td, & th": {px: 0.5, py: 0.15, fontSize: "0.65rem", textAlign: "center"}, "& th": {fontWeight: "bold", borderBottom: "1px solid rgba(255,255,255,0.2)"}}}>
            <thead><tr><th></th><th>ATK</th><th>DEF</th><th>HP</th><th>RNG</th><th>SPD</th><th>$</th></tr></thead>
            <tbody>
              {Object.entries(DefaultUnitStatsMap).map(([name, s]) => (
                <tr key={name}>
                  <td style={{textAlign: "left", fontWeight: "bold"}}>{name}</td>
                  <td>{s.attack}</td><td>{s.defend}</td><td>{s.health}</td><td>{s.range}</td><td>{s.speed}</td><td>{s.cost}</td>
                </tr>
              ))}
            </tbody>
          </Box>
        </Paper>

        {/* ===== TOP RIGHT: All Players Stats ===== */}
        <Paper elevation={2} sx={{position: "absolute", top: 8, right: 8, p: 1, ...overlayPaper, width: 220}}>
          <Typography variant="caption" sx={{fontWeight: "bold", display: "block", mb: 0.5}}>
            Turn {state.turnCount}
          </Typography>
          {state.players.map(player => {
            const color = playerCssColors[player.name] ?? "#888";
            const units = player.armies.reduce((sum, a) => sum + a.units.length, 0);
            const income = api.getNodeIncome(player.name);
            const upkeep = api.getUpkeep(player.name);
            const interest = Math.floor(player.money * DefaultConfig.interestRate);
            const net = income + interest - upkeep;
            const isCurrent = player === currentPlayer;
            return (
              <Box key={player.name} sx={{mb: 1, opacity: player.defeated ? 0.4 : 1, borderLeft: isCurrent ? `3px solid ${color}` : "3px solid transparent", pl: 0.5}}>
                <Box sx={{display: "flex", alignItems: "center", gap: 0.5, mb: 0.25}}>
                  <Chip label={player.name} size="small" sx={{bgcolor: color, color: "#fff", height: 18, fontSize: "0.65rem"}}/>
                  {player.defeated && <Typography component="span" sx={{color: "#f44336", fontSize: "0.65rem", lineHeight: "18px"}}>Defeated</Typography>}
                </Box>
                <Box component="table" sx={{width: "100%", "& td": {fontSize: "0.65rem", py: 0}, "& td:last-child": {textAlign: "right"}}}>
                  <tbody>
                    <tr><td>Cash</td><td>${player.money}</td></tr>
                    <tr><td>Units</td><td>{units}</td></tr>
                    <tr><td>Income</td><td>+{income}</td></tr>
                    <tr><td>Interest ({DefaultConfig.interestRate * 100}%)</td><td>+{interest}</td></tr>
                    <tr><td>Upkeep</td><td style={{color: upkeep > 0 ? "#f44336" : undefined}}>-{upkeep}</td></tr>
                    <tr><td><b>Net/turn</b></td><td style={{color: net >= 0 ? "#66bb6a" : "#f44336", fontWeight: "bold"}}>{net >= 0 ? "+" : ""}{net}</td></tr>
                  </tbody>
                </Box>
              </Box>
            );
          })}
          {gameOver && (
            <Typography variant="body2" sx={{fontWeight: "bold", mt: 0.5, color: winner ? playerCssColors[winner] : undefined}}>
              {winner ? `${winner} Wins!` : "Draw!"}
            </Typography>
          )}
        </Paper>

        {/* ===== TOP CENTER: Battle Status ===== */}
        {battle && (
          <Paper elevation={3} sx={{position: "absolute", top: 8, left: "50%", transform: "translateX(-50%)", p: 1, ...overlayPaper}}>
            <Typography variant="body2" sx={{fontWeight: "bold"}}>
              Battle at {battle.targetLocation} — Round {battle.round} — {
                battle.phase === BattlePhase.AttackerTurn ? battle.attackerPlayer.name : battle.defenderPlayer.name
              }'s Turn
            </Typography>
          </Paper>
        )}

        {/* ===== TOP CENTER: AI failure ===== */}
        {aiError && (
          <Paper elevation={3} sx={{position: "absolute", top: 48, left: "50%", transform: "translateX(-50%)", p: 1, ...overlayPaper, maxWidth: 520}}>
            <Typography variant="body2" sx={{fontWeight: "bold", color: "#f44336"}}>
              AI failed: {aiError}
            </Typography>
          </Paper>
        )}

        {/* ===== BOTTOM CENTER: Context Action Area ===== */}

        {/* Pre-battle targeting — positioned below target node */}
        {battleTarget && !battle && (() => {
          const pos = getNodeScreenPos(battleTarget);
          if (!pos) return null;
          const picked = [...battlePicked.values()];
          const total = picked.reduce((sum, {selections}) => sum + countOf(selections), 0);
          const ready = picked.length > 0 && picked.every(({selections}) => countOf(selections) > 0);
          return (
            <Paper elevation={3} sx={{position: "absolute", left: pos.left, top: pos.top + 45 * pos.scale, transform: "translateX(-50%)", p: 1, ...overlayPaper, display: "flex", flexDirection: "column", gap: 1, width: 420}}>
              {picked.map(({armyId, selections}) => {
                const army = findArmy(state, armyId);
                if (!army) return null;
                return (
                  <Box key={keyOfId(armyId)}>
                    <Typography variant="caption" sx={{fontWeight: "bold"}}>{armyId.unitType} at {armyId.location}</Typography>
                    <UnitPicker units={attackReady(army)} selected={selections} onChange={(next) => setBattlePicked(prev => new Map(prev).set(keyOfId(armyId), {armyId, selections: next}))}/>
                  </Box>
                );
              })}
              <Button variant="contained" color="error" size="small" disabled={!ready} sx={{textTransform: "none"}} onClick={handleStartBattle}>
                Start Battle ({total})
              </Button>
            </Paper>
          );
        })()}

        {/* Recruit UI */}
        {canRecruitAtSelected && (
          <Paper elevation={3} sx={{position: "absolute", bottom: 8, left: 8, p: 1.5, ...overlayPaper, display: "flex", flexDirection: "column", gap: 1, width: 340}}>
            <Select size="small" value={unitType} onChange={(e) => setUnitType(e.target.value as DefaultUnitName)} fullWidth>
              {Object.entries(state.unitStatsMap).map(([name, stats]) => (
                <MenuItem key={name} value={name}>{name} (${stats.cost})</MenuItem>
              ))}
            </Select>
            <Box sx={{display: "flex", alignItems: "center", gap: 0.5}}>
              <Slider size="small" value={recruitCountClamped} min={1} max={maxRecruit}
                onChange={(_, v) => setRecruitCount(v as number)} sx={{flex: 1}}/>
              <Button size="small" variant="text" sx={{minWidth: 28, p: 0}} onClick={() => setRecruitCount(Math.max(1, recruitCountClamped - 1))}>-</Button>
              <Typography variant="body2" sx={{width: 28, textAlign: "center"}}>{recruitCountClamped}</Typography>
              <Button size="small" variant="text" sx={{minWidth: 28, p: 0}} onClick={() => setRecruitCount(Math.min(maxRecruit, recruitCountClamped + 1))}>+</Button>
              <Button
                variant="contained" size="small" sx={{textTransform: "none", width: 72}}
                disabled={!api.canBuy(unitType, recruitCountClamped)}
                onClick={() => { apiRef.current.recruit(unitType, selectedNode!, recruitCountClamped); update(); }}
              >
                Recruit
              </Button>
            </Box>
          </Paper>
        )}

        {/* Army management UI */}
        {ownArmy && (
          <Paper elevation={3} sx={{position: "absolute", bottom: 8, left: 8, p: 1.5, ...overlayPaper, display: "flex", flexDirection: "column", gap: 1, width: 420}}>
            <Typography variant="body2" sx={{fontWeight: "bold"}}>
              {ownArmy.units.length} {ownArmy.unitType} at {ownArmy.location}
            </Typography>
            <UnitPicker units={ownArmy.units} selected={selectedUnits} onChange={setSelectedUnits}/>
            <Box sx={{display: "flex", alignItems: "center", gap: 0.5}}>
              <Typography variant="caption" sx={{flex: 1, color: moveHint.warning ? "#FFB300" : "text.secondary"}}>
                {moveHint.text}
              </Typography>
              <Button variant="contained" color="error" size="small" sx={{textTransform: "none", width: 72}} disabled={selectedCount === 0} onClick={handleDisband}>
                Disband
              </Button>
            </Box>
          </Paper>
        )}

        {/* Battle action overlay — positioned below target node */}
        {battle && !battleAttackArmyId && !isAITurn
          && battle.phase === BattlePhase.AttackerTurn && battle.actedArmies.size === 0 && (() => {
          const pos = getNodeScreenPos(battle.targetLocation);
          if (!pos) return null;
          const oneVsOne = battle.attackerArmies.length === 1 && battle.defenderArmies.length === 1;
          return (
            <Paper elevation={3} sx={{position: "absolute", left: pos.left, top: pos.top + 45 * pos.scale, transform: "translateX(-50%)", p: 1, ...overlayPaper, display: "flex", gap: 0.5}}>
              {oneVsOne && (
                <Button variant="contained" size="small" sx={{textTransform: "none"}} onClick={handleBattleAttackAll}>Attack</Button>
              )}
              <Button variant="contained" color="error" size="small" sx={{textTransform: "none"}} onClick={handleBattleRetreat}>Retreat</Button>
            </Paper>
          );
        })()}

        {/* Send allocation slider — positioned above target army's node */}
        {attackingArmyId && battleAllocTargetId && battleRemaining > 0 && (() => {
          const pos = getNodeScreenPos(battleAllocTargetId.location);
          if (!pos) return null;
          const needed = api.getUnitsNeeded(attackingArmyId, battleAllocTargetId);
          const sliderMax = Math.min(battleRemaining, needed);
          return (
            <Paper elevation={3} sx={{position: "absolute", left: pos.left, top: pos.top - 90 * pos.scale, transform: "translateX(-50%)", p: 1, ...overlayPaper, width: 280}}>
              <Box sx={{display: "flex", alignItems: "center", gap: 0.5}}>
                <Slider size="small" value={battleAllocCount} min={1} max={sliderMax}
                  onChange={(_, v) => setBattleAllocCount(v as number)} sx={{flex: 1}}/>
                <Button size="small" variant="text" sx={{minWidth: 28, p: 0}} onClick={() => setBattleAllocCount(c => Math.max(1, c - 1))}>-</Button>
                <Typography variant="body2" sx={{width: 28, textAlign: "center"}}>{battleAllocCount}</Typography>
                <Button size="small" variant="text" sx={{minWidth: 28, p: 0}} onClick={() => setBattleAllocCount(c => Math.min(sliderMax, c + 1))}>+</Button>
                <Button variant="contained" size="small" sx={{textTransform: "none", width: 72}} onClick={handleBattleAllocate}>
                  Send
                </Button>
              </Box>
            </Paper>
          );
        })()}

        {/* Confirm Attack / Reset — positioned below target node */}
        {battle && attackingArmyId && battleAllocations.length > 0 && !battleAllocTargetId && (() => {
          const pos = getNodeScreenPos(battle.targetLocation);
          if (!pos) return null;
          return (
            <Paper elevation={3} sx={{position: "absolute", left: pos.left, top: pos.top + 45 * pos.scale, transform: "translateX(-50%)", p: 1, ...overlayPaper, display: "flex", gap: 0.5}}>
              <Button variant="contained" size="small" sx={{textTransform: "none"}} onClick={handleBattleConfirm}>Confirm</Button>
              <Button variant="outlined" size="small" sx={{textTransform: "none"}} onClick={() => { setBattleAllocations([]); setBattleAllocTargetId(null); }}>Reset</Button>
            </Paper>
          );
        })()}

        {/* ===== BOTTOM RIGHT: End Turn / Save / Load / Menu ===== */}
        {!battle && (
          <Paper elevation={2} sx={{position: "absolute", bottom: 8, right: 8, p: 1, ...overlayPaper, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 0.5}}>
            <Button variant="outlined" size="small" sx={{textTransform: "none"}} onClick={() => setSaveLoadMode("save")}>Save</Button>
            <Button variant="outlined" size="small" sx={{textTransform: "none"}} onClick={() => setSaveLoadMode("load")}>Load</Button>
            {gameOver ? (
              <Button variant="contained" size="small" sx={{textTransform: "none", gridColumn: "1 / -1"}} onClick={() => {
                apiRef.current = GameAPI.create(DefaultConfig);
                setVersion(0);
                clearSelection();
                clearBattleAttack();
                setGameMode("menu");
              }}>
                Back to Menu
              </Button>
            ) : (
              <>
                <Button variant="contained" color="secondary" size="small" sx={{textTransform: "none", whiteSpace: "nowrap"}} disabled={isAITurn} onClick={handleEndTurn}>
                  End Turn
                </Button>
                <Button variant="outlined" size="small" sx={{textTransform: "none"}} onClick={() => {
                  apiRef.current = GameAPI.create(DefaultConfig);
                  setVersion(0);
                  clearSelection();
                  clearBattleAttack();
                  setGameMode("menu");
                }}>Menu</Button>
              </>
            )}
          </Paper>
        )}

        {/* ===== Save/Load overlay ===== */}
        {saveLoadMode && (
          <Paper elevation={4} sx={{position: "absolute", bottom: 50, right: 8, p: 1.5, ...overlayPaper, width: 250, display: "flex", flexDirection: "column", gap: 1}}>
            <Typography variant="subtitle2">{saveLoadMode === "save" ? "Save Game" : "Load Game"}</Typography>

            {saveLoadMode === "save" && (
              <Box sx={{display: "flex", gap: 0.5}}>
                <TextField size="small" value={saveName} onChange={(e) => setSaveName(e.target.value)}
                  placeholder="Save name" sx={{flex: 1}} onKeyDown={(e) => { if (e.key === "Enter") handleSave(saveName); }}/>
                <Button variant="contained" size="small" sx={{textTransform: "none"}} disabled={!saveName.trim()} onClick={() => handleSave(saveName)}>
                  Save
                </Button>
              </Box>
            )}

            {getSaveList().map(name => (
              <Box key={name} sx={{display: "flex", alignItems: "center", gap: 0.5}}>
                <Typography variant="body2" sx={{flex: 1, overflow: "hidden", textOverflow: "ellipsis"}}>{name}</Typography>
                {saveLoadMode === "save" ? (
                  <Button size="small" variant="outlined" sx={{textTransform: "none", minWidth: 0}} onClick={() => handleSave(name)}>
                    Overwrite
                  </Button>
                ) : (
                  <Button size="small" variant="contained" sx={{textTransform: "none", minWidth: 0}} onClick={() => handleLoad(name)}>
                    Load
                  </Button>
                )}
                <Button size="small" color="error" variant="text" sx={{textTransform: "none", minWidth: 0, p: 0.5}} onClick={() => handleDeleteSave(name)}>
                  X
                </Button>
              </Box>
            ))}

            {getSaveList().length === 0 && saveLoadMode === "load" && (
              <Typography variant="caption" sx={{color: "text.secondary"}}>No saves found</Typography>
            )}

            <Button size="small" variant="text" sx={{textTransform: "none", alignSelf: "flex-end"}} onClick={() => setSaveLoadMode(null)}>
              Close
            </Button>
          </Paper>
        )}
      </Box>
    </Box>
  );
}

export default App;
