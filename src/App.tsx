import {useState, useRef, useCallback, useEffect} from "react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Slider from "@mui/material/Slider";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import TextField from "@mui/material/TextField";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import Chip from "@mui/material/Chip";
import GameSystem, {type GameSave} from "./lib/GameSystem";
import Config from "./lib/data/Config";
import Player from "./lib/Player";
import Army from "./lib/Army";
import type Unit from "./lib/Unit";
import {BattlePhase, BattleResult} from "./lib/Battle";
import type {DefaultUnitName} from "./lib/data/DefaultUnitStatsMap.ts";
import DefaultUnitStatsMap from "./lib/data/DefaultUnitStatsMap.ts";
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

function App() {
  const gameRef = useRef(new GameSystem(Config));
  const [version, setVersion] = useState(0);
  const [selectedArmy, setSelectedArmy] = useState<Army | null>(null);
  const [selectedPlayer, setSelectedPlayer] = useState<Player | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [gameMode, setGameMode] = useState<GameMode>("menu");
  const [gameEpoch, setGameEpoch] = useState(0); // bumped when gameRef is replaced, so game-bound effects restart
  const canvasContainerRef = useRef<HTMLDivElement>(null);

  // Recruit UI
  const [unitType, setUnitType] = useState<DefaultUnitName>("Infantry");
  const [recruitCount, setRecruitCount] = useState(1);

  // Army management UI
  const [selectedUnits, setSelectedUnits] = useState<Set<Unit>>(new Set());

  // Pre-battle targeting
  const [battleTarget, setBattleTarget] = useState<string | null>(null);
  const [battlePicked, setBattlePicked] = useState<Map<Army, Unit[]>>(new Map());

  // In-battle attack allocation
  const [battleAttackArmy, setBattleAttackArmy] = useState<Army | null>(null);
  const [battleAllocations, setBattleAllocations] = useState<Map<Army, number>>(new Map());
  const [battleAllocTarget, setBattleAllocTarget] = useState<Army | null>(null);
  const [battleAllocCount, setBattleAllocCount] = useState(1);

  // Save/Load
  const [saveLoadMode, setSaveLoadMode] = useState<"save" | "load" | null>(null);
  const [saveName, setSaveName] = useState("");

  const game = gameRef.current;
  const battle = game.currentBattle;
  const currentPlayer = game.currentPlayer;

  const update = useCallback(() => {
    setVersion(v => v + 1);
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedArmy(null);
    setSelectedPlayer(null);
    setSelectedNode(null);
    setSelectedUnits(new Set());
    setBattleTarget(null);
    setBattlePicked(new Map());
  }, []);

  const clearBattleAttack = useCallback(() => {
    setBattleAttackArmy(null);
    setBattleAllocations(new Map());
    setBattleAllocTarget(null);
    setBattleAllocCount(1);
  }, []);

  const selectDefaultNode = useCallback(() => {
    const g = gameRef.current;
    const player = g.currentPlayer;
    if (g.nodeOwnership.get(player.homeLocation) === player) {
      setSelectedNode(player.homeLocation);
      return;
    }
    const recruitLocs = g.recruitLocations;
    if (recruitLocs.length === 1) {
      setSelectedNode(recruitLocs[0]);
    }
  }, []);

  // --- AI turn ---
  const aiPlayers = gameMode === "autoplay" ? ALL_PLAYERS
    : gameMode === "ai" || gameMode === "greedy" ? VS_AI_PLAYERS
    : NO_PLAYERS;
  const isAITurn = (gameMode === "ai" || gameMode === "greedy" || gameMode === "autoplay") && !game.gameOver && aiPlayers.has(currentPlayer.name);

  useEffect(() => {
    if (!isAITurn) return;
    let cancelled = false;
    const runAllAITurns = async () => {
      const g = gameRef.current;
      const aiSet = gameMode === "autoplay" ? ALL_PLAYERS : VS_AI_PLAYERS;
      while (!cancelled && !g.gameOver && aiSet.has(g.currentPlayer.name)) {
        const gen = gameMode === "greedy"
          ? greedyTurnStepsUI(g)
          : await aiTurnSteps(g);
        if (cancelled) return;
        if (!gen) { update(); continue; }
        await new Promise<void>((resolve) => {
          const step = () => {
            if (cancelled) { resolve(); return; }
            const result = gen.next();
            update();
            if (!result.done) setTimeout(step, 0); else resolve();
          };
          setTimeout(step, 0);
        });
      }
      if (!cancelled) { clearSelection(); selectDefaultNode(); update(); }
    };
    runAllAITurns();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAITurn, gameMode, gameEpoch]);

  // --- Battle auto-actions (skip during AI turns — AI handles battles itself) ---
  useEffect(() => {
    if (!battle || isAITurn) return;

    // Auto-resolve when battle ends
    if (battle.result !== BattleResult.Ongoing) {
      gameRef.current.resolveBattle();
      clearBattleAttack();
      update();
      return;
    }

    // AI-controlled defender uses their model (NN or greedy)
    if (battle.phase === BattlePhase.DefenderTurn && aiPlayers.has(battle.defenderPlayer.name)) {
      aiDefenderPhase(gameRef.current, battle, gameMode).then(update);
      return;
    }

    // Auto-allocate the defender's forced move when both sides have exactly 1 army.
    // The attacker is never auto-played: acting would foreclose the retreat decision.
    if (battle.phase === BattlePhase.DefenderTurn
      && battle.attackerArmies.length === 1 && battle.defenderArmies.length === 1) {
      const army = battle.defenderArmies[0];
      const target = battle.attackerArmies[0];
      if (battle.canAct(army)) {
        battle.allocateAttack(army, new Map([[target, army.battleUnits.length]]));
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
    gameRef.current = new GameSystem(Config);
    setVersion(0);
    clearSelection();
    clearBattleAttack();
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
    const save = gameRef.current.toJSON();
    localStorage.setItem(SAVE_PREFIX + name.trim(), JSON.stringify(save));
    setSaveLoadMode(null);
    setSaveName("");
  };

  const handleLoad = (name: string) => {
    const raw = localStorage.getItem(SAVE_PREFIX + name);
    if (!raw) return;
    const save: GameSave = JSON.parse(raw);
    gameRef.current = GameSystem.fromJSON(save);
    setGameEpoch(e => e + 1);
    clearSelection();
    clearBattleAttack();
    setSaveLoadMode(null);
    update();
  };

  const handleDeleteSave = (name: string) => {
    localStorage.removeItem(SAVE_PREFIX + name);
    update();
  };

  const handleEndTurn = useCallback(() => {
    gameRef.current.endTurn();
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
  const armiesInRange = battleTarget
    ? new Set(gameRef.current.getArmiesInRange(battleTarget))
    : new Set<Army>();

  const battleUnacted = battle && battle.result === BattleResult.Ongoing
    ? new Set(battle.unactedArmies)
    : new Set<Army>();
  const allocatedTargets = new Set(battleAllocations.keys());
  const battleTargets = battle && battleAttackArmy && !battleAllocTarget
    ? new Set(battle.getTargetsInRange(battleAttackArmy).filter(a => !allocatedTargets.has(a)))
    : new Set<Army>();

  const battleRemaining = battleAttackArmy
    ? battleAttackArmy.battleUnits.length - [...battleAllocations.values()].reduce((sum, count) => sum + count, 0)
    : 0;

  // --- Click handlers ---
  const attackableLocations = !battle && !battleTarget && !selectedArmy && !isAITurn ? game.attackableLocations : new Set<string>();

  const handleNodeClick = useCallback((location: string) => {
    if (battle || isAITurn) return;

    // Cancel targeting mode
    if (battleTarget) {
      setBattleTarget(null);
      setBattlePicked(new Map());
      return;
    }

    // Move selected army to clicked node
    if (selectedArmy && selectedPlayer === gameRef.current.currentPlayer) {
      gameRef.current.movePlayerUnits(selectedArmy, location, [...selectedUnits]);
      clearSelection();
      update();
      return;
    }

    // Click attackable enemy node → enter battle targeting
    if (attackableLocations.has(location)) {
      setSelectedArmy(null);
      setSelectedPlayer(null);
      setSelectedNode(null);
      setBattleTarget(location);
      setBattlePicked(new Map());
      return;
    }

    // Select node (for recruit UI etc.)
    setSelectedArmy(null);
    setSelectedPlayer(null);
    setSelectedNode(location);
  }, [battle, isAITurn, selectedArmy, selectedPlayer, selectedUnits, battleTarget, attackableLocations, update, clearSelection]);

  const handleArmyClick = useCallback((army: Army, player: Player) => {
    if (isAITurn) return;
    const g = gameRef.current;

    if (g.currentBattle) {
      const b = g.currentBattle;
      if (b.result !== BattleResult.Ongoing) return;
      if (b.currentArmies.includes(army) && b.canAct(army)) {
        setBattleAttackArmy(army);
        setBattleAllocations(new Map());
        setBattleAllocTarget(null);
        setBattleAllocCount(1);
        return;
      }
      if (battleAttackArmy && battleRemaining > 0) {
        const targets = b.getTargetsInRange(battleAttackArmy);
        if (targets.includes(army) && !battleAllocations.has(army)) {
          const needed = b.getUnitsNeeded(battleAttackArmy, army);
          setBattleAllocTarget(army);
          setBattleAllocCount(Math.min(needed, battleRemaining));
          return;
        }
      }
      return;
    }

    const current = g.currentPlayer;
    if (battleTarget && player === current) {
      if (!armiesInRange.has(army)) return;
      setBattlePicked((prev) => {
        const next = new Map(prev);
        if (next.has(army)) next.delete(army);
        else next.set(army, army.attackCandidates);
        return next;
      });
      return;
    }

    // Click enemy army → treat as clicking the node (enter targeting if attackable)
    if (player !== current) {
      if (attackableLocations.has(army.location)) {
        setSelectedArmy(null);
        setSelectedPlayer(null);
        setSelectedNode(null);
        setBattleTarget(army.location);
        setBattlePicked(new Map());
      }
      return;
    }

    // Click own army → select it
    setBattleTarget(null);
    setBattlePicked(new Map());
    setSelectedArmy(army);
    setSelectedPlayer(player);
    setSelectedUnits(new Set(army.units));
    setSelectedNode(null);
  }, [battle, isAITurn, battleTarget, battleAttackArmy, battleRemaining, armiesInRange, attackableLocations]);

  // --- Army actions ---
  const handleDisband = useCallback(() => {
    if (!selectedArmy) return;
    gameRef.current.disbandPlayerUnits(selectedArmy, [...selectedUnits]);
    if (selectedArmy.units.length === 0) clearSelection();
    else setSelectedUnits(new Set(selectedArmy.units));
    update();
  }, [selectedArmy, selectedUnits, update, clearSelection]);

  // --- Battle actions ---
  const handleStartBattle = useCallback(() => {
    if (!battleTarget || battlePicked.size === 0) return;
    gameRef.current.startBattle(battleTarget, battlePicked);
    clearSelection();
    clearBattleAttack();
    update();
  }, [battleTarget, battlePicked, update, clearSelection, clearBattleAttack]);

  const handleBattleAllocate = useCallback(() => {
    if (!battleAllocTarget) return;
    setBattleAllocations(prev => new Map(prev).set(battleAllocTarget, battleAllocCount));
    setBattleAllocTarget(null);
    setBattleAllocCount(1);
  }, [battleAllocTarget, battleAllocCount]);

  const handleBattleConfirm = useCallback(() => {
    if (!battleAttackArmy || !battle) return;
    battle.allocateAttack(battleAttackArmy, battleAllocations);
    clearBattleAttack();
    update();
  }, [battle, battleAttackArmy, battleAllocations, update, clearBattleAttack]);


  const handleBattleRetreat = useCallback(() => {
    if (!battle) return;
    battle.retreat();
    clearBattleAttack();
    update();
  }, [battle, update, clearBattleAttack]);

  // One-click full allocation for the 1v1 case (replaces the former auto-allocation)
  const handleBattleAttackAll = useCallback(() => {
    if (!battle) return;
    if (battle.attackerArmies.length !== 1 || battle.defenderArmies.length !== 1) return;
    const army = battle.attackerArmies[0];
    const target = battle.defenderArmies[0];
    if (battle.canAct(army)) {
      battle.allocateAttack(army, new Map([[target, army.battleUnits.length]]));
    }
    clearBattleAttack();
    update();
  }, [battle, update, clearBattleAttack]);


  // --- Render ---
  if (gameMode === "menu") {
    return <ModeSelect onSelect={handleModeSelect}/>;
  }

  const canvasSelectedArmy = battle ? battleAttackArmy : selectedArmy;

  const canvasHighlightPulse = battle
    ? (battleAttackArmy ? new Set<Army>() : battleUnacted)
    : battleTarget ? armiesInRange : new Set<Army>();
  const canvasHighlightSolid = battle
    ? new Set([
        ...battleAllocations.keys(),
        ...(battleAllocTarget ? [battleAllocTarget] : []),
      ])
    : new Set(battlePicked.keys());

  // Recruit conditions
  const canRecruitAtSelected = !battle && !isAITurn && selectedNode !== null
    && game.recruitLocations.includes(selectedNode);
  const maxRecruit = Math.max(1, Math.floor(currentPlayer.money / game.unitStatsMap[unitType].cost) || 1);
  const recruitCountClamped = Math.min(recruitCount, maxRecruit);

  // Own army selected (not in battle)
  const ownArmySelected = !battle && !isAITurn && selectedArmy && selectedPlayer === currentPlayer;
  // Nodes every selected unit can reach
  const moveDestinations = ownArmySelected && selectedUnits.size > 0
    ? new Set(selectedArmy.getMovableLocations(game.gameMap, game.enemyLocations).filter((location) => {
        const candidates = new Set(selectedArmy.getMoveCandidates(location, game.gameMap, game.enemyLocations));
        return [...selectedUnits].every((unit) => candidates.has(unit));
      }))
    : new Set<string>();
  const moveHint = selectedUnits.size === 0
    ? {text: "Select units to move or disband", warning: false}
    : moveDestinations.size > 0
      ? {text: `Click a highlighted node to move ${selectedUnits.size} ${selectedUnits.size === 1 ? "unit" : "units"}`, warning: false}
      : [...selectedUnits].some((unit) => unit.remainingMoves === 0)
        ? {text: "Some selected units have no moves left", warning: true}
        : {text: "No node within reach of the selected units", warning: true};

  return (
    <Box className="local-scroll-root" sx={{flexDirection: "row"}}>
      <Box ref={canvasContainerRef} className="inflex-fill flex-center-nowrap" sx={{position: "relative"}}>
        <GameCanvas
          game={game}
          version={version}
          selectedArmy={canvasSelectedArmy}
          selectedNode={selectedNode}
          moveDestinations={moveDestinations}
          highlightPulse={canvasHighlightPulse}
          highlightSolid={canvasHighlightSolid}
          targetPulse={battleTargets}
          attackableLocations={attackableLocations}
          battleAllocationDisplay={battleAttackArmy ? [...battleAllocations].map(([target, unitCount]) => ({attacker: battleAttackArmy, target, unitCount})) : []}
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
            Turn {game.turnCount}
          </Typography>
          {game.players.map(player => {
            const color = playerCssColors[player.name] ?? "#888";
            const units = player.armies.reduce((sum, a) => sum + a.units.length, 0);
            let income = 0;
            for (const [node, owner] of game.nodeOwnership) {
              if (owner === player) income += game.gameMap.getNodeData(node)!.income;
            }
            const upkeep = player.getUpkeep(Config.upkeepRate);
            const interest = Math.floor(player.money * Config.interestRate);
            const net = income + interest - upkeep;
            const isCurrent = player === currentPlayer;
            return (
              <Box key={player.name} sx={{mb: 1, opacity: player.defeated ? 0.4 : 1, borderLeft: isCurrent ? `3px solid ${color}` : "3px solid transparent", pl: 0.5}}>
                <Box sx={{display: "flex", alignItems: "center", gap: 0.5, mb: 0.25}}>
                  <Chip label={player.name} size="small" sx={{bgcolor: color, color: "#fff", height: 18, fontSize: "0.65rem"}}/>
                  {player.defeated && <Typography variant="caption" sx={{color: "#f44336"}}>Defeated</Typography>}
                </Box>
                <Box component="table" sx={{width: "100%", "& td": {fontSize: "0.65rem", py: 0}, "& td:last-child": {textAlign: "right"}}}>
                  <tbody>
                    <tr><td>Cash</td><td>${player.money}</td></tr>
                    <tr><td>Units</td><td>{units}</td></tr>
                    <tr><td>Income</td><td>+{income}</td></tr>
                    <tr><td>Interest ({Config.interestRate * 100}%)</td><td>+{interest}</td></tr>
                    <tr><td>Upkeep</td><td style={{color: upkeep > 0 ? "#f44336" : undefined}}>-{upkeep}</td></tr>
                    <tr><td><b>Net/turn</b></td><td style={{color: net >= 0 ? "#66bb6a" : "#f44336", fontWeight: "bold"}}>{net >= 0 ? "+" : ""}{net}</td></tr>
                  </tbody>
                </Box>
              </Box>
            );
          })}
          {game.gameOver && (
            <Typography variant="body2" sx={{fontWeight: "bold", mt: 0.5, color: game.winner ? playerCssColors[game.winner.name] : undefined}}>
              {game.winner ? `${game.winner.name} Wins!` : "Draw!"}
            </Typography>
          )}
        </Paper>

        {/* ===== TOP CENTER: Battle Status ===== */}
        {battle && (
          <Paper elevation={3} sx={{position: "absolute", top: 8, left: "50%", transform: "translateX(-50%)", p: 1, ...overlayPaper}}>
            {battle.result === BattleResult.Ongoing ? (
              <Typography variant="body2" sx={{fontWeight: "bold"}}>
                Battle at {battle.targetLocation} — Round {battle.round} — {
                  battle.phase === BattlePhase.AttackerTurn ? battle.attackerPlayer.name : battle.defenderPlayer.name
                }'s Turn
              </Typography>
            ) : (
              <Typography variant="body2" sx={{fontWeight: "bold"}}>
                {battle.result === BattleResult.AttackerWins && `${battle.attackerPlayer.name} Wins!`}
                {battle.result === BattleResult.DefenderWins && `${battle.defenderPlayer.name} Wins!`}
                {battle.result === BattleResult.Retreat && `${battle.attackerPlayer.name} Retreated`}
                {battle.result === BattleResult.Draw && "Draw (Attacks Exhausted)"}
              </Typography>
            )}
          </Paper>
        )}

        {/* ===== BOTTOM CENTER: Context Action Area ===== */}

        {/* Pre-battle targeting — positioned below target node */}
        {battleTarget && !battle && (() => {
          const pos = getNodeScreenPos(battleTarget);
          if (!pos) return null;
          const picked = [...battlePicked];
          const total = picked.reduce((sum, [, units]) => sum + units.length, 0);
          const ready = picked.length > 0 && picked.every(([, units]) => units.length > 0);
          return (
            <Paper elevation={3} sx={{position: "absolute", left: pos.left, top: pos.top + 45 * pos.scale, transform: "translateX(-50%)", p: 1, ...overlayPaper, display: "flex", flexDirection: "column", gap: 1, width: 420}}>
              {picked.map(([army, units]) => (
                <Box key={`${army.location}|${army.unitType}`}>
                  <Typography variant="caption" sx={{fontWeight: "bold"}}>{army.unitType} at {army.location}</Typography>
                  <UnitPicker units={army.attackCandidates} selected={new Set(units)} onChange={(next) => setBattlePicked(prev => new Map(prev).set(army, [...next]))}/>
                </Box>
              ))}
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
              {Object.entries(game.unitStatsMap).map(([name, stats]) => (
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
                disabled={!currentPlayer.canBuy(unitType, recruitCountClamped)}
                onClick={() => { gameRef.current.recruitPlayerArmy(unitType, selectedNode!, recruitCountClamped); update(); }}
              >
                Recruit
              </Button>
            </Box>
          </Paper>
        )}

        {/* Army management UI */}
        {ownArmySelected && (
          <Paper elevation={3} sx={{position: "absolute", bottom: 8, left: 8, p: 1.5, ...overlayPaper, display: "flex", flexDirection: "column", gap: 1, width: 420}}>
            <Typography variant="body2" sx={{fontWeight: "bold"}}>
              {selectedArmy.units.length} {selectedArmy.unitType} at {selectedArmy.location}
            </Typography>
            <UnitPicker units={selectedArmy.units} selected={selectedUnits} onChange={setSelectedUnits}/>
            <Box sx={{display: "flex", alignItems: "center", gap: 0.5}}>
              <Typography variant="caption" sx={{flex: 1, color: moveHint.warning ? "#FFB300" : "text.secondary"}}>
                {moveHint.text}
              </Typography>
              <Button variant="contained" color="error" size="small" sx={{textTransform: "none", width: 72}} disabled={selectedUnits.size === 0} onClick={handleDisband}>
                Disband
              </Button>
            </Box>
          </Paper>
        )}

        {/* Battle action overlay — positioned below target node */}
        {battle && battle.result === BattleResult.Ongoing && !battleAttackArmy && !isAITurn
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
        {battle && battleAttackArmy && battleAllocTarget && battleRemaining > 0 && (() => {
          const pos = getNodeScreenPos(battleAllocTarget.location);
          if (!pos) return null;
          const needed = battle.getUnitsNeeded(battleAttackArmy, battleAllocTarget);
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
        {battle && battleAttackArmy && battleAllocations.size > 0 && !battleAllocTarget && (() => {
          const pos = getNodeScreenPos(battle.targetLocation);
          if (!pos) return null;
          return (
            <Paper elevation={3} sx={{position: "absolute", left: pos.left, top: pos.top + 45 * pos.scale, transform: "translateX(-50%)", p: 1, ...overlayPaper, display: "flex", gap: 0.5}}>
              <Button variant="contained" size="small" sx={{textTransform: "none"}} onClick={handleBattleConfirm}>Confirm</Button>
              <Button variant="outlined" size="small" sx={{textTransform: "none"}} onClick={() => { setBattleAllocations(new Map()); setBattleAllocTarget(null); }}>Reset</Button>
            </Paper>
          );
        })()}

        {/* ===== BOTTOM RIGHT: End Turn / Save / Load / Menu ===== */}
        {!battle && (
          <Paper elevation={2} sx={{position: "absolute", bottom: 8, right: 8, p: 1, ...overlayPaper, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 0.5}}>
            <Button variant="outlined" size="small" sx={{textTransform: "none"}} onClick={() => setSaveLoadMode("save")}>Save</Button>
            <Button variant="outlined" size="small" sx={{textTransform: "none"}} onClick={() => setSaveLoadMode("load")}>Load</Button>
            {game.gameOver ? (
              <Button variant="contained" size="small" sx={{textTransform: "none", gridColumn: "1 / -1"}} onClick={() => {
                gameRef.current = new GameSystem(Config);
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
                  gameRef.current = new GameSystem(Config);
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
