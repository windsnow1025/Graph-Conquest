import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import ButtonGroup from "@mui/material/ButtonGroup";
import Slider from "@mui/material/Slider";
import Typography from "@mui/material/Typography";
import FavoriteIcon from "@mui/icons-material/Favorite";
import DirectionsRunIcon from "@mui/icons-material/DirectionsRun";
import GpsFixedIcon from "@mui/icons-material/GpsFixed";
import type {UnitData} from "../lib/Unit";
import type {UnitSelection} from "../lib/GameAPI";
import {sameState} from "../AI/nn/UnitGroups";

interface UnitPickerProps {
  units: UnitData[];
  selected: UnitSelection[];
  onChange: (selected: UnitSelection[]) => void;
}

/** The units of one army that agree on health, remaining moves and attack readiness, with the count of them selected. */
interface UnitGroup {
  key: string;
  unit: UnitData;
  total: number;
  count: number;
}

// Same colors as the status dots on the map: green = can move, red = can attack
const Move_Color = "#4CAF50";
const Attack_Color = "#FF5252";
const Idle_Color = "rgba(255,255,255,0.22)";
const Columns = "76px 44px 24px 1fr 24px 40px 24px";

function healthColor(ratio: number): string {
  if (ratio >= 0.7) return "#4CAF50";
  if (ratio >= 0.3) return "#FFB300";
  return "#FF5252";
}

// Units of one army differ only in health, remaining moves, and attack readiness;
// units that agree on all three are interchangeable, so the player picks a count
// per group instead of individual units, which is how the API selects units.
function groupUnits(units: UnitData[], selected: UnitSelection[]): UnitGroup[] {
  const groups = new Map<string, UnitGroup>();
  for (const unit of units) {
    const key = `${unit.currentHealth}|${unit.remainingMoves}|${unit.canAttack}`;
    const group = groups.get(key);
    if (group) group.total++;
    else groups.set(key, {key, unit, total: 1, count: selected.find(selection => sameState(selection, unit))?.count ?? 0});
  }
  return [...groups.values()].sort((a, b) => {
    if (a.unit.currentHealth !== b.unit.currentHealth) return b.unit.currentHealth - a.unit.currentHealth;
    if (a.unit.remainingMoves !== b.unit.remainingMoves) return b.unit.remainingMoves - a.unit.remainingMoves;
    return Number(b.unit.canAttack) - Number(a.unit.canAttack);
  });
}

function selectionOf(group: UnitGroup, count: number): UnitSelection {
  const {currentHealth, remainingMoves, canAttack} = group.unit;
  return {currentHealth, remainingMoves, canAttack, count};
}

function UnitPicker({units, selected, onChange}: UnitPickerProps) {
  const groups = groupUnits(units, selected);
  const selectedCount = groups.reduce((sum, group) => sum + group.count, 0);

  const setGroupCount = (target: UnitGroup, count: number) => {
    onChange(groups.map(group => selectionOf(group, group === target ? count : group.count)).filter(selection => selection.count > 0));
  };

  return (
    <Box sx={{display: "flex", flexDirection: "column", gap: 0.5}}>
      <Box sx={{display: "grid", gridTemplateColumns: Columns, alignItems: "center", columnGap: 1, color: "text.secondary"}}>
        <FavoriteIcon sx={{fontSize: 14, justifySelf: "center"}}/>
        <DirectionsRunIcon sx={{fontSize: 16, justifySelf: "center"}}/>
        <GpsFixedIcon sx={{fontSize: 14, justifySelf: "center"}}/>
        <Typography variant="caption" sx={{gridColumn: "5 / span 3", textAlign: "center"}}>Take</Typography>
      </Box>
      {groups.map((group) => {
        const {unit, total, count} = group;
        const ratio = unit.currentHealth / unit.health;
        return (
          <Box key={group.key} sx={{display: "grid", gridTemplateColumns: Columns, alignItems: "center", columnGap: 1}}>
            <Box sx={{position: "relative", height: 16, borderRadius: 0.5, bgcolor: "rgba(255,255,255,0.12)", overflow: "hidden"}}>
              <Box sx={{position: "absolute", inset: 0, width: `${ratio * 100}%`, bgcolor: healthColor(ratio), opacity: 0.85}}/>
              <Typography sx={{position: "relative", fontSize: "0.65rem", lineHeight: "16px", textAlign: "center", fontWeight: "bold", fontVariantNumeric: "tabular-nums"}}>
                {unit.currentHealth}/{unit.health}
              </Typography>
            </Box>
            <Box sx={{display: "flex", justifyContent: "center", gap: 0.5}}>
              {Array.from({length: unit.speed}, (_, i) => (
                <Box key={i} sx={{width: 9, height: 9, borderRadius: "50%", bgcolor: i < unit.remainingMoves ? Move_Color : Idle_Color}}/>
              ))}
            </Box>
            <GpsFixedIcon sx={{fontSize: 16, justifySelf: "center", color: unit.canAttack ? Attack_Color : Idle_Color}}/>
            <Slider size="small" value={count} min={0} max={total}
              onChange={(_, v) => setGroupCount(group, v as number)}/>
            <Button size="small" variant="text" sx={{minWidth: 24, p: 0}} onClick={() => setGroupCount(group, Math.max(0, count - 1))}>-</Button>
            <Typography variant="body2" sx={{textAlign: "center", fontVariantNumeric: "tabular-nums"}}>{count}/{total}</Typography>
            <Button size="small" variant="text" sx={{minWidth: 24, p: 0}} onClick={() => setGroupCount(group, Math.min(total, count + 1))}>+</Button>
          </Box>
        );
      })}
      <Box sx={{display: "flex", alignItems: "center", justifyContent: "space-between"}}>
        <Typography variant="caption" sx={{fontVariantNumeric: "tabular-nums"}}>Selected {selectedCount}/{units.length}</Typography>
        <ButtonGroup size="small" variant="outlined">
          <Button sx={{textTransform: "none", width: 56, py: 0}} onClick={() => onChange(groups.map(group => selectionOf(group, group.total)))}>All</Button>
          <Button sx={{textTransform: "none", width: 56, py: 0}} onClick={() => onChange([])}>None</Button>
        </ButtonGroup>
      </Box>
    </Box>
  );
}

export default UnitPicker;
