import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import ButtonGroup from "@mui/material/ButtonGroup";
import Slider from "@mui/material/Slider";
import Typography from "@mui/material/Typography";
import FavoriteIcon from "@mui/icons-material/Favorite";
import DirectionsRunIcon from "@mui/icons-material/DirectionsRun";
import GpsFixedIcon from "@mui/icons-material/GpsFixed";
import type Unit from "../lib/Unit";

interface UnitPickerProps {
  units: Unit[];
  selected: Set<Unit>;
  onChange: (selected: Set<Unit>) => void;
}

interface UnitGroup {
  key: string;
  units: Unit[];
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
// per group instead of individual units.
function groupUnits(units: Unit[]): UnitGroup[] {
  const groups = new Map<string, UnitGroup>();
  for (const unit of units) {
    const key = `${unit.currentHealth}|${unit.remainingMoves}|${unit.canAttack}`;
    const group = groups.get(key);
    if (group) group.units.push(unit);
    else groups.set(key, {key, units: [unit]});
  }
  return [...groups.values()].sort((a, b) => {
    const unitA = a.units[0];
    const unitB = b.units[0];
    if (unitA.currentHealth !== unitB.currentHealth) return unitB.currentHealth - unitA.currentHealth;
    if (unitA.remainingMoves !== unitB.remainingMoves) return unitB.remainingMoves - unitA.remainingMoves;
    return Number(unitB.canAttack) - Number(unitA.canAttack);
  });
}

function UnitPicker({units, selected, onChange}: UnitPickerProps) {
  const groups = groupUnits(units);

  const setGroupCount = (group: UnitGroup, count: number) => {
    const next = new Set(selected);
    for (const unit of group.units) next.delete(unit);
    for (const unit of group.units.slice(0, count)) next.add(unit);
    onChange(next);
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
        const unit = group.units[0];
        const ratio = unit.currentHealth / unit.health;
        const count = group.units.filter((u) => selected.has(u)).length;
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
            <Slider size="small" value={count} min={0} max={group.units.length}
              onChange={(_, v) => setGroupCount(group, v as number)}/>
            <Button size="small" variant="text" sx={{minWidth: 24, p: 0}} onClick={() => setGroupCount(group, Math.max(0, count - 1))}>-</Button>
            <Typography variant="body2" sx={{textAlign: "center", fontVariantNumeric: "tabular-nums"}}>{count}/{group.units.length}</Typography>
            <Button size="small" variant="text" sx={{minWidth: 24, p: 0}} onClick={() => setGroupCount(group, Math.min(group.units.length, count + 1))}>+</Button>
          </Box>
        );
      })}
      <Box sx={{display: "flex", alignItems: "center", justifyContent: "space-between"}}>
        <Typography variant="caption" sx={{fontVariantNumeric: "tabular-nums"}}>Selected {selected.size}/{units.length}</Typography>
        <ButtonGroup size="small" variant="outlined">
          <Button sx={{textTransform: "none", width: 56, py: 0}} onClick={() => onChange(new Set(units))}>All</Button>
          <Button sx={{textTransform: "none", width: 56, py: 0}} onClick={() => onChange(new Set())}>None</Button>
        </ButtonGroup>
      </Box>
    </Box>
  );
}

export default UnitPicker;
