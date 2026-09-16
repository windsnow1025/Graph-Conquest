import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import Paper from "@mui/material/Paper";

export type GameMode = "menu" | "local" | "ai" | "greedy" | "autoplay";

interface ModeSelectProps {
  onSelect: (mode: Exclude<GameMode, "menu">) => void;
}

function ModeSelect({onSelect}: ModeSelectProps) {
  return (
    <Box className="local-scroll-root" sx={{justifyContent: "center", alignItems: "center"}}>
      <Paper elevation={3} sx={{p: 4, display: "flex", flexDirection: "column", gap: 3, minWidth: 300}}>
        <Typography variant="h4" align="center">Graph Conquest</Typography>
        <Button variant="contained" size="large" sx={{textTransform: "none"}} onClick={() => onSelect("local")}>
          Local Multiplayer
        </Button>
        <Button variant="contained" size="large" color="secondary" sx={{textTransform: "none"}} onClick={() => onSelect("ai")}>
          vs AI (NN)
        </Button>
        <Button variant="contained" size="large" color="warning" sx={{textTransform: "none"}} onClick={() => onSelect("greedy")}>
          vs Greedy
        </Button>
        <Button variant="contained" size="large" color="success" sx={{textTransform: "none"}} onClick={() => onSelect("autoplay")}>
          AI Autoplay
        </Button>
      </Paper>
    </Box>
  );
}

export default ModeSelect;
