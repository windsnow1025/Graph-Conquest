import {createRoot} from "react-dom/client";
import {ThemeProvider, createTheme} from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import "./index.css";
import App from "./App.tsx";

const theme = createTheme({
  colorSchemes: {light: true, dark: true},
});

createRoot(document.getElementById("root")!).render(
  <ThemeProvider theme={theme}>
    <CssBaseline/>
    <App/>
  </ThemeProvider>,
);
