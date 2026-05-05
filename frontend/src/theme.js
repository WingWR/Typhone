import { createTheme } from "@mui/material/styles";

const theme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#f6df58",
    },
    secondary: {
      main: "#47d4ff",
    },
    background: {
      default: "#060b14",
      paper: "rgba(7, 13, 24, 0.82)",
    },
    text: {
      primary: "#ecf3ff",
      secondary: "rgba(211, 223, 242, 0.78)",
    },
  },
  shape: {
    borderRadius: 18,
  },
  typography: {
    fontFamily: "'IBM Plex Sans', 'Segoe UI', sans-serif",
    h5: {
      fontWeight: 700,
      letterSpacing: "-0.02em",
    },
    button: {
      textTransform: "none",
      fontWeight: 600,
    },
  },
});

export default theme;
