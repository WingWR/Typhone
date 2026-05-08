import { createTheme } from "@mui/material/styles";

const theme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: "#1565c0",
    },
    secondary: {
      main: "#0288d1",
    },
    background: {
      default: "#eef1f5",
      paper: "rgba(255, 255, 255, 0.92)",
    },
    text: {
      primary: "#152433",
      secondary: "rgba(30, 50, 70, 0.7)",
    },
  },
  shape: {
    borderRadius: 14,
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
