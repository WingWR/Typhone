import { useRef, useState } from "react";
import {
  Box,
  Button,
  Chip,
  IconButton,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";

const PANEL_WIDTH = 340;

function ForecastSidebar({
  activeSourceName,
  activeTrackPoint,
  error,
  loadingPrediction,
  onUploadFile,
  prediction,
}) {
  const [isOpen, setIsOpen] = useState(true);
  const fileInputRef = useRef(null);
  const observedTrack = prediction?.observedTrack ?? [];
  const stormName = prediction?.stormName;
  const stormId = prediction?.stormId;
  const forecastSteps = prediction?.forecastSteps;

  const panelBg = "rgba(255,255,255,0.92)";
  const border = "1px solid rgba(0,0,0,0.08)";
  const textMuted = "rgba(30,50,70,0.55)";

  const toggleButton = (
    <Paper
      elevation={0}
      sx={{
        position: "absolute",
        top: 180,
        left: isOpen ? 24 + PANEL_WIDTH : 0,
        zIndex: 15,
        width: 28,
        height: 56,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        borderRadius: "0 10px 10px 0",
        border: border,
        borderLeft: "none",
        background: panelBg,
        backdropFilter: "blur(14px)",
        cursor: "pointer",
        transition: "left 0.3s ease",
        boxShadow: "0 2px 12px rgba(0,0,0,0.06)",
      }}
      onClick={() => setIsOpen((v) => !v)}
    >
      <Typography sx={{ color: "#1565c0", fontSize: "1rem", userSelect: "none", lineHeight: 1 }}>
        {isOpen ? "⟨" : "⟩"}
      </Typography>
    </Paper>
  );

  const emptyState = (
    <Stack spacing={2.5} alignItems="center" sx={{ py: 4, px: 1 }}>
      <Typography variant="h6" sx={{ color: "#152433", fontWeight: 600, textAlign: "center" }}>
        Welcome to TyphoonAI
      </Typography>
      <Typography variant="body2" sx={{ color: textMuted, textAlign: "center", lineHeight: 1.8 }}>
        Upload typhoon observation data to start forecasting.
      </Typography>
      <Button
        variant="contained"
        onClick={() => fileInputRef.current?.click()}
        disabled={loadingPrediction}
        sx={{ px: 3, mt: 1 }}
      >
        Upload JSON
      </Button>
      <input
        ref={fileInputRef}
        type="file"
        accept=".json,application/json"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (!file) return;
          if (!file.name.toLowerCase().endsWith(".json")) return;
          onUploadFile(file);
          event.target.value = "";
        }}
      />
    </Stack>
  );

  const dataState = (
    <Stack spacing={1.8}>
      {/* Upload row */}
      <Stack spacing={1}>
        <Button
          variant="contained"
          size="small"
          onClick={() => fileInputRef.current?.click()}
          disabled={loadingPrediction}
          sx={{ fontSize: "0.76rem" }}
        >
          Upload JSON
        </Button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".json,application/json"
          hidden
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (!file) return;
            if (!file.name.toLowerCase().endsWith(".json")) return;
            onUploadFile(file);
            event.target.value = "";
          }}
        />
        <Typography variant="caption" sx={{ color: textMuted, fontSize: "0.65rem" }}>
          Source: {activeSourceName}
        </Typography>
      </Stack>

      {/* Storm card */}
      <Paper
        elevation={0}
        sx={{
          p: 1.5,
          borderRadius: 2,
          border: border,
          background: "rgba(240, 244, 250, 0.5)",
        }}
      >
        <Stack direction="row" justifyContent="space-between" alignItems="flex-start">
          <Box>
            <Typography sx={{ fontSize: "1.3rem", fontWeight: 700, color: "#d47400", lineHeight: 1.2 }}>
              {stormName ?? "Typhoon"}
            </Typography>
            <Typography variant="caption" sx={{ color: textMuted }}>
              ID: {stormId ?? "--"}
            </Typography>
          </Box>
          <Chip
            label={`${forecastSteps ?? "--"} steps`}
            size="small"
            sx={{
              backgroundColor: "rgba(212, 116, 0, 0.08)",
              color: "#d47400",
              border: "1px solid rgba(212, 116, 0, 0.18)",
              fontSize: "0.68rem",
              fontWeight: 600,
            }}
          />
        </Stack>
        <Stack direction="row" spacing={0.8} useFlexGap flexWrap="wrap" sx={{ mt: 1.2 }}>
          <Chip label={`Obs ${observedTrack.length}`} size="small" variant="outlined" sx={{ fontSize: "0.66rem", borderColor: "rgba(0,0,0,0.12)" }} />
          <Chip label={`Pred ${prediction?.predictedTrack?.length ?? 0}`} size="small" variant="outlined" sx={{ fontSize: "0.66rem", borderColor: "rgba(0,0,0,0.12)" }} />
          <Chip label={`Wind ${prediction?.summary?.max_wind_speed ?? "--"} m/s`} size="small" variant="outlined" sx={{ fontSize: "0.66rem", borderColor: "rgba(0,0,0,0.12)" }} />
          <Chip label={`Prs ${prediction?.summary?.min_pressure ?? "--"} hPa`} size="small" variant="outlined" sx={{ fontSize: "0.66rem", borderColor: "rgba(0,0,0,0.12)" }} />
        </Stack>
      </Paper>

      {/* Observation table */}
      {observedTrack.length > 0 ? (
        <Box>
          <Typography variant="overline" sx={{ color: textMuted, letterSpacing: "0.1em", fontSize: "0.62rem" }}>
            Observations
          </Typography>
          <TableContainer
            sx={{
              mt: 0.6,
              borderRadius: 1.5,
              border: border,
              background: "rgba(250, 251, 253, 0.7)",
              maxHeight: 180,
              "&::-webkit-scrollbar": { width: 4 },
              "&::-webkit-scrollbar-thumb": { backgroundColor: "rgba(0,0,0,0.12)", borderRadius: 4 },
            }}
          >
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  {["#", "Lng", "Lat", "Wind", "Prs", "Time"].map((h) => (
                    <TableCell
                      key={h}
                      sx={{
                        py: 0.5,
                        color: textMuted,
                        fontSize: "0.62rem",
                        fontWeight: 600,
                        borderBottom: "1px solid rgba(0,0,0,0.06)",
                        background: "rgba(248, 250, 252, 0.95)",
                      }}
                    >
                      {h}
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {observedTrack.map((point, index) => {
                  const isActive = activeTrackPoint?.timestamp === point.timestamp;
                  return (
                    <TableRow
                      key={index}
                      sx={{
                        "&:hover": { background: "rgba(21, 101, 192, 0.04)" },
                        background: isActive ? "rgba(21, 101, 192, 0.08)" : "transparent",
                      }}
                    >
                      <TableCell sx={{ py: 0.35, color: textMuted, fontSize: "0.62rem", borderBottom: "none" }}>
                        {index + 1}
                      </TableCell>
                      <TableCell sx={{ py: 0.35, color: "#152433", fontSize: "0.68rem", borderBottom: "none" }}>
                        {point.lng}
                      </TableCell>
                      <TableCell sx={{ py: 0.35, color: "#152433", fontSize: "0.68rem", borderBottom: "none" }}>
                        {point.lat}
                      </TableCell>
                      <TableCell sx={{ py: 0.35, color: "#d47400", fontSize: "0.68rem", fontWeight: isActive ? 600 : 400, borderBottom: "none" }}>
                        {point.wind_speed ?? "--"}
                      </TableCell>
                      <TableCell sx={{ py: 0.35, color: "#1565c0", fontSize: "0.68rem", fontWeight: isActive ? 600 : 400, borderBottom: "none" }}>
                        {point.pressure ?? "--"}
                      </TableCell>
                      <TableCell sx={{ py: 0.35, color: textMuted, fontSize: "0.6rem", borderBottom: "none", whiteSpace: "nowrap" }}>
                        {point.timestamp ? new Date(point.timestamp).toLocaleString([], { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "--"}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
      ) : null}

      {/* Active time */}
      <Box
        sx={{
          borderRadius: 2,
          p: 1.2,
          border: border,
          background: "rgba(248, 250, 252, 0.6)",
        }}
      >
        <Typography variant="body2" sx={{ color: "#152433", fontWeight: 600, fontSize: "0.78rem" }}>
          {activeTrackPoint
            ? new Date(activeTrackPoint.timestamp).toLocaleString()
            : "Waiting for track"}
        </Typography>
        <Typography variant="caption" sx={{ display: "block", mt: 0.5, color: textMuted }}>
          {activeTrackPoint
            ? `${activeTrackPoint.source}  |  ${activeTrackPoint.lng}°E  ${activeTrackPoint.lat}°N`
            : "Drag the timeline to inspect points"}
        </Typography>
      </Box>
    </Stack>
  );

  const sidebar = (
    <Paper
      elevation={0}
      sx={{
        position: "absolute",
        top: { xs: 100, md: 100 },
        left: { xs: 16, md: 24 },
        width: { xs: "calc(100% - 32px)", md: PANEL_WIDTH },
        maxHeight: { xs: "calc(100vh - 180px)", md: "calc(100vh - 140px)" },
        overflow: "auto",
        p: 1.8,
        borderRadius: 3,
        border: border,
        background: panelBg,
        backdropFilter: "blur(18px)",
        boxShadow: "0 4px 24px rgba(0,0,0,0.06)",
        zIndex: 14,
        transition: "opacity 0.3s ease, transform 0.3s ease",
        "&::-webkit-scrollbar": { width: 4 },
        "&::-webkit-scrollbar-thumb": { backgroundColor: "rgba(0,0,0,0.1)", borderRadius: 4 },
      }}
    >
      <Stack spacing={1.5}>
        {/* Header */}
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Typography variant="overline" sx={{ color: "#1565c0", letterSpacing: "0.14em", fontSize: "0.68rem", fontWeight: 600 }}>
            Typhoon Data
          </Typography>
          <IconButton onClick={() => setIsOpen(false)} size="small" sx={{ color: textMuted, p: 0.3 }}>
            <Typography sx={{ fontSize: "0.9rem" }}>{'✕'}</Typography>
          </IconButton>
        </Stack>

        {error ? (
          <Typography variant="body2" sx={{ color: "#d32f2f", fontSize: "0.73rem", px: 0.5 }}>
            {error}
          </Typography>
        ) : null}

        {prediction ? dataState : emptyState}
      </Stack>
    </Paper>
  );

  return (
    <>
      {isOpen ? sidebar : toggleButton}
      {!isOpen ? toggleButton : null}
    </>
  );
}

export default ForecastSidebar;
