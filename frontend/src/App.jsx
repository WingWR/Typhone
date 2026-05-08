import { useEffect, useState } from "react";
import { Alert, Box, Snackbar, Stack, Typography } from "@mui/material";
import ForecastSidebar from "./components/ForecastSidebar";
import MapLegend from "./components/MapLegend";
import MapScene from "./components/MapScene";
import TimelineBar from "./components/TimelineBar";
import { useTyphoonVisualizer } from "./hooks/useTyphoonVisualizer";

function App() {
  const {
    prediction,
    loadingPrediction,
    error,
    currentTime,
    setCurrentTime,
    manualControl,
    setManualControl,
    timeRange,
    activeTrackPoint,
    activeSourceName,
    hasPrediction,
    uploadTyphoonJson,
  } = useTyphoonVisualizer();
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState("");

  useEffect(() => {
    if (error) {
      setSnackbarMessage(error);
      setSnackbarOpen(true);
    }
  }, [error]);

  return (
    <Box sx={{ minHeight: "100vh", position: "relative", overflow: "hidden" }}>
      <Box
        sx={{
          position: "absolute",
          inset: 0,
          background: "linear-gradient(180deg, #e8edf2 0%, #d5dce3 100%)",
        }}
      />

      {/* Header */}
      <Stack
        direction="row"
        justifyContent="space-between"
        alignItems={{ xs: "flex-start", md: "center" }}
        spacing={1}
        sx={{
          position: "absolute",
          zIndex: 18,
          left: { xs: 16, md: 24 },
          right: { xs: 16, md: 24 },
          top: { xs: 12, md: 12 },
          px: { xs: 0.5, md: 1.5 },
          py: 1.5,
          borderRadius: 3,
          border: "1px solid rgba(0,0,0,0.08)",
          background: "rgba(255,255,255,0.9)",
          backdropFilter: "blur(14px)",
          boxShadow: "0 2px 16px rgba(0,0,0,0.05)",
        }}
      >
        <Box>
          <Typography variant="h5" sx={{ color: "#152433", fontWeight: 700 }}>
            TyphoonAI <Box component="span" sx={{ color: "#1565c0" }}>|</Box> Shanghai Forecast
          </Typography>
          <Typography variant="body2" sx={{ color: "rgba(30,50,70,0.55)", mt: 0.35 }}>
            Upload observations and replay typhoon motion continuously on the map
          </Typography>
        </Box>
        <Stack direction="row" spacing={2.5} sx={{ display: { xs: "none", md: "flex" }, color: "rgba(30,50,70,0.55)" }}>
          <Typography variant="body2">Model: TyPhoonPINN</Typography>
          <Typography variant="body2">Region: East China</Typography>
        </Stack>
      </Stack>

      {/* Map */}
      <MapScene
        activeTrackPoint={activeTrackPoint}
        actualTrack={prediction?.actualTrack ?? []}
        baselineTrack={prediction?.baselineTrack ?? []}
        combinedTrack={prediction?.combinedTrack ?? []}
        currentTime={currentTime}
        observedTrack={prediction?.observedTrack ?? []}
      />

      {/* Sidebar */}
      <ForecastSidebar
        activeSourceName={activeSourceName}
        activeTrackPoint={activeTrackPoint}
        error={error}
        loadingPrediction={loadingPrediction}
        onUploadFile={uploadTyphoonJson}
        prediction={prediction}
      />

      {/* Legend */}
      {hasPrediction ? <MapLegend activeTrackPoint={activeTrackPoint} prediction={prediction} /> : null}

      {/* Timeline */}
      {hasPrediction ? (
        <TimelineBar
          currentTime={currentTime}
          manualControl={manualControl}
          setCurrentTime={setCurrentTime}
          setManualControl={setManualControl}
          timeRange={timeRange}
          track={prediction?.combinedTrack ?? []}
        />
      ) : null}

      {/* Error snackbar */}
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={5000}
        onClose={() => setSnackbarOpen(false)}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
        sx={{ bottom: { xs: 120, md: 100 } }}
      >
        <Alert
          severity="error"
          variant="filled"
          onClose={() => setSnackbarOpen(false)}
          sx={{ borderRadius: 2 }}
        >
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </Box>
  );
}

export default App;
