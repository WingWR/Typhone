import { Box, Stack, Typography } from "@mui/material";
import ForecastSidebar from "./components/ForecastSidebar";
import MapLegend from "./components/MapLegend";
import MapScene from "./components/MapScene";
import TimelineBar from "./components/TimelineBar";
import { WEATHER_STYLES } from "./constants/map";
import { useTyphoonVisualizer } from "./hooks/useTyphoonVisualizer";

function App() {
  const {
    prediction,
    weatherField,
    setWeatherField,
    weatherPoints,
    weatherMeta,
    weatherSummary,
    loadingPrediction,
    error,
    currentTime,
    setCurrentTime,
    manualControl,
    setManualControl,
    timeRange,
    activeTrackPoint,
    activeSourceName,
    loadSampleForecast,
    uploadTyphoonJson,
  } = useTyphoonVisualizer();

  return (
    <Box sx={{ minHeight: "100vh", position: "relative", overflow: "hidden" }}>
      <Box
        sx={{
          position: "absolute",
          inset: 0,
          background:
            "radial-gradient(circle at top left, rgba(35, 94, 150, 0.22), transparent 28%), radial-gradient(circle at bottom right, rgba(4, 197, 255, 0.14), transparent 24%), linear-gradient(180deg, #07101c 0%, #04070e 100%)",
        }}
      />
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
          border: "1px solid rgba(115, 143, 184, 0.14)",
          background: "rgba(4, 10, 19, 0.72)",
          backdropFilter: "blur(14px)",
        }}
      >
        <Box>
          <Typography variant="h5" sx={{ color: "#f9fbff", fontWeight: 700 }}>
            TyphoonAI <Box component="span" sx={{ color: "#1ca2ff" }}>|</Box> Shanghai Forecast
          </Typography>
          <Typography variant="body2" sx={{ color: "rgba(217, 230, 249, 0.64)", mt: 0.35 }}>
            Upload observations and replay typhoon motion continuously on the map
          </Typography>
        </Box>
        <Stack direction="row" spacing={2.5} sx={{ display: { xs: "none", md: "flex" }, color: "rgba(214, 224, 241, 0.72)" }}>
          <Typography variant="body2">Model: PINN-v1</Typography>
          <Typography variant="body2">Region: East China</Typography>
        </Stack>
      </Stack>
      <MapScene
        activeTrackPoint={activeTrackPoint}
        actualTrack={prediction?.actualTrack ?? []}
        baselineTrack={prediction?.baselineTrack ?? []}
        combinedTrack={prediction?.combinedTrack ?? []}
        currentTime={currentTime}
        observedTrack={prediction?.observedTrack ?? []}
        weatherField={weatherField}
        weatherPoints={weatherPoints}
        weatherUnit={WEATHER_STYLES[weatherField].unit}
      />
      <ForecastSidebar
        activeSourceName={activeSourceName}
        activeTrackPoint={activeTrackPoint}
        error={error}
        loadingPrediction={loadingPrediction}
        onLoadSample={loadSampleForecast}
        onUploadFile={uploadTyphoonJson}
        prediction={prediction}
        setWeatherField={setWeatherField}
        weatherField={weatherField}
        weatherMeta={weatherMeta}
        weatherSummary={weatherSummary}
      />
      <MapLegend />
      <TimelineBar
        currentTime={currentTime}
        manualControl={manualControl}
        setCurrentTime={setCurrentTime}
        setManualControl={setManualControl}
        timeRange={timeRange}
        track={prediction?.combinedTrack ?? []}
      />
    </Box>
  );
}

export default App;
