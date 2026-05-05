import {
  Alert,
  Box,
  Chip,
  Paper,
  Stack,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from "@mui/material";
import { WEATHER_STYLES } from "../constants/map";
import UploadPanel from "./UploadPanel";

function formatMetric(value, digits = 3, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  return `${Number(value).toFixed(digits)}${suffix}`;
}

function ForecastSidebar({
  activeSourceName,
  activeTrackPoint,
  error,
  loadingPrediction,
  onLoadSample,
  onUploadFile,
  prediction,
  setWeatherField,
  weatherField,
  weatherMeta,
  weatherSummary,
}) {
  const activeStyle = WEATHER_STYLES[weatherField];
  const losses = prediction?.losses ?? {};
  const metrics = prediction?.metrics ?? {};
  const summary = prediction?.summary ?? {};

  return (
    <Paper
      elevation={0}
      sx={{
        position: "absolute",
        top: { xs: 86, md: 58 },
        left: { xs: 16, md: 24 },
        width: { xs: "calc(100% - 32px)", md: 344 },
        maxHeight: { xs: "calc(100vh - 180px)", md: "calc(100vh - 108px)" },
        overflow: "auto",
        p: 1.8,
        borderRadius: 3,
        border: "1px solid rgba(108, 136, 181, 0.18)",
        background: "linear-gradient(180deg, rgba(8, 15, 27, 0.92), rgba(6, 12, 22, 0.98))",
        backdropFilter: "blur(18px)",
        zIndex: 14,
      }}
    >
      <Stack spacing={2}>
        <UploadPanel
          activeSourceName={activeSourceName}
          onLoadSample={onLoadSample}
          onUploadFile={onUploadFile}
          loadingPrediction={loadingPrediction}
        />

        {error ? <Alert severity="error">{error}</Alert> : null}

        <Stack spacing={1.2}>
          <Typography variant="overline" sx={{ color: "rgba(173, 191, 223, 0.78)", letterSpacing: "0.16em" }}>
            Storm Overview
          </Typography>
          <Stack direction="row" justifyContent="space-between" alignItems="flex-start">
            <Box>
              <Typography variant="h4" sx={{ fontWeight: 700, color: "#f0c648", lineHeight: 1.05 }}>
                {prediction?.stormName ?? "Typhoon"}
              </Typography>
              <Typography variant="body2" sx={{ color: "rgba(213, 225, 244, 0.62)", mt: 0.5 }}>
                ID: {prediction?.stormId ?? "--"}
              </Typography>
            </Box>
            <Chip
              label={`${prediction?.forecastSteps ?? "--"} steps`}
              sx={{
                backgroundColor: "rgba(240, 198, 72, 0.12)",
                color: "#f3cf62",
                border: "1px solid rgba(240, 198, 72, 0.24)",
              }}
            />
          </Stack>
        </Stack>

        <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
          <Chip label={`Obs ${prediction?.observedTrack.length ?? 0}`} variant="outlined" />
          <Chip label={`Pred ${prediction?.predictedTrack.length ?? 0}`} variant="outlined" />
          <Chip label={`Base ${prediction?.baselineTrack.length ?? 0}`} variant="outlined" />
          <Chip label={`Truth ${prediction?.actualTrack.length ?? 0}`} variant="outlined" />
          <Chip label={`Wind ${prediction?.summary?.max_wind_speed ?? "--"} m/s`} variant="outlined" />
          <Chip label={`Pressure ${prediction?.summary?.min_pressure ?? "--"} hPa`} variant="outlined" />
        </Stack>

        <Stack spacing={1}>
          <Typography variant="overline" sx={{ color: "rgba(173, 191, 223, 0.78)", letterSpacing: "0.16em" }}>
            Model Diagnostics
          </Typography>
          <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
            <Chip label={`Model ${summary.model_name ?? prediction?.modelName ?? "--"}`} variant="outlined" />
            <Chip label={`Mode ${summary.inference_mode ?? "--"}`} variant="outlined" />
            <Chip label={`Phys Score ${formatMetric(summary.physics_consistency_score, 3)}`} variant="outlined" />
            <Chip label={`Data Loss ${formatMetric(losses.data_loss, 3)}`} variant="outlined" />
            <Chip label={`Physics Loss ${formatMetric(losses.physics_loss, 3)}`} variant="outlined" />
            <Chip label={`Track MAE ${formatMetric(metrics.track_mae_km, 2, " km")}`} variant="outlined" />
            <Chip label={`Final Err ${formatMetric(metrics.final_position_error_km, 2, " km")}`} variant="outlined" />
            <Chip label={`Base MAE ${formatMetric(metrics.baseline_track_mae_km, 2, " km")}`} variant="outlined" />
            <Chip label={`PINN-Base ${formatMetric(metrics.baseline_vs_pinn_mean_km, 2, " km")}`} variant="outlined" />
            <Chip label={`Wind MAE ${formatMetric(metrics.wind_mae_mps, 2, " m/s")}`} variant="outlined" />
            <Chip label={`Pressure MAE ${formatMetric(metrics.pressure_mae_hpa, 2, " hPa")}`} variant="outlined" />
          </Stack>
        </Stack>

        <Box>
          <Typography variant="overline" sx={{ color: activeStyle.accent, letterSpacing: "0.16em" }}>
            Layers
          </Typography>
          <ToggleButtonGroup
            color="primary"
            exclusive
            fullWidth
            value={weatherField}
            onChange={(_, value) => {
              if (value) {
                setWeatherField(value);
              }
            }}
            sx={{
              mt: 1,
              "& .MuiToggleButton-root": {
                borderColor: "rgba(115, 152, 209, 0.14)",
                color: "rgba(224, 232, 247, 0.72)",
                py: 1,
                backgroundColor: "rgba(13, 21, 37, 0.78)",
              },
              "& .Mui-selected": {
                color: "#04111c",
                backgroundColor: "#2aa7ff !important",
              },
            }}
          >
            <ToggleButton value="rain">Rain</ToggleButton>
            <ToggleButton value="wind">Wind</ToggleButton>
            <ToggleButton value="pressure">Pressure</ToggleButton>
          </ToggleButtonGroup>
        </Box>

        <Box
          sx={{
            borderRadius: 2.5,
            p: 1.5,
            border: "1px solid rgba(126, 168, 226, 0.12)",
            background: "rgba(10, 18, 31, 0.84)",
          }}
        >
          <Typography variant="body2" sx={{ color: "#edf4ff", fontWeight: 600 }}>
            {activeTrackPoint ? `Active Time: ${new Date(activeTrackPoint.timestamp).toLocaleString()}` : "Waiting for track"}
          </Typography>
          <Typography variant="body2" sx={{ color: "rgba(216, 229, 248, 0.68)", mt: 0.8, lineHeight: 1.65 }}>
            {weatherMeta?.description ?? "Waiting for weather field response"}
          </Typography>
          <Typography variant="caption" sx={{ display: "block", mt: 0.8, color: "rgba(216, 229, 248, 0.54)" }}>
            Domain 120.5E-122.5E / 30.5N-32.0N | Grid{" "}
            {weatherMeta ? `${weatherMeta.grid_shape.lat} x ${weatherMeta.grid_shape.lng}` : "--"}
          </Typography>
        </Box>

        <Stack spacing={1}>
          <Typography variant="overline" sx={{ color: "rgba(173, 191, 223, 0.78)", letterSpacing: "0.16em" }}>
            Weather Metrics
          </Typography>
          <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
            <Chip label={`${activeStyle.label} Min ${weatherSummary.min} ${activeStyle.unit}`} variant="outlined" />
            <Chip label={`${activeStyle.label} Mean ${weatherSummary.mean} ${activeStyle.unit}`} variant="outlined" />
            <Chip label={`${activeStyle.label} Max ${weatherSummary.max} ${activeStyle.unit}`} variant="outlined" />
          </Stack>
        </Stack>
      </Stack>
    </Paper>
  );
}

export default ForecastSidebar;
