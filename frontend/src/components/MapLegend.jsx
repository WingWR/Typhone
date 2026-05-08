import { useEffect, useRef, useState } from "react";
import { Box, keyframes, Paper, Stack, Typography } from "@mui/material";

const TRACK_LEGENDS = [
  { label: "PINN", color: "#e6a800" },
  { label: "Baseline", color: "#e0551f" },
  { label: "Actual", color: "#1e9e4b" },
];

const SOURCE_LABELS = {
  observed: "Observed",
  forecast: "PINN Forecast",
  baseline: "Baseline",
  actual: "Actual",
};

const METRICS = [
  { key: "wind_speed", label: "Wind", unit: "m/s", min: 8, max: 75, color: "#d47400", colorLight: "rgba(212,116,0,0.15)" },
  { key: "pressure", label: "Pressure", unit: "hPa", min: 880, max: 1020, color: "#1565c0", colorLight: "rgba(21,101,192,0.15)" },
];

const bounce = keyframes`
  0%, 100% { transform: scaleY(1); }
  50% { transform: scaleY(1.12); }
`;

const pulse = keyframes`
  0% { opacity: 0; transform: scale(0.6); }
  40% { opacity: 0.7; }
  100% { opacity: 0; transform: scale(2.2); }
`;

function formatValue(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toFixed(digits);
}

function clampRatio(value, min, max) {
  if (value === null || value === undefined) return 0;
  return Math.min(Math.max((Number(value) - min) / (max - min), 0), 1);
}

function findPointByTime(track, timeValue) {
  if (!track || !track.length) return null;
  let best = track[0];
  let bestDist = Math.abs(track[0].timeValue - timeValue);
  for (const point of track) {
    const dist = Math.abs(point.timeValue - timeValue);
    if (dist < bestDist) {
      bestDist = dist;
      best = point;
    }
  }
  return bestDist < 3600 ? best : null;
}

function MetricBar({ label, activeValue, actualValue, unit, min, max, color, colorLight }) {
  const prevRef = useRef(activeValue);
  const [animKey, setAnimKey] = useState(0);
  const [ripple, setRipple] = useState(false);

  useEffect(() => {
    if (activeValue !== null && activeValue !== undefined && activeValue !== prevRef.current) {
      setAnimKey((k) => k + 1);
      setRipple(true);
      const timer = setTimeout(() => setRipple(false), 500);
      prevRef.current = activeValue;
      return () => clearTimeout(timer);
    }
  }, [activeValue]);

  const activeRatio = clampRatio(activeValue, min, max);
  const actualRatio = clampRatio(actualValue, min, max);

  return (
    <Box sx={{ position: "relative" }}>
      <Stack direction="row" justifyContent="space-between" alignItems="baseline" sx={{ mb: 0.25 }}>
        <Typography variant="caption" sx={{ color: "rgba(30,50,70,0.65)", fontSize: "0.6rem", fontWeight: 600, letterSpacing: "0.04em" }}>
          {label}
        </Typography>
        <Stack direction="row" spacing={1} alignItems="baseline">
          <Typography variant="caption" sx={{ color, fontSize: "0.68rem", fontWeight: 700 }}>
            {formatValue(activeValue)} <Box component="span" sx={{ fontSize: "0.55rem", fontWeight: 400, opacity: 0.6 }}>{unit}</Box>
          </Typography>
          {actualValue !== null && actualValue !== undefined ? (
            <Typography variant="caption" sx={{ color: "#1e9e4b", fontSize: "0.62rem", fontWeight: 600 }}>
              / {formatValue(actualValue)}
            </Typography>
          ) : null}
        </Stack>
      </Stack>

      <Box
        sx={{
          height: 10,
          borderRadius: 999,
          background: "rgba(0,0,0,0.05)",
          overflow: "hidden",
          position: "relative",
        }}
      >
        {/* Actual marker line */}
        {actualValue !== null && actualValue !== undefined ? (
          <Box
            sx={{
              position: "absolute",
              left: `${actualRatio * 100}%`,
              top: -2,
              bottom: -2,
              width: 3,
              borderRadius: 2,
              backgroundColor: "#1e9e4b",
              zIndex: 3,
              transform: "translateX(-50%)",
              opacity: 0.8,
            }}
          />
        ) : null}

        {/* Active value bar */}
        <Box
          key={animKey}
          sx={{
            height: "100%",
            width: `${activeRatio * 100}%`,
            borderRadius: 999,
            background: `linear-gradient(90deg, ${colorLight} 0%, ${color} 100%)`,
            position: "relative",
            zIndex: 2,
            transformOrigin: "bottom",
            animation: `${bounce} 0.4s ease`,
            transition: "width 0.35s ease",
          }}
        >
          {/* Ripple effect on change */}
          {ripple ? (
            <Box
              sx={{
                position: "absolute",
                right: -6,
                top: "50%",
                width: 24,
                height: 24,
                borderRadius: "50%",
                backgroundColor: color,
                transform: "translate(0, -50%)",
                animation: `${pulse} 0.5s ease-out`,
                pointerEvents: "none",
              }}
            />
          ) : null}
        </Box>
      </Box>

      {/* Min/Max labels */}
      <Stack direction="row" justifyContent="space-between" sx={{ mt: 0.15 }}>
        <Typography variant="caption" sx={{ color: "rgba(30,50,70,0.35)", fontSize: "0.5rem" }}>
          {min}
        </Typography>
        <Typography variant="caption" sx={{ color: "rgba(30,50,70,0.35)", fontSize: "0.5rem" }}>
          {max}
        </Typography>
      </Stack>
    </Box>
  );
}

function MapLegend({ activeTrackPoint, prediction }) {
  const border = "1px solid rgba(0,0,0,0.08)";
  const textMuted = "rgba(30,50,70,0.55)";
  const textBody = "rgba(30,50,70,0.8)";

  const currentTime = activeTrackPoint?.timeValue ?? 0;
  const actualPoint = findPointByTime(prediction?.actualTrack, currentTime);
  const baselinePoint = findPointByTime(prediction?.baselineTrack, currentTime);

  return (
    <Paper
      elevation={0}
      sx={{
        position: "absolute",
        right: { xs: 16, md: 24 },
        top: { xs: 100, md: 100 },
        width: { xs: "calc(100% - 32px)", md: 270 },
        p: 1.75,
        borderRadius: 3,
        border: border,
        background: "rgba(255,255,255,0.92)",
        backdropFilter: "blur(16px)",
        boxShadow: "0 4px 24px rgba(0,0,0,0.06)",
        zIndex: 12,
        maxHeight: { xs: "calc(100vh - 280px)", md: "calc(100vh - 140px)" },
        overflow: "auto",
        "&::-webkit-scrollbar": { width: 4 },
        "&::-webkit-scrollbar-thumb": { backgroundColor: "rgba(0,0,0,0.1)", borderRadius: 4 },
      }}
    >
      <Stack spacing={1.8}>
        {/* Active point */}
        <Stack spacing={0.5}>
          <Typography variant="caption" sx={{ color: "#1565c0", letterSpacing: "0.08em", fontSize: "0.6rem", fontWeight: 600 }}>
            ACTIVE POINT
          </Typography>
          <Box
            sx={{
              borderRadius: 1.5,
              p: 1.2,
              border: border,
              background: "rgba(240, 244, 250, 0.5)",
            }}
          >
            <Stack spacing={0.35}>
              <Typography variant="caption" sx={{ color: "#d47400", fontSize: "0.66rem", fontWeight: 600 }}>
                {activeTrackPoint ? SOURCE_LABELS[activeTrackPoint.source] ?? activeTrackPoint.source : "--"}
              </Typography>
              <Stack direction="row" spacing={2}>
                <Typography variant="caption" sx={{ color: textBody, fontSize: "0.63rem" }}>
                  Lng {formatValue(activeTrackPoint?.lng, 4)}
                </Typography>
                <Typography variant="caption" sx={{ color: textBody, fontSize: "0.63rem" }}>
                  Lat {formatValue(activeTrackPoint?.lat, 4)}
                </Typography>
              </Stack>
              <Typography variant="caption" sx={{ color: textMuted, fontSize: "0.58rem" }}>
                {activeTrackPoint?.timestamp
                  ? new Date(activeTrackPoint.timestamp).toLocaleString()
                  : "--"}
              </Typography>
            </Stack>
          </Box>
        </Stack>

        {/* Animated data bars */}
        <Stack spacing={1}>
          <Typography variant="caption" sx={{ color: textBody, letterSpacing: "0.06em", fontWeight: 600, fontSize: "0.62rem" }}>
            LIVE METRICS
          </Typography>
          {METRICS.map((metric) => (
            <MetricBar
              key={metric.key}
              label={metric.label}
              activeValue={activeTrackPoint?.[metric.key]}
              actualValue={actualPoint?.[metric.key]}
              unit={metric.unit}
              min={metric.min}
              max={metric.max}
              color={metric.color}
              colorLight={metric.colorLight}
            />
          ))}
        </Stack>

        {/* Track legend */}
        <Stack spacing={0.6}>
          <Typography variant="caption" sx={{ color: textBody, letterSpacing: "0.06em", fontWeight: 600, fontSize: "0.62rem" }}>
            TRACKS
          </Typography>
          <Stack direction="row" spacing={1.4} useFlexGap flexWrap="wrap">
            {TRACK_LEGENDS.map((item) => (
              <Stack key={item.label} direction="row" spacing={0.7} alignItems="center">
                <Box sx={{ width: 22, height: 3, borderRadius: 999, backgroundColor: item.color }} />
                <Typography variant="caption" sx={{ color: textMuted }}>
                  {item.label}
                </Typography>
              </Stack>
            ))}
          </Stack>
        </Stack>
      </Stack>
    </Paper>
  );
}

export default MapLegend;
