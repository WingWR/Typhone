import { Button, Paper, Slider, Stack, Typography } from "@mui/material";

function buildTimelineMarks(track) {
  if (!track.length) {
    return [];
  }

  const stride = Math.max(1, Math.floor(track.length / 6));
  return track
    .filter((_, index) => index % stride === 0 || index === track.length - 1)
    .map((point) => ({
      value: point.timeValue,
      label: new Date(point.timestamp).toLocaleString([], {
        month: "numeric",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      }),
    }));
}

function TimelineBar({ currentTime, manualControl, setCurrentTime, setManualControl, timeRange, track }) {
  const marks = buildTimelineMarks(track);

  return (
    <Paper
      elevation={0}
      sx={{
        position: "absolute",
        left: "50%",
        transform: "translateX(-50%)",
        width: { xs: "calc(100% - 32px)", md: "50%", lg: "44%" },
        bottom: { xs: 16, md: 22 },
        p: { xs: 1.5, md: 1.9 },
        borderRadius: 3,
        border: "1px solid rgba(0,0,0,0.08)",
        background: "rgba(255,255,255,0.92)",
        backdropFilter: "blur(18px)",
        boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
        zIndex: 14,
      }}
    >
      <Stack spacing={1.2}>
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Typography variant="overline" sx={{ color: "#1565c0", letterSpacing: "0.14em", fontWeight: 600 }}>
            Forecast Timeline (UTC)
          </Typography>
          <Stack direction="row" spacing={1} alignItems="center">
            <Button
              variant="outlined"
              size="small"
              onClick={() => setManualControl((value) => !value)}
              sx={{ minWidth: 90 }}
            >
              {manualControl ? "Play" : "Pause"}
            </Button>
            <Typography variant="caption" sx={{ color: "rgba(30,50,70,0.55)", minWidth: 128 }}>
              {track.length ? new Date(currentTime * 1000).toLocaleString() : "--"}
            </Typography>
          </Stack>
        </Stack>

        <Slider
          min={timeRange[0]}
          max={timeRange[1]}
          marks={marks}
          value={Math.min(Math.max(currentTime, timeRange[0]), timeRange[1])}
          onChange={(_, value) => {
            setManualControl(true);
            setCurrentTime(Number(value));
          }}
          sx={{
            color: "#1565c0",
            "& .MuiSlider-thumb": {
              width: 18,
              height: 18,
              backgroundColor: "#1565c0",
              boxShadow: "0 0 0 6px rgba(21, 101, 192, 0.12)",
            },
            "& .MuiSlider-track": {
              border: "none",
              height: 6,
            },
            "& .MuiSlider-rail": {
              opacity: 0.2,
              height: 6,
              backgroundColor: "#8a9fb0",
            },
            "& .MuiSlider-mark": {
              display: "none",
            },
            "& .MuiSlider-markLabel": {
              top: 40,
              color: "rgba(30,50,70,0.5)",
              fontSize: "0.7rem",
            },
          }}
        />
      </Stack>
    </Paper>
  );
}

export default TimelineBar;
