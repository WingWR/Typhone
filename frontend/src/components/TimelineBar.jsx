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
  const sliderStep = (timeRange[1] - timeRange[0]) / Math.max(track.length - 1, 1);

  return (
    <Paper
      elevation={0}
      sx={{
        position: "absolute",
        left: { xs: 16, md: 392 },
        right: { xs: 16, md: 24 },
        bottom: { xs: 16, md: 22 },
        p: { xs: 1.5, md: 1.9 },
        borderRadius: 3,
        border: "1px solid rgba(112, 139, 182, 0.16)",
        background: "rgba(6, 12, 23, 0.88)",
        backdropFilter: "blur(18px)",
        zIndex: 14,
      }}
    >
      <Stack spacing={1.2}>
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Typography variant="overline" sx={{ color: "#8dbdf8", letterSpacing: "0.16em" }}>
            Forecast Timeline (UTC)
          </Typography>
          <Stack direction="row" spacing={1} alignItems="center">
            <Button
              variant="outlined"
              size="small"
              onClick={() => setManualControl((value) => !value)}
              sx={{
                minWidth: 90,
                color: "#dbe8fb",
                borderColor: "rgba(130, 159, 206, 0.22)",
              }}
            >
              {manualControl ? "Play" : "Pause"}
            </Button>
            <Typography variant="caption" sx={{ color: "rgba(219, 229, 245, 0.7)", minWidth: 128 }}>
              {track.length ? new Date(currentTime * 1000).toLocaleString() : "--"}
            </Typography>
          </Stack>
        </Stack>

        <Slider
          min={timeRange[0]}
          max={timeRange[1]}
          step={sliderStep || 1}
          marks={marks}
          value={Math.min(Math.max(currentTime, timeRange[0]), timeRange[1])}
          onChange={(_, value) => {
            setManualControl(true);
            setCurrentTime(Number(value));
          }}
          sx={{
            color: "#28a7ff",
            "& .MuiSlider-thumb": {
              width: 18,
              height: 18,
              backgroundColor: "#0fa2ff",
              boxShadow: "0 0 0 6px rgba(15, 162, 255, 0.14)",
            },
            "& .MuiSlider-track": {
              border: "none",
              height: 6,
            },
            "& .MuiSlider-rail": {
              opacity: 0.35,
              height: 6,
              backgroundColor: "#557199",
            },
            "& .MuiSlider-mark": {
              display: "none",
            },
            "& .MuiSlider-markLabel": {
              top: 40,
              color: "rgba(212, 224, 245, 0.58)",
              fontSize: "0.7rem",
            },
          }}
        />
      </Stack>
    </Paper>
  );
}

export default TimelineBar;
