import { Box, Paper, Stack, Typography } from "@mui/material";

const LEGENDS = [
  {
    title: "RAINFALL (mm/h)",
    gradient: "linear-gradient(90deg, #203f92 0%, #1680d0 35%, #2bd6ff 72%, #aef8ff 100%)",
    labels: ["0.1", "1", "5", "10", "25", "50", "100+"],
  },
  {
    title: "WIND (m/s)",
    gradient: "linear-gradient(90deg, #1ca3a7 0%, #50d0ff 28%, #f2d44e 70%, #e63f5c 100%)",
    labels: ["0", "10", "20", "30", "40", "50+"],
  },
  {
    title: "PRESSURE (hPa)",
    gradient: "linear-gradient(90deg, #557cff 0%, #50c5ff 40%, #5df0d7 72%, #3cff9b 100%)",
    labels: ["1020", "1005", "990", "975", "960"],
  },
];

const TRACK_LEGENDS = [
  { label: "PINN", color: "#ffdd5c" },
  { label: "Baseline", color: "#ff8b46" },
  { label: "Actual", color: "#4bee92" },
];

function MapLegend() {
  return (
    <Paper
      elevation={0}
      sx={{
        position: "absolute",
        right: { xs: 16, md: 24 },
        top: { xs: 92, md: 72 },
        width: { xs: "calc(100% - 32px)", md: 300 },
        p: 1.75,
        borderRadius: 3,
        border: "1px solid rgba(108, 136, 181, 0.18)",
        background: "rgba(7, 14, 26, 0.82)",
        backdropFilter: "blur(16px)",
        zIndex: 12,
      }}
    >
      <Stack spacing={1.8}>
        <Stack spacing={0.8}>
          <Typography variant="caption" sx={{ color: "rgba(234, 242, 255, 0.82)", letterSpacing: "0.08em" }}>
            TRACKS
          </Typography>
          <Stack direction="row" spacing={1.4} useFlexGap flexWrap="wrap">
            {TRACK_LEGENDS.map((item) => (
              <Stack key={item.label} direction="row" spacing={0.7} alignItems="center">
                <Box sx={{ width: 22, height: 3, borderRadius: 999, backgroundColor: item.color }} />
                <Typography variant="caption" sx={{ color: "rgba(214, 225, 246, 0.66)" }}>
                  {item.label}
                </Typography>
              </Stack>
            ))}
          </Stack>
        </Stack>
        {LEGENDS.map((legend) => (
          <Stack key={legend.title} spacing={0.7}>
            <Typography variant="caption" sx={{ color: "rgba(234, 242, 255, 0.82)", letterSpacing: "0.08em" }}>
              {legend.title}
            </Typography>
            <Paper
              elevation={0}
              sx={{
                height: 12,
                borderRadius: 999,
                background: legend.gradient,
                border: "1px solid rgba(255, 255, 255, 0.1)",
              }}
            />
            <Stack direction="row" justifyContent="space-between">
              {legend.labels.map((label) => (
                <Typography key={label} variant="caption" sx={{ color: "rgba(214, 225, 246, 0.58)" }}>
                  {label}
                </Typography>
              ))}
            </Stack>
          </Stack>
        ))}
      </Stack>
    </Paper>
  );
}

export default MapLegend;
