import { useRef } from "react";
import { Button, Stack, Typography } from "@mui/material";

function UploadPanel({ activeSourceName, onLoadSample, onUploadFile, loadingPrediction }) {
  const fileInputRef = useRef(null);

  return (
    <Stack spacing={1.25}>
      <Typography variant="overline" sx={{ color: "#4be4ff", letterSpacing: "0.16em" }}>
        Input Dataset
      </Typography>
      <Typography variant="body2" sx={{ color: "rgba(219, 231, 247, 0.82)" }}>
        Upload a `.json` file with observed typhoon positions. The system will infer the next track segment and draw the
        animated path over the map.
      </Typography>
      <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
        <Button
          variant="contained"
          onClick={() => fileInputRef.current?.click()}
          disabled={loadingPrediction}
          sx={{
            background: "linear-gradient(135deg, #178de8 0%, #0a66c2 100%)",
            color: "#f6fbff",
            boxShadow: "0 10px 24px rgba(17, 122, 224, 0.24)",
          }}
        >
          Upload JSON
        </Button>
        <Button variant="outlined" onClick={onLoadSample} disabled={loadingPrediction}>
          Use Sample
        </Button>
      </Stack>
      <input
        ref={fileInputRef}
        type="file"
        accept=".json,application/json"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) {
            onUploadFile(file);
            event.target.value = "";
          }
        }}
      />
      <Typography variant="caption" sx={{ color: "rgba(216, 229, 248, 0.58)" }}>
        Current source: {activeSourceName}
      </Typography>
    </Stack>
  );
}

export default UploadPanel;
