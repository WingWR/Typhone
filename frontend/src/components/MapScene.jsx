import { useDeferredValue, useMemo } from "react";
import DeckGL from "@deck.gl/react";
import { TileLayer } from "@deck.gl/geo-layers";
import { BitmapLayer } from "@deck.gl/layers";
import { HeatmapLayer, PathLayer, ScatterplotLayer, TripsLayer } from "deck.gl";
import { Box } from "@mui/material";
import { DARK_TILE_URL, INITIAL_VIEW_STATE } from "../constants/map";
import { buildTyphoonParticles } from "../utils/particles";
import { buildTripData } from "../utils/track";

function buildComparisonTripData(observedTrack, forecastTrack) {
  if (!forecastTrack.length) {
    return [];
  }

  const anchor = observedTrack[observedTrack.length - 1];
  return buildTripData(anchor ? [anchor, ...forecastTrack] : forecastTrack);
}

function MapScene({
  activeTrackPoint,
  actualTrack,
  baselineTrack,
  combinedTrack,
  currentTime,
  observedTrack,
  weatherField,
  weatherPoints,
  weatherUnit,
}) {
  const deferredTime = useDeferredValue(currentTime);
  const tripData = useMemo(() => buildTripData(combinedTrack), [combinedTrack]);
  const baselineTripData = useMemo(
    () => buildComparisonTripData(observedTrack, baselineTrack),
    [baselineTrack, observedTrack]
  );
  const actualTripData = useMemo(
    () => buildComparisonTripData(observedTrack, actualTrack),
    [actualTrack, observedTrack]
  );
  const particleData = useMemo(
    () => buildTyphoonParticles(activeTrackPoint, currentTime),
    [activeTrackPoint, currentTime]
  );

  const layers = useMemo(() => {
    const baseMapLayer = new TileLayer({
      id: "dark-basemap",
      data: DARK_TILE_URL,
      minZoom: 0,
      maxZoom: 19,
      tileSize: 256,
      renderSubLayers: (props) => {
        const {
          bbox: { west, south, east, north },
        } = props.tile;

        return new BitmapLayer(props, {
          data: null,
          image: props.data,
          bounds: [west, south, east, north],
          desaturate: 0.15,
        });
      },
    });

    const corridorLayer = new PathLayer({
      id: "track-corridor",
      data: tripData,
      getPath: (d) => d.waypoints,
      getColor: [255, 215, 104, 54],
      widthMinPixels: 22,
      rounded: true,
    });

    const glowPathLayer = new PathLayer({
      id: "track-glow",
      data: tripData,
      getPath: (d) => d.waypoints,
      getColor: [255, 226, 108, 62],
      widthMinPixels: 10,
      rounded: true,
    });

    const staticPathLayer = new PathLayer({
      id: "track-outline",
      data: tripData,
      getPath: (d) => d.waypoints,
      getColor: [238, 202, 82, 142],
      widthMinPixels: 3,
      rounded: true,
    });

    const baselinePathLayer = new PathLayer({
      id: "baseline-track",
      data: baselineTripData,
      getPath: (d) => d.waypoints,
      getColor: [255, 139, 70, 175],
      widthMinPixels: 4,
      rounded: true,
    });

    const actualPathLayer = new PathLayer({
      id: "actual-track",
      data: actualTripData,
      getPath: (d) => d.waypoints,
      getColor: [75, 238, 146, 220],
      widthMinPixels: 5,
      rounded: true,
    });

    const markerLayer = new ScatterplotLayer({
      id: "track-markers",
      data: combinedTrack,
      pickable: true,
      radiusMinPixels: 4,
      radiusMaxPixels: 8,
      getPosition: (d) => d.coordinates,
      getRadius: (d) => (d.source === "forecast" ? 7200 : 5600),
      getFillColor: (d) => (d.source === "forecast" ? [255, 217, 95, 210] : [201, 232, 255, 176]),
      getLineColor: [255, 247, 210, 220],
      stroked: true,
      lineWidthMinPixels: 1,
    });

    const baselineMarkerLayer = new ScatterplotLayer({
      id: "baseline-markers",
      data: baselineTrack,
      pickable: true,
      radiusMinPixels: 3,
      radiusMaxPixels: 6,
      getPosition: (d) => d.coordinates,
      getRadius: 5000,
      getFillColor: [255, 139, 70, 165],
      getLineColor: [255, 205, 166, 220],
      stroked: true,
      lineWidthMinPixels: 1,
    });

    const actualMarkerLayer = new ScatterplotLayer({
      id: "actual-markers",
      data: actualTrack,
      pickable: true,
      radiusMinPixels: 4,
      radiusMaxPixels: 7,
      getPosition: (d) => d.coordinates,
      getRadius: 5600,
      getFillColor: [75, 238, 146, 210],
      getLineColor: [202, 255, 226, 230],
      stroked: true,
      lineWidthMinPixels: 1,
    });

    const particleLayer = new ScatterplotLayer({
      id: "typhoon-particles",
      data: particleData,
      pickable: false,
      radiusUnits: "pixels",
      stroked: false,
      getPosition: (d) => d.position,
      getRadius: (d) => d.radiusPixels,
      radiusMinPixels: 1,
      radiusMaxPixels: 12,
      getFillColor: (d) => d.color,
      opacity: 0.92,
    });

    const activeCenterLayer = activeTrackPoint
      ? new ScatterplotLayer({
          id: "active-center",
          data: [activeTrackPoint],
          radiusMinPixels: 12,
          radiusMaxPixels: 18,
          getPosition: (d) => d.coordinates,
          getRadius: 12000,
          getFillColor: [255, 236, 122, 72],
          getLineColor: [255, 235, 120, 255],
          stroked: true,
          lineWidthMinPixels: 3,
        })
      : null;

    const trackLayer = new TripsLayer({
      id: "typhoon-track",
      data: tripData,
      getPath: (d) => d.waypoints,
      getTimestamps: (d) => d.timestamps,
      getColor: [255, 221, 92],
      opacity: 0.98,
      widthMinPixels: 7,
      rounded: true,
      trailLength: 5 * 3600,
      currentTime: deferredTime,
      capRounded: true,
      jointRounded: true,
      fadeTrail: true,
    });

    if (weatherField === "rain") {
      return [
        baseMapLayer,
        corridorLayer,
        new HeatmapLayer({
          id: "rain-heatmap",
          data: weatherPoints,
          getPosition: (d) => d.position,
          getWeight: (d) => d.value,
          aggregation: "SUM",
          radiusPixels: 46,
          intensity: 0.84,
          threshold: 0.02,
          colorRange: [
            [7, 20, 48],
            [11, 42, 92],
            [0, 81, 164],
            [0, 141, 222],
            [58, 217, 255],
            [172, 248, 255],
          ],
        }),
        baselinePathLayer,
        actualPathLayer,
        glowPathLayer,
        staticPathLayer,
        markerLayer,
        baselineMarkerLayer,
        actualMarkerLayer,
        particleLayer,
        trackLayer,
        activeCenterLayer,
      ].filter(Boolean);
    }

    const weatherFieldLayer =
      weatherField === "wind"
        ? new ScatterplotLayer({
            id: "wind-points",
            data: weatherPoints,
            pickable: true,
            opacity: 0.68,
            radiusMinPixels: 2,
            radiusMaxPixels: 6,
            getPosition: (d) => d.position,
            getRadius: (d) => Math.max(2400, d.value * 180),
            getFillColor: (d) => {
              const ratio = Math.min(d.value / 40, 1);
              return [84 + ratio * 170, 186 + ratio * 40, 255 - ratio * 110, 112 + ratio * 74];
            },
          })
        : new ScatterplotLayer({
            id: "pressure-points",
            data: weatherPoints,
            pickable: true,
            opacity: 0.5,
            radiusMinPixels: 2,
            radiusMaxPixels: 5,
            getPosition: (d) => d.position,
            getRadius: () => 2600,
            getFillColor: (d) => {
              const ratio = Math.min(Math.max((1015 - d.value) / 22, 0), 1);
              return [72 + ratio * 56, 120 + ratio * 88, 235, 72 + ratio * 88];
            },
          });

    return [
      baseMapLayer,
      weatherFieldLayer,
      baselinePathLayer,
      actualPathLayer,
      corridorLayer,
      glowPathLayer,
      staticPathLayer,
      markerLayer,
      baselineMarkerLayer,
      actualMarkerLayer,
      particleLayer,
      trackLayer,
      activeCenterLayer,
    ].filter(Boolean);
  }, [
    activeTrackPoint,
    actualTrack,
    actualTripData,
    baselineTrack,
    baselineTripData,
    combinedTrack,
    deferredTime,
    particleData,
    tripData,
    weatherField,
    weatherPoints,
  ]);

  return (
    <Box sx={{ position: "relative", height: "100vh", width: "100%" }}>
      <DeckGL
        controller
        initialViewState={INITIAL_VIEW_STATE}
        layers={layers}
        getTooltip={({ object }) => {
          if (!object) {
            return null;
          }

          if (object.value !== undefined) {
            return `${weatherField.toUpperCase()}: ${object.value.toFixed(2)} ${weatherUnit}`;
          }

          if (object.timestamp) {
            const labelMap = {
              observed: "Observed",
              baseline: "Baseline",
              actual: "Actual",
              forecast: "PINN Forecast",
            };
            return `${labelMap[object.source] ?? "Track"}\n${object.timestamp}`;
          }

          return null;
        }}
      />
      <Box
        sx={{
          pointerEvents: "none",
          position: "absolute",
          inset: 0,
          background:
            "radial-gradient(circle at 68% 64%, rgba(255, 206, 74, 0.2), transparent 16%), radial-gradient(circle at 34% 28%, rgba(26, 118, 255, 0.12), transparent 20%), linear-gradient(180deg, rgba(5, 10, 18, 0.03), rgba(3, 8, 17, 0.3))",
        }}
      />
    </Box>
  );
}

export default MapScene;
