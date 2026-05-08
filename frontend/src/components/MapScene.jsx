import { useMemo } from "react";
import DeckGL from "@deck.gl/react";
import { TileLayer } from "@deck.gl/geo-layers";
import { BitmapLayer } from "@deck.gl/layers";
import { PathLayer, ScatterplotLayer, TripsLayer } from "deck.gl";
import { Box } from "@mui/material";
import { DARK_TILE_URL, INITIAL_VIEW_STATE } from "../constants/map";
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
}) {
  const tripData = useMemo(() => buildTripData(combinedTrack), [combinedTrack]);
  const baselineTripData = useMemo(
    () => buildComparisonTripData(observedTrack, baselineTrack),
    [baselineTrack, observedTrack]
  );
  const actualTripData = useMemo(
    () => buildComparisonTripData(observedTrack, actualTrack),
    [actualTrack, observedTrack]
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
          desaturate: 0.0,
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
      currentTime: currentTime,
      capRounded: true,
      jointRounded: true,
      fadeTrail: true,
    });

    return [
      baseMapLayer,
      corridorLayer,
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
  }, [
    activeTrackPoint,
    actualTrack,
    actualTripData,
    baselineTrack,
    baselineTripData,
    combinedTrack,
    currentTime,
    particleData,
    tripData,
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
    </Box>
  );
}

export default MapScene;
