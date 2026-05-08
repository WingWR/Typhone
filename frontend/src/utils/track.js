export function toUnixSeconds(timestamp) {
  return Math.floor(new Date(timestamp).getTime() / 1000);
}

function normalizeTrackPoint(point) {
  return {
    ...point,
    coordinates: [point.lng, point.lat],
    timeValue: toUnixSeconds(point.timestamp),
  };
}

export function normalizePredictionResponse(payload) {
  const observedTrack = (payload.observed_track ?? []).map(normalizeTrackPoint);
  const predictedTrack = (payload.predicted_track ?? []).map(normalizeTrackPoint);
  const pinnTrack = (payload.pinn_track ?? []).map(normalizeTrackPoint);
  const baselineTrack = (payload.baseline_track ?? []).map(normalizeTrackPoint);
  const actualTrack = (payload.actual_track ?? []).map(normalizeTrackPoint);
  const combinedTrack = (payload.combined_track ?? []).map(normalizeTrackPoint);

  return {
    stormId: payload.storm_id,
    stormName: payload.storm_name,
    basin: payload.basin,
    modelName: payload.model_name,
    modelType: payload.model_type,
    forecastSteps: payload.forecast_steps,
    timeStepHours: payload.time_step_hours,
    summary: payload.summary,
    losses: payload.losses ?? {},
    metrics: payload.metrics ?? {},
    weatherContext: {
      centerLng: payload.weather_context.center_lng,
      centerLat: payload.weather_context.center_lat,
      maxWindSpeed: payload.weather_context.max_wind_speed,
      centralPressure: payload.weather_context.central_pressure,
    },
    observedTrack,
    predictedTrack,
    pinnTrack,
    baselineTrack,
    actualTrack,
    combinedTrack,
  };
}

export function buildTripData(track) {
  if (!track.length) {
    return [];
  }

  return [
    {
      waypoints: track.map((point) => point.coordinates),
      timestamps: track.map((point) => point.timeValue),
    },
  ];
}

export function computeTimeRange(track) {
  if (!track.length) {
    return [0, 1];
  }

  return [track[0].timeValue, track[track.length - 1].timeValue];
}

export function findActiveTrackPoint(track, currentTime) {
  if (!track.length) {
    return null;
  }

  let closest = track[0];
  let smallestDistance = Math.abs(track[0].timeValue - currentTime);

  for (const point of track) {
    const distance = Math.abs(point.timeValue - currentTime);
    if (distance < smallestDistance) {
      smallestDistance = distance;
      closest = point;
    }
  }

  return closest;
}
