const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:5000/api";

async function parseResponse(response) {
  if (!response.ok) {
    let message = "Request failed";
    try {
      const payload = await response.json();
      message = payload.error ?? message;
    } catch {
      message = response.statusText || message;
    }
    throw new Error(message);
  }

  return response.json();
}

export async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    const payload = await response.json();
    return payload.status === "ok";
  } catch {
    return false;
  }
}

export async function fetchSampleTyphoonInput() {
  const response = await fetch(`${API_BASE}/sample_typhoon_input`);
  return parseResponse(response);
}

export async function predictTyphoonFromPayload(payload) {
  const response = await fetch(`${API_BASE}/predict_typhoon`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return parseResponse(response);
}

export async function predictTyphoonFromFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/predict_typhoon`, {
    method: "POST",
    body: formData,
  });
  return parseResponse(response);
}

export async function fetchWeatherConditions(field, weatherContext) {
  const params = new URLSearchParams({
    field,
    center_lng: weatherContext.centerLng.toString(),
    center_lat: weatherContext.centerLat.toString(),
    max_wind_speed: weatherContext.maxWindSpeed.toString(),
    central_pressure: weatherContext.centralPressure.toString(),
  });
  const response = await fetch(`${API_BASE}/get_weather_conditions?${params.toString()}`);
  return parseResponse(response);
}
