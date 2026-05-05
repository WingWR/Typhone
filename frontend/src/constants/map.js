export const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN ?? "";
export const MAP_STYLE_URL = "mapbox://styles/mapbox/dark-v11";
export const DARK_TILE_URL = "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png";

export const INITIAL_VIEW_STATE = {
  longitude: 122.25,
  latitude: 29.55,
  zoom: 5.7,
  pitch: 0,
  bearing: 0,
};

export const WEATHER_STYLES = {
  rain: {
    label: "Rain",
    unit: "mm/h",
    accent: "#4be4ff",
  },
  wind: {
    label: "Wind",
    unit: "m/s",
    accent: "#f6df58",
  },
  pressure: {
    label: "Pressure",
    unit: "hPa",
    accent: "#91b4ff",
  },
};
