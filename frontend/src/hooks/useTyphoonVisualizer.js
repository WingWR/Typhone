import { startTransition, useEffect, useMemo, useRef, useState } from "react";
import {
  fetchSampleTyphoonInput,
  fetchWeatherConditions,
  predictTyphoonFromFile,
  predictTyphoonFromPayload,
} from "../api/typhoonApi";
import {
  computeTimeRange,
  computeWeatherSummary,
  findActiveTrackPoint,
  normalizePredictionResponse,
  normalizeWeatherPoints,
} from "../utils/track";

export function useTyphoonVisualizer() {
  const [prediction, setPrediction] = useState(null);
  const [weatherField, setWeatherField] = useState("rain");
  const [weatherPoints, setWeatherPoints] = useState([]);
  const [weatherMeta, setWeatherMeta] = useState(null);
  const [loadingPrediction, setLoadingPrediction] = useState(true);
  const [loadingWeather, setLoadingWeather] = useState(false);
  const [error, setError] = useState("");
  const [currentTime, setCurrentTime] = useState(0);
  const [manualControl, setManualControl] = useState(false);
  const [activeSourceName, setActiveSourceName] = useState("sample_typhoon_input.json");
  const animationFrame = useRef(null);

  const combinedTrack = prediction?.combinedTrack ?? [];
  const timeRange = useMemo(() => computeTimeRange(combinedTrack), [combinedTrack]);
  const weatherSummary = useMemo(() => computeWeatherSummary(weatherPoints), [weatherPoints]);
  const activeTrackPoint = useMemo(
    () => findActiveTrackPoint(combinedTrack, currentTime),
    [combinedTrack, currentTime]
  );

  async function applyPrediction(requestFactory, sourceName) {
    setLoadingPrediction(true);
    setError("");

    try {
      const payload = await requestFactory();
      const normalized = normalizePredictionResponse(payload);
      startTransition(() => {
        setPrediction(normalized);
        setCurrentTime(normalized.combinedTrack[0]?.timeValue ?? 0);
        setManualControl(false);
        setActiveSourceName(sourceName);
      });
    } catch (requestError) {
      setError(requestError.message || "Failed to generate forecast.");
    } finally {
      setLoadingPrediction(false);
    }
  }

  async function loadSampleForecast() {
    return applyPrediction(async () => {
      const samplePayload = await fetchSampleTyphoonInput();
      return predictTyphoonFromPayload(samplePayload);
    }, "sample_typhoon_input.json");
  }

  async function uploadTyphoonJson(file) {
    return applyPrediction(() => predictTyphoonFromFile(file), file.name);
  }

  useEffect(() => {
    loadSampleForecast();
  }, []);

  useEffect(() => {
    if (!prediction?.weatherContext) {
      return;
    }

    let cancelled = false;

    async function loadWeather() {
      setLoadingWeather(true);
      try {
        const payload = await fetchWeatherConditions(weatherField, prediction.weatherContext);
        if (cancelled) {
          return;
        }

        startTransition(() => {
          setWeatherPoints(normalizeWeatherPoints(payload.points));
          setWeatherMeta(payload.metadata);
          setError("");
        });
      } catch (requestError) {
        if (!cancelled) {
          setError(requestError.message || "Failed to load weather field.");
        }
      } finally {
        if (!cancelled) {
          setLoadingWeather(false);
        }
      }
    }

    loadWeather();

    return () => {
      cancelled = true;
    };
  }, [prediction, weatherField]);

  useEffect(() => {
    if (manualControl || combinedTrack.length < 2) {
      return undefined;
    }

    const start = combinedTrack[0].timeValue;
    const end = combinedTrack[combinedTrack.length - 1].timeValue;
    const speed = 120;
    let localTime = start;

    const animate = () => {
      localTime = localTime >= end ? start : localTime + speed;
      setCurrentTime(localTime);
      animationFrame.current = window.requestAnimationFrame(animate);
    };

    animationFrame.current = window.requestAnimationFrame(animate);

    return () => {
      if (animationFrame.current) {
        window.cancelAnimationFrame(animationFrame.current);
      }
    };
  }, [combinedTrack, manualControl]);

  return {
    prediction,
    weatherField,
    setWeatherField,
    weatherPoints,
    weatherMeta,
    weatherSummary,
    loadingPrediction,
    loadingWeather,
    error,
    currentTime,
    setCurrentTime,
    manualControl,
    setManualControl,
    timeRange,
    activeTrackPoint,
    activeSourceName,
    loadSampleForecast,
    uploadTyphoonJson,
  };
}
