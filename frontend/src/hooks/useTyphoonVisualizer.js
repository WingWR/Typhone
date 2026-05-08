import { startTransition, useEffect, useMemo, useRef, useState } from "react";
import {
  checkHealth,
  predictTyphoonFromFile,
} from "../api/typhoonApi";
import {
  computeTimeRange,
  findActiveTrackPoint,
  normalizePredictionResponse,
} from "../utils/track";

export function useTyphoonVisualizer() {
  const [prediction, setPrediction] = useState(null);
  const [loadingPrediction, setLoadingPrediction] = useState(true);
  const [error, setError] = useState("");
  const [currentTime, setCurrentTime] = useState(0);
  const [manualControl, setManualControl] = useState(false);
  const [activeSourceName, setActiveSourceName] = useState("");
  const animTimer = useRef(null);
  const animTime = useRef(0);
  const wasPaused = useRef(false);
  const currentTimeRef = useRef(currentTime);
  currentTimeRef.current = currentTime;

  const combinedTrack = prediction?.combinedTrack ?? [];
  const timeRange = useMemo(() => computeTimeRange(combinedTrack), [combinedTrack]);
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

  async function uploadTyphoonJson(file) {
    return applyPrediction(() => predictTyphoonFromFile(file), file.name);
  }

  useEffect(() => {
    async function initBackendHealth() {
      const ok = await checkHealth();
      if (!ok) {
        setError("Backend is unreachable. Check that Flask is running on port 5000.");
      }
      setLoadingPrediction(false);
    }
    initBackendHealth();
  }, []);

  const hasPrediction = prediction !== null && (prediction?.combinedTrack?.length ?? 0) > 0;

  useEffect(() => {
    if (manualControl || combinedTrack.length < 2) {
      wasPaused.current = true;
      return undefined;
    }

    const start = combinedTrack[0].timeValue;
    const end = combinedTrack[combinedTrack.length - 1].timeValue;
    const fps = 30;
    const simHoursPerSec = 1.5;
    const step = Math.round((simHoursPerSec * 3600) / fps);

    if (wasPaused.current || animTime.current < start || animTime.current > end) {
      animTime.current = currentTimeRef.current >= start && currentTimeRef.current <= end
        ? currentTimeRef.current
        : start;
      wasPaused.current = false;
    }

    animTimer.current = setInterval(() => {
      animTime.current = animTime.current >= end ? start : animTime.current + step;
      setCurrentTime(animTime.current);
    }, 1000 / fps);

    return () => {
      if (animTimer.current) {
        clearInterval(animTimer.current);
        animTimer.current = null;
      }
    };
  }, [combinedTrack, manualControl]);

  return {
    prediction,
    hasPrediction,
    loadingPrediction,
    error,
    currentTime,
    setCurrentTime,
    manualControl,
    setManualControl,
    timeRange,
    activeTrackPoint,
    activeSourceName,
    uploadTyphoonJson,
  };
}
