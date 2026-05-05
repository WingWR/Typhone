function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

export function buildTyphoonParticles(activeTrackPoint, currentTime) {
  if (!activeTrackPoint) {
    return [];
  }

  const centerLng = activeTrackPoint.lng;
  const centerLat = activeTrackPoint.lat;
  const intensity = clamp((activeTrackPoint.wind_speed ?? 28) / 42, 0.6, 1.3);
  const phase = (currentTime % 7200) / 7200;
  const particleCount = 220;
  const particles = [];

  for (let index = 0; index < particleCount; index += 1) {
    const arm = index % 3;
    const normalized = index / particleCount;
    const swirlTurns = 2.2 + arm * 0.42;
    const angle = normalized * Math.PI * 2 * swirlTurns + phase * Math.PI * 8 + arm * 2.1;
    const radius = 0.035 + normalized * 0.68;
    const eyeCompression = 1 - Math.exp(-normalized * 5.5);
    const radialScale = radius * eyeCompression * intensity;
    const lngOffset = Math.cos(angle) * radialScale * 0.54;
    const latOffset = Math.sin(angle) * radialScale * 0.42;
    const brightness = clamp(1 - normalized * 0.82 + arm * 0.04, 0.18, 1);
    const alpha = Math.round(58 + brightness * 176);

    particles.push({
      position: [centerLng + lngOffset, centerLat + latOffset],
      radiusPixels: 1.5 + brightness * 4.8,
      color: [
        Math.round(248 + brightness * 7),
        Math.round(176 + brightness * 54),
        Math.round(52 + brightness * 60),
        alpha,
      ],
    });
  }

  return particles;
}
