import React, { useCallback, useMemo, useRef, useState } from "react";
import type {
  InteractionState,
  LatentCoords,
  LatentSampleItem,
  Landmark,
  PathSample,
} from "../../api/types";

interface Props {
  samples: LatentSampleItem[];
  xRange: [number, number] | null;
  yRange: [number, number] | null;
  activePoint: LatentCoords | null;
  landmarks: Landmark[];
  pathControlPoints: { id: string; x: number; y: number }[];
  pathSamples: PathSample[];
  interactionState: InteractionState;
  onMoveActivePoint: (coords: LatentCoords) => void;
  onGenerateAtPoint: (coords: LatentCoords) => void;
  onFinishDrag: () => void;
  onJumpToLandmark: (id: string) => void;
}

interface CameraState {
  centerX: number;
  centerY: number;
  scale: number;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export const LatentCanvas: React.FC<Props> = ({
  samples,
  xRange,
  yRange,
  activePoint,
  landmarks,
  pathControlPoints,
  pathSamples,
  interactionState,
  onMoveActivePoint,
  onGenerateAtPoint,
  onFinishDrag,
  onJumpToLandmark,
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [isDraggingPoint, setIsDraggingPoint] = useState(false);
  const [isPanning, setIsPanning] = useState(false);
  const panStart = useRef<{ x: number; y: number } | null>(null);
  const [camera, setCamera] = useState<CameraState>(() => {
    const [xMin, xMax] = xRange ?? [-1, 1];
    const [yMin, yMax] = yRange ?? [-1, 1];
    return {
      centerX: (xMin + xMax) / 2,
      centerY: (yMin + yMax) / 2,
      scale: 1,
    };
  });

  const worldBounds = useMemo(() => {
    const [xMin, xMax] = xRange ?? [-1, 1];
    const [yMin, yMax] = yRange ?? [-1, 1];
    return { xMin, xMax, yMin, yMax };
  }, [xRange, yRange]);

  const worldToScreen = useCallback(
    (x: number, y: number): { sx: number; sy: number } => {
      const { xMin, xMax, yMin, yMax } = worldBounds;
      const spanX = xMax - xMin || 1;
      const spanY = yMax - yMin || 1;
      const nx = (x - camera.centerX) / spanX;
      const ny = (y - camera.centerY) / spanY;
      const sx = 0.5 + nx * camera.scale;
      const sy = 0.5 - ny * camera.scale;
      return { sx, sy };
    },
    [camera.centerX, camera.centerY, camera.scale, worldBounds]
  );

  const screenToWorld = useCallback(
    (clientX: number, clientY: number): LatentCoords | null => {
      const svg = svgRef.current;
      if (!svg) return null;
      const rect = svg.getBoundingClientRect();
      const u = (clientX - rect.left) / rect.width;
      const v = (clientY - rect.top) / rect.height;

      const { xMin, xMax, yMin, yMax } = worldBounds;
      const spanX = xMax - xMin || 1;
      const spanY = yMax - yMin || 1;

      const nx = (u - 0.5) / camera.scale;
      const ny = (0.5 - v) / camera.scale;

      const x = camera.centerX + nx * spanX;
      const y = camera.centerY + ny * spanY;
      return { x, y };
    },
    [camera.centerX, camera.centerY, camera.scale, worldBounds]
  );

  const handleWheel: React.WheelEventHandler<SVGSVGElement> = (e) => {
    e.preventDefault();
    const delta = e.deltaY;
    const factor = delta > 0 ? 0.9 : 1.1;

    setCamera((prev) => ({
      ...prev,
      scale: clamp(prev.scale * factor, 0.3, 4),
    }));
  };

  const handleMouseDown: React.MouseEventHandler<SVGSVGElement> = (e) => {
    if (e.button !== 0) return;
    const world = screenToWorld(e.clientX, e.clientY);
    if (!world) return;

    if (activePoint) {
      const { sx, sy } = worldToScreen(activePoint.x, activePoint.y);
      const svg = svgRef.current;
      if (svg) {
        const rect = svg.getBoundingClientRect();
        const px = sx * rect.width;
        const py = sy * rect.height;
        const dx = e.clientX - (rect.left + px);
        const dy = e.clientY - (rect.top + py);
        const distSq = dx * dx + dy * dy;
        const radiusPx = 12;
        if (distSq <= radiusPx * radiusPx) {
          setIsDraggingPoint(true);
          return;
        }
      }
    }

    setIsPanning(true);
    panStart.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove: React.MouseEventHandler<SVGSVGElement> = (e) => {
    if (isDraggingPoint) {
      const world = screenToWorld(e.clientX, e.clientY);
      if (world) {
        onMoveActivePoint(world);
      }
      return;
    }
    if (isPanning && panStart.current) {
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const dx = (e.clientX - panStart.current.x) / rect.width;
      const dy = (e.clientY - panStart.current.y) / rect.height;

      const { xMin, xMax, yMin, yMax } = worldBounds;
      const spanX = xMax - xMin || 1;
      const spanY = yMax - yMin || 1;

      setCamera((prev) => ({
        ...prev,
        centerX: prev.centerX - dx * spanX * prev.scale,
        centerY: prev.centerY + dy * spanY * prev.scale,
      }));
      panStart.current = { x: e.clientX, y: e.clientY };
    }
  };

  const handleMouseUp: React.MouseEventHandler<SVGSVGElement> = (e) => {
    if (isDraggingPoint) {
      setIsDraggingPoint(false);
      onFinishDrag();
      return;
    }
    if (isPanning) {
      setIsPanning(false);
      panStart.current = null;
      return;
    }

    if (e.button === 0) {
      const world = screenToWorld(e.clientX, e.clientY);
      if (world) {
        onGenerateAtPoint(world);
      }
    }
  };

  const handleMouseLeave: React.MouseEventHandler<SVGSVGElement> = () => {
    if (isDraggingPoint) {
      setIsDraggingPoint(false);
      onFinishDrag();
    }
    if (isPanning) {
      setIsPanning(false);
      panStart.current = null;
    }
  };

  const activeScreen = activePoint
    ? worldToScreen(activePoint.x, activePoint.y)
    : null;

  return (
    <div className="flex h-full w-full flex-col">
      <div className="relative flex-1">
        <svg
          ref={svgRef}
          className="h-full w-full cursor-crosshair rounded-t-2xl bg-slate-50"
          viewBox="0 0 1 1"
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
        >
          <defs>
            <linearGradient id="bgGrid" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#f9fafb" />
              <stop offset="100%" stopColor="#eef2ff" />
            </linearGradient>
          </defs>
          <rect x={0} y={0} width={1} height={1} fill="url(#bgGrid)" />

          {[0.25, 0.5, 0.75].map((t) => (
            <g key={t}>
              <line
                x1={t}
                y1={0}
                x2={t}
                y2={1}
                stroke="#e5e7eb"
                strokeWidth={0.002}
              />
              <line
                x1={0}
                y1={t}
                x2={1}
                y2={t}
                stroke="#e5e7eb"
                strokeWidth={0.002}
              />
            </g>
          ))}

          {samples.map((s) => {
            const { sx, sy } = worldToScreen(s.x, s.y);
            return (
              <circle
                key={s.id}
                cx={sx}
                cy={sy}
                r={0.004}
                fill="#cbd5f5"
                opacity={0.7}
              />
            );
          })}

          {pathSamples.length > 1 && (
            <polyline
              points={pathSamples
                .map((p) => {
                  const { sx, sy } = worldToScreen(
                    p.coords_2d.x,
                    p.coords_2d.y
                  );
                  return `${sx},${sy}`;
                })
                .join(" ")}
              fill="none"
              stroke="#22c55e"
              strokeWidth={0.004}
              strokeOpacity={0.7}
            />
          )}

          {pathControlPoints.map((p) => {
            const { sx, sy } = worldToScreen(p.x, p.y);
            return (
              <circle
                key={p.id}
                cx={sx}
                cy={sy}
                r={0.006}
                fill="#22c55e"
                stroke="#065f46"
                strokeWidth={0.002}
              />
            );
          })}

          {landmarks.map((lm) => {
            const { sx, sy } = worldToScreen(lm.x, lm.y);
            return (
              <g
                key={lm.id}
                onClick={(e) => {
                  e.stopPropagation();
                  onJumpToLandmark(lm.id);
                }}
                className="cursor-pointer"
              >
                <circle
                  cx={sx}
                  cy={sy}
                  r={0.007}
                  fill="#f97316"
                  stroke="#7c2d12"
                  strokeWidth={0.002}
                />
              </g>
            );
          })}

          {activeScreen && (
            <g>
              <circle
                cx={activeScreen.sx}
                cy={activeScreen.sy}
                r={0.012}
                fill="#1d4ed8"
                fillOpacity={0.15}
              />
              <circle
                cx={activeScreen.sx}
                cy={activeScreen.sy}
                r={0.006}
                fill="#2563eb"
                stroke="#1d4ed8"
                strokeWidth={0.002}
              />
            </g>
          )}
        </svg>

        {(interactionState === "loading_generation" ||
          interactionState === "dragging") && (
          <div className="pointer-events-none absolute inset-0 flex items-start justify-end p-3">
            <div className="flex items-center gap-2 rounded-full bg-white/80 px-3 py-1 text-xs text-slate-600 shadow-sm">
              <span className="h-2 w-2 animate-pulse rounded-full bg-accent" />
              <span>
                {interactionState === "dragging"
                  ? "Adjusting point…"
                  : "Generating orchestration…"}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

