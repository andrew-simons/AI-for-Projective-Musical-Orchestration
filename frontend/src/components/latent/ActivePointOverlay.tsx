import React from "react";
import type { InteractionState, LatentCoords } from "../../api/types";

interface Props {
  activePoint: LatentCoords | null;
  interactionState: InteractionState;
}

export const ActivePointOverlay: React.FC<Props> = ({
  activePoint,
  interactionState,
}) => {
  if (!activePoint) {
    return (
      <div className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-500">
        <span>No active point</span>
        <span>Upload a piano excerpt to begin</span>
      </div>
    );
  }

  const { x, y } = activePoint;

  const label =
    interactionState === "loading_generation"
      ? "Loading orchestration preview…"
      : interactionState === "generation_ready"
      ? "Orchestration ready"
      : interactionState === "dragging"
      ? "Dragging in latent space"
      : "Idle";

  return (
    <div className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-600">
      <div className="flex items-baseline gap-2">
        <span className="font-medium text-slate-700">Active point</span>
        <span className="font-mono text-[11px] text-slate-500">
          x={x.toFixed(3)}, y={y.toFixed(3)}
        </span>
      </div>
      <span className="text-[11px] text-slate-500">{label}</span>
    </div>
  );
};

