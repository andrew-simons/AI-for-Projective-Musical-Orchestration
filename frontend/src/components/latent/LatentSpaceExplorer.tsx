import React, { useCallback } from "react";
import { useExplorerStore } from "../../state/useExplorerStore";
import { LatentCanvas } from "./LatentCanvas";
import { ActivePointOverlay } from "./ActivePointOverlay";

export const LatentSpaceExplorer: React.FC = () => {
  const {
    samples,
    xRange,
    yRange,
    activePoint,
    interactionState,
    landmarks,
    pathControlPoints,
    pathSamples,
    moveActivePoint,
    generateAtPointNow,
    finishDrag,
    jumpToLandmark,
    uploadExcerpt,
  } = useExplorerStore((s) => ({
    samples: s.samples,
    xRange: s.xRange,
    yRange: s.yRange,
    activePoint: s.activePoint,
    interactionState: s.interactionState,
    landmarks: s.landmarks,
    pathControlPoints: s.pathControlPoints,
    pathSamples: s.pathSamples,
    moveActivePoint: s.moveActivePoint,
    generateAtPointNow: s.generateAtPointNow,
    finishDrag: s.finishDrag,
    jumpToLandmark: s.jumpToLandmark,
    uploadExcerpt: s.uploadExcerpt,
  }));

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        void uploadExcerpt(file);
      }
    },
    [uploadExcerpt]
  );

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
        <div className="flex flex-col">
          <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Latent space
          </span>
          <span className="text-sm text-slate-600">
            Click or drag to explore orchestration behaviour
          </span>
        </div>
        <label className="inline-flex cursor-pointer items-center gap-2 rounded-full border border-dashed border-slate-300 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-600 hover:border-accent hover:bg-accent-soft">
          <span>Upload piano MusicXML</span>
          <input
            type="file"
            accept=".musicxml,application/vnd.recordare.musicxml+xml"
            className="hidden"
            onChange={handleFileChange}
          />
        </label>
      </div>
      <div className="flex min-h-0 flex-1 flex-col">
        <LatentCanvas
          samples={samples}
          xRange={xRange}
          yRange={yRange}
          activePoint={activePoint}
          landmarks={landmarks}
          pathControlPoints={pathControlPoints}
          pathSamples={pathSamples}
          interactionState={interactionState}
          onMoveActivePoint={moveActivePoint}
          onGenerateAtPoint={generateAtPointNow}
          onFinishDrag={finishDrag}
          onJumpToLandmark={jumpToLandmark}
        />
        <div className="border-t border-slate-100 p-3">
          <ActivePointOverlay
            activePoint={activePoint}
            interactionState={interactionState}
          />
        </div>
      </div>
    </div>
  );
};

