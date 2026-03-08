import React, { useState } from "react";
import { useExplorerStore } from "../../state/useExplorerStore";

export const PathEditor: React.FC = () => {
  const {
    pathControlPoints,
    pathSamples,
    addPathPointFromActive,
    clearPath,
    samplePath,
    generateAtPointNow,
  } = useExplorerStore((s) => ({
    pathControlPoints: s.pathControlPoints,
    pathSamples: s.pathSamples,
    addPathPointFromActive: s.addPathPointFromActive,
    clearPath: s.clearPath,
    samplePath: s.samplePath,
    generateAtPointNow: s.generateAtPointNow,
  }));
  const [numSamples, setNumSamples] = useState(16);
  const [sliderT, setSliderT] = useState(0);

  const currentSample =
    pathSamples.length > 0
      ? pathSamples[Math.round(sliderT * (pathSamples.length - 1))]
      : null;

  const handlePlayPosition = async () => {
    if (!currentSample) return;
    await generateAtPointNow(currentSample.coords_2d);
  };

  return (
    <div className="space-y-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs">
      <div className="flex items-center justify-between">
        <span className="font-medium text-slate-700">Path exploration</span>
        <span className="text-[11px] text-slate-400">
          Interpolate through latent space
        </span>
      </div>
      <div className="flex gap-2">
        <button
          type="button"
          className="flex-1 rounded-full border border-slate-200 bg-slate-50 px-2 py-1 text-[11px] font-medium text-slate-700 hover:border-accent hover:bg-accent-soft"
          onClick={() => addPathPointFromActive()}
        >
          Add control point
        </button>
        <button
          type="button"
          className="rounded-full border border-slate-200 bg-white px-2 py-1 text-[11px] text-slate-500 hover:border-red-200 hover:text-red-600"
          onClick={() => clearPath()}
        >
          Clear
        </button>
      </div>
      <div className="space-y-1">
        <div className="flex items-center justify-between text-[11px] text-slate-500">
          <span>{pathControlPoints.length} control points</span>
          <span>{pathSamples.length} samples</span>
        </div>
        <div className="flex items-center gap-2">
          <input
            type="range"
            min={4}
            max={64}
            step={4}
            value={numSamples}
            onChange={(e) => setNumSamples(Number(e.target.value))}
            className="flex-1"
          />
          <span className="w-10 text-right text-[11px] text-slate-500">
            {numSamples}
          </span>
        </div>
        <button
          type="button"
          className="w-full rounded-full bg-slate-900 px-2 py-1 text-[11px] font-medium text-white disabled:cursor-not-allowed disabled:bg-slate-300"
          disabled={pathControlPoints.length === 0}
          onClick={() => void samplePath(numSamples)}
        >
          Sample path
        </button>
      </div>
      {pathSamples.length > 0 && (
        <div className="space-y-1">
          <div className="flex items-center justify-between text-[11px] text-slate-500">
            <span>Position along path</span>
            <span className="font-mono">
              t={sliderT.toFixed(2)}
              {currentSample && (
                <>
                  {" "}
                  • x={currentSample.coords_2d.x.toFixed(2)} y=
                  {currentSample.coords_2d.y.toFixed(2)}
                </>
              )}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={1}
            step={0.02}
            value={sliderT}
            onChange={(e) => setSliderT(Number(e.target.value))}
            className="w-full"
          />
          <button
            type="button"
            className="w-full rounded-full border border-slate-200 bg-white px-2 py-1 text-[11px] font-medium text-slate-700 hover:border-accent hover:bg-accent-soft disabled:cursor-not-allowed disabled:text-slate-300"
            disabled={!currentSample}
            onClick={() => void handlePlayPosition()}
          >
            Generate at current path position
          </button>
        </div>
      )}
    </div>
  );
};

