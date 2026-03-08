import React, { useState } from "react";
import { useExplorerStore } from "../../state/useExplorerStore";

export const LandmarkList: React.FC = () => {
  const {
    landmarks,
    addLandmarkAtActive,
    removeLandmark,
    jumpToLandmark,
  } = useExplorerStore((s) => ({
    landmarks: s.landmarks,
    addLandmarkAtActive: s.addLandmarkAtActive,
    removeLandmark: s.removeLandmark,
    jumpToLandmark: s.jumpToLandmark,
  }));
  const activePoint = useExplorerStore((s) => s.activePoint);
  const [name, setName] = useState("");

  const handleSave = async () => {
    if (!name.trim() || !activePoint) return;
    await addLandmarkAtActive(name.trim());
    setName("");
  };

  return (
    <div className="space-y-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs">
      <div className="flex items-center justify-between">
        <span className="font-medium text-slate-700">Landmarks</span>
        <span className="text-[11px] text-slate-400">
          Save interesting regions
        </span>
      </div>
      <div className="flex gap-2">
        <input
          type="text"
          placeholder="Name current point"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="flex-1 rounded-full border border-slate-200 px-2 py-1 text-[11px] text-slate-700 placeholder:text-slate-400 focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent"
        />
        <button
          type="button"
          className="rounded-full bg-slate-900 px-3 py-1 text-[11px] font-medium text-white disabled:cursor-not-allowed disabled:bg-slate-300"
          disabled={!activePoint || !name.trim()}
          onClick={() => void handleSave()}
        >
          Save
        </button>
      </div>
      <div className="space-y-1 max-h-32 overflow-y-auto">
        {landmarks.map((lm) => (
          <div
            key={lm.id}
            className="flex items-center justify-between rounded-lg border border-slate-100 bg-slate-50 px-2 py-1"
          >
            <button
              type="button"
              className="flex flex-1 flex-col items-start text-left"
              onClick={() => void jumpToLandmark(lm.id)}
            >
              <span className="text-[11px] font-medium text-slate-700">
                {lm.name}
              </span>
              <span className="font-mono text-[10px] text-slate-500">
                x={lm.x.toFixed(2)}, y={lm.y.toFixed(2)}
              </span>
            </button>
            <button
              type="button"
              className="text-[11px] text-slate-400 hover:text-red-500"
              onClick={() => void removeLandmark(lm.id)}
            >
              ×
            </button>
          </div>
        ))}
        {landmarks.length === 0 && (
          <div className="text-[11px] text-slate-400">
            No landmarks yet. Save a few favourite orchestrations while you
            explore.
          </div>
        )}
      </div>
    </div>
  );
};

