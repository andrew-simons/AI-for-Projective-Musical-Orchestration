import React from "react";
import { useExplorerStore } from "../../state/useExplorerStore";

export const PlaybackPanel: React.FC = () => {
  const preview = useExplorerStore((s) => s.orchestrationPreview);
  const interactionState = useExplorerStore((s) => s.interactionState);

  const hasPreview = Boolean(preview?.musicxmlUrl || preview?.midiUrl);

  const disabled =
    interactionState === "loading_generation" || !hasPreview;

  const handleOpen = (url?: string | null) => {
    if (!url) return;
    window.open(url, "_blank", "noopener,noreferrer");
  };

  return (
    <div className="space-y-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs">
      <div className="flex items-center justify-between">
        <span className="font-medium text-slate-700">Playback & export</span>
        {interactionState === "loading_generation" && (
          <span className="flex items-center gap-1 text-[11px] text-slate-400">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
            Updating…
          </span>
        )}
      </div>
      <p className="text-[11px] text-slate-500">
        Download MusicXML or MIDI for playback in your notation or DAW
        environment. The last successful orchestration is preserved while new
        generations load.
      </p>
      <div className="flex gap-2">
        <button
          type="button"
          className="flex-1 rounded-full bg-accent px-3 py-1 text-xs font-medium text-white shadow-sm hover:bg-blue-600 disabled:cursor-not-allowed disabled:bg-slate-300"
          disabled={disabled}
          onClick={() => handleOpen(preview?.musicxmlUrl)}
        >
          Download MusicXML
        </button>
        <button
          type="button"
          className="flex-1 rounded-full border border-slate-300 bg-white px-3 py-1 text-xs font-medium text-slate-700 shadow-sm hover:bg-slate-50 disabled:cursor-not-allowed disabled:border-slate-200 disabled:text-slate-300"
          disabled={disabled}
          onClick={() => handleOpen(preview?.midiUrl)}
        >
          Download MIDI
        </button>
      </div>
    </div>
  );
};

