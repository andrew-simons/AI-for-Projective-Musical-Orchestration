import React from "react";
import { useExplorerStore } from "../../state/useExplorerStore";

export const InputExcerptCard: React.FC = () => {
  const excerpt = useExplorerStore((s) => s.excerpt);

  if (!excerpt) {
    return (
      <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-500">
        Upload a piano MusicXML file to encode it into latent space and view
        orchestration behaviour.
      </div>
    );
  }

  const { piano_summary: p } = excerpt;

  return (
    <div className="space-y-1 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs">
      <div className="flex items-center justify-between">
        <span className="font-medium text-slate-700">Piano excerpt</span>
        {p.filename && (
          <span className="max-w-[10rem] truncate text-[11px] text-slate-500">
            {p.filename}
          </span>
        )}
      </div>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px] text-slate-500">
        <div>
          <dt className="text-slate-400">Duration</dt>
          <dd className="font-mono text-slate-700">
            {p.duration_s.toFixed(1)}s
          </dd>
        </div>
        <div>
          <dt className="text-slate-400">Events</dt>
          <dd className="font-mono text-slate-700">{p.num_events}</dd>
        </div>
        <div>
          <dt className="text-slate-400">Tempo</dt>
          <dd className="font-mono text-slate-700">
            {p.bpm.toFixed(1)} bpm
          </dd>
        </div>
        <div>
          <dt className="text-slate-400">Meter</dt>
          <dd className="font-mono text-slate-700">{p.time_signature}</dd>
        </div>
      </dl>
    </div>
  );
};

