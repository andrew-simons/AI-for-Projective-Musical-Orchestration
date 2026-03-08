import React from "react";
import { useExplorerStore } from "../../state/useExplorerStore";

export const OrchestrationSummary: React.FC = () => {
  const excerpt = useExplorerStore((s) => s.excerpt);
  const preview = useExplorerStore((s) => s.orchestrationPreview);

  const summary =
    preview?.summary ?? excerpt?.instrument_activity_summary ?? null;

  if (!summary) {
    return (
      <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-500">
        Instrument activity and orchestration summaries will appear here after
        generation.
      </div>
    );
  }

  const top = summary.top.slice(0, 8);

  return (
    <div className="space-y-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs">
      <div className="flex items-center justify-between">
        <span className="font-medium text-slate-700">Orchestration summary</span>
        <span className="text-[11px] text-slate-400">Top instruments</span>
      </div>
      <div className="space-y-1">
        {top.map((t) => (
          <div key={t.program} className="flex items-center gap-2">
            <span className="w-8 text-[11px] font-mono text-slate-500">
              {t.program.toString().padStart(3, "0")}
            </span>
            <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-slate-100">
              <div
                className="absolute inset-y-0 left-0 rounded-full bg-accent"
                style={{ width: `${Math.min(100, t.score * 120)}%` }}
              />
            </div>
            <span className="w-10 text-right text-[11px] font-mono text-slate-500">
              {t.score.toFixed(2)}
            </span>
          </div>
        ))}
        {top.length === 0 && (
          <div className="text-[11px] text-slate-400">
            No active instruments detected.
          </div>
        )}
      </div>
    </div>
  );
};

