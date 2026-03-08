import React, { useEffect } from "react";
import { TopBar } from "./TopBar";
import { LatentSpaceExplorer } from "../latent/LatentSpaceExplorer";
import { InspectorPanel } from "../inspector/InspectorPanel";
import { useExplorerStore } from "../../state/useExplorerStore";

export const AppShell: React.FC = () => {
  const initialize = useExplorerStore((s) => s.initialize);
  const errorMessage = useExplorerStore((s) => s.errorMessage);
  const clearError = useExplorerStore((s) => s.clearError);

  useEffect(() => {
    void initialize();
  }, [initialize]);

  return (
    <div className="flex h-screen flex-col">
      <TopBar />
      <div className="flex min-h-0 flex-1 gap-4 bg-slate-50 p-4">
        <section className="flex min-h-0 flex-1 flex-col rounded-2xl border border-slate-200 bg-white shadow-sm">
          <LatentSpaceExplorer />
        </section>
        <aside className="w-96 min-w-[20rem] max-w-xs rounded-2xl border border-slate-200 bg-white shadow-sm">
          <InspectorPanel />
        </aside>
      </div>
      {errorMessage && (
        <div className="pointer-events-none fixed inset-x-0 bottom-4 flex justify-center">
          <div className="pointer-events-auto flex items-center gap-3 rounded-lg bg-red-50 px-4 py-2 text-sm text-red-800 shadow">
            <span>{errorMessage}</span>
            <button
              type="button"
              className="rounded border border-red-200 px-2 py-0.5 text-xs font-medium text-red-700 hover:bg-red-100"
              onClick={() => clearError()}
            >
              Dismiss
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

