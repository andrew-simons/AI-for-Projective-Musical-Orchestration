import React from "react";

export const TopBar: React.FC = () => {
  return (
    <header className="flex items-center justify-between border-b border-slate-200 bg-white px-6 py-3">
      <div className="flex flex-col">
        <span className="text-sm font-semibold uppercase tracking-wide text-slate-500">
          Projective Orchestration
        </span>
        <span className="text-lg font-medium text-slate-900">
          Latent Space Explorer
        </span>
      </div>
      <div className="text-xs text-slate-400">
        AI orchestration research demo
      </div>
    </header>
  );
};

