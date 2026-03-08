import React from "react";
import { InputExcerptCard } from "./InputExcerptCard";
import { OrchestrationSummary } from "./OrchestrationSummary";
import { PlaybackPanel } from "./PlaybackPanel";
import { LandmarkList } from "../latent/LandmarkList";
import { PathEditor } from "../latent/PathEditor";

export const InspectorPanel: React.FC = () => {
  return (
    <div className="flex h-full flex-col gap-3 p-3">
      <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
        Inspector
      </h2>
      <div className="flex flex-1 flex-col gap-3 overflow-y-auto pb-2">
        <InputExcerptCard />
        <OrchestrationSummary />
        <PlaybackPanel />
        <LandmarkList />
        <PathEditor />
      </div>
    </div>
  );
};

