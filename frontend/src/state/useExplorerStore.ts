import { create } from "zustand";
import {
  encodeExcerpt,
  fetchLatentSamples,
  fetchLandmarks,
  createLandmark as apiCreateLandmark,
  deleteLandmark as apiDeleteLandmark,
  generateFromLatent,
  interpolatePath,
} from "../api/orchestrationApi";
import type {
  EncodeResponse,
  GenerateResponse,
  InteractionState,
  LatentCoords,
  LatentSampleItem,
  Landmark,
  PathSample,
  InstrumentActivitySummary,
} from "../api/types";
import { base64ToBlobUrl } from "../api/client";

interface OrchestrationPreview {
  musicxmlUrl?: string;
  midiUrl?: string;
  summary?: InstrumentActivitySummary;
}

interface ExplorerState {
  interactionState: InteractionState;
  errorMessage: string | null;

  excerpt: EncodeResponse | null;
  activePoint: LatentCoords | null;
  lastGeneratedCoords: LatentCoords | null;

  samples: LatentSampleItem[];
  xRange: [number, number] | null;
  yRange: [number, number] | null;

  landmarks: Landmark[];

  pathControlPoints: { id: string; x: number; y: number }[];
  pathSamples: PathSample[];

  orchestrationPreview: OrchestrationPreview | null;

  // actions
  initialize: () => Promise<void>;
  uploadExcerpt: (file: File) => Promise<void>;
  setActivePoint: (coords: LatentCoords) => void;
  moveActivePoint: (coords: LatentCoords) => void;
  finishDrag: () => void;
  generateAtPointNow: (coords: LatentCoords) => Promise<void>;

  addLandmarkAtActive: (name: string) => Promise<void>;
  removeLandmark: (id: string) => Promise<void>;
  jumpToLandmark: (id: string) => Promise<void>;

  addPathPointFromActive: () => void;
  clearPath: () => void;
  samplePath: (numSamples: number) => Promise<void>;

  clearError: () => void;
}

let generationDebounceHandle: number | null = null;

function buildPreviewFromResponse(resp: GenerateResponse): OrchestrationPreview {
  const { musicxml_base64, midi_base64 } = resp.orchestration;
  let musicxmlUrl: string | undefined;
  let midiUrl: string | undefined;

  try {
    musicxmlUrl = base64ToBlobUrl(
      musicxml_base64,
      "application/vnd.recordare.musicxml+xml"
    );
  } catch {
    musicxmlUrl = undefined;
  }
  if (midi_base64) {
    try {
      midiUrl = base64ToBlobUrl(midi_base64, "audio/midi");
    } catch {
      midiUrl = undefined;
    }
  }

  return {
    musicxmlUrl,
    midiUrl,
    summary: resp.instrument_activity_summary,
  };
}

export const useExplorerStore = create<ExplorerState>((set, get) => ({
  interactionState: "idle",
  errorMessage: null,

  excerpt: null,
  activePoint: null,
  lastGeneratedCoords: null,

  samples: [],
  xRange: null,
  yRange: null,

  landmarks: [],

  pathControlPoints: [],
  pathSamples: [],

  orchestrationPreview: null,

  initialize: async () => {
    try {
      const [samplesRes, landmarks] = await Promise.all([
        fetchLatentSamples(),
        fetchLandmarks(),
      ]);
      set({
        samples: samplesRes.samples,
        xRange: samplesRes.x_range,
        yRange: samplesRes.y_range,
        landmarks,
      });
    } catch (err) {
      set({
        errorMessage:
          err instanceof Error ? err.message : "Failed to initialize explorer",
        interactionState: "error",
      });
    }
  },

  uploadExcerpt: async (file: File) => {
    set({
      interactionState: "loading_generation",
      errorMessage: null,
      orchestrationPreview: null,
    });
    try {
      const resp = await encodeExcerpt(file);
      const activePoint = resp.latent.coords_2d;
      set({
        excerpt: resp,
        activePoint,
        lastGeneratedCoords: null,
        interactionState: "idle",
        errorMessage: null,
        pathControlPoints: [],
        pathSamples: [],
      });
    } catch (err) {
      set({
        interactionState: "error",
        errorMessage:
          err instanceof Error ? err.message : "Failed to encode excerpt",
      });
    }
  },

  setActivePoint: (coords: LatentCoords) => {
    set({
      activePoint: coords,
      interactionState: "idle",
    });
  },

  moveActivePoint: (coords: LatentCoords) => {
    set({
      activePoint: coords,
      interactionState: "dragging",
    });

    if (generationDebounceHandle !== null) {
      window.clearTimeout(generationDebounceHandle);
    }

    generationDebounceHandle = window.setTimeout(async () => {
      const { excerpt } = get();
      if (!excerpt) return;

      set({
        interactionState: "loading_generation",
        errorMessage: null,
      });

      try {
        const resp = await generateFromLatent(excerpt.excerpt_id, coords);
        const preview = buildPreviewFromResponse(resp);
        set({
          orchestrationPreview: preview,
          lastGeneratedCoords: coords,
          interactionState: "generation_ready",
          errorMessage: null,
        });
      } catch (err) {
        set({
          interactionState: "error",
          errorMessage:
            err instanceof Error ? err.message : "Failed to generate orchestration",
        });
      }
    }, 350);
  },

  finishDrag: () => {
    const { interactionState } = get();
    if (interactionState === "dragging") {
      set({ interactionState: "idle" });
    }
  },

  generateAtPointNow: async (coords: LatentCoords) => {
    const { excerpt } = get();
    if (!excerpt) return;
    set({
      interactionState: "loading_generation",
      errorMessage: null,
    });
    try {
      const resp = await generateFromLatent(excerpt.excerpt_id, coords);
      const preview = buildPreviewFromResponse(resp);
      set({
        orchestrationPreview: preview,
        lastGeneratedCoords: coords,
        interactionState: "generation_ready",
        errorMessage: null,
        activePoint: coords,
      });
    } catch (err) {
      set({
        interactionState: "error",
        errorMessage:
          err instanceof Error ? err.message : "Failed to generate orchestration",
      });
    }
  },

  addLandmarkAtActive: async (name: string) => {
    const { activePoint } = get();
    if (!activePoint || !name.trim()) return;
    try {
      const lm = await apiCreateLandmark(name.trim(), activePoint);
      set((state) => ({
        landmarks: [...state.landmarks, lm],
      }));
    } catch (err) {
      set({
        interactionState: "error",
        errorMessage:
          err instanceof Error ? err.message : "Failed to save landmark",
      });
    }
  },

  removeLandmark: async (id: string) => {
    try {
      await apiDeleteLandmark(id);
      set((state) => ({
        landmarks: state.landmarks.filter((lm) => lm.id !== id),
      }));
    } catch (err) {
      set({
        interactionState: "error",
        errorMessage:
          err instanceof Error ? err.message : "Failed to delete landmark",
      });
    }
  },

  jumpToLandmark: async (id: string) => {
    const { landmarks } = get();
    const lm = landmarks.find((l) => l.id === id);
    if (!lm) return;
    const coords = { x: lm.x, y: lm.y };
    await get().generateAtPointNow(coords);
  },

  addPathPointFromActive: () => {
    const { activePoint, pathControlPoints } = get();
    if (!activePoint) return;
    const id = `p-${Date.now().toString(36)}`;
    set({
      pathControlPoints: [
        ...pathControlPoints,
        { id, x: activePoint.x, y: activePoint.y },
      ],
    });
  },

  clearPath: () => {
    set({
      pathControlPoints: [],
      pathSamples: [],
    });
  },

  samplePath: async (numSamples: number) => {
    const { pathControlPoints } = get();
    if (pathControlPoints.length === 0) return;
    try {
      const samples = await interpolatePath(
        pathControlPoints.map((p) => ({ x: p.x, y: p.y })),
        numSamples
      );
      set({ pathSamples: samples });
    } catch (err) {
      set({
        interactionState: "error",
        errorMessage:
          err instanceof Error ? err.message : "Failed to sample path",
      });
    }
  },

  clearError: () => set({ errorMessage: null, interactionState: "idle" }),
}));

