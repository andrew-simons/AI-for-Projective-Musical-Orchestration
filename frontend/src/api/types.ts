export type InteractionState =
  | "idle"
  | "hovering"
  | "dragging"
  | "loading_generation"
  | "generation_ready"
  | "error";

export interface LatentCoords {
  x: number;
  y: number;
}

export interface InstrumentTopItem {
  program: number;
  score: number;
}

export interface InstrumentActivitySummary {
  mean: number[];
  top: InstrumentTopItem[];
}

export interface PianoSummary {
  duration_s: number;
  num_events: number;
  bpm: number;
  time_signature: string;
  filename?: string | null;
}

export interface LatentSummary {
  coords_2d: LatentCoords;
  vector_129: number[];
}

export interface EncodeResponse {
  excerpt_id: string;
  piano_summary: PianoSummary;
  latent: LatentSummary;
  instrument_activity_summary: InstrumentActivitySummary;
}

export interface OrchestrationAssets {
  musicxml_base64: string;
  midi_base64?: string | null;
}

export interface GenerateResponse {
  excerpt_id: string;
  coords_2d: LatentCoords;
  piano_summary: PianoSummary;
  instrument_activity_summary: InstrumentActivitySummary;
  orchestration: OrchestrationAssets;
}

export interface LatentSampleItem {
  id: string;
  x: number;
  y: number;
  label?: string | null;
  meta: Record<string, string>;
}

export interface LatentSamplesResponse {
  samples: LatentSampleItem[];
  x_range: [number, number];
  y_range: [number, number];
}

export interface Landmark {
  id: string;
  name: string;
  x: number;
  y: number;
}

export interface PathSample {
  t: number;
  coords_2d: LatentCoords;
}

