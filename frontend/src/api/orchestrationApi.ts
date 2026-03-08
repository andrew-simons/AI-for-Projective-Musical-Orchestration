import { apiFetch } from "./client";
import type {
  EncodeResponse,
  GenerateResponse,
  LatentSamplesResponse,
  Landmark,
  LatentCoords,
  PathSample,
} from "./types";

export async function encodeExcerpt(file: File): Promise<EncodeResponse> {
  const form = new FormData();
  form.append("piano_xml", file);
  return apiFetch<EncodeResponse>("/api/encode-excerpt", {
    method: "POST",
    body: form,
  });
}

export async function generateFromLatent(
  excerptId: string,
  coords: LatentCoords
): Promise<GenerateResponse> {
  return apiFetch<GenerateResponse>("/api/generate-from-latent", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      excerpt_id: excerptId,
      coords_2d: coords,
    }),
  });
}

export async function fetchLatentSamples(): Promise<LatentSamplesResponse> {
  return apiFetch<LatentSamplesResponse>("/api/latent/samples?limit=512");
}

export async function fetchLandmarks(): Promise<Landmark[]> {
  return apiFetch<Landmark[]>("/api/landmarks");
}

export async function createLandmark(
  name: string,
  coords: LatentCoords
): Promise<Landmark> {
  return apiFetch<Landmark>("/api/landmarks", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ name, coords_2d: coords }),
  });
}

export async function deleteLandmark(id: string): Promise<void> {
  return apiFetch<void>(`/api/landmarks/${id}`, {
    method: "DELETE",
  });
}

export async function interpolatePath(
  points: LatentCoords[],
  numSamples: number
): Promise<PathSample[]> {
  const res = await apiFetch<{ samples: PathSample[] }>(
    "/api/latent/path-samples",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ points, num_samples: numSamples }),
    }
  );
  return res.samples;
}

