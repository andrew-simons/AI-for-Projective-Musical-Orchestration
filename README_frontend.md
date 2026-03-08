## Orchestration Latent Space Explorer Frontend

This repository now includes a small web frontend for interactive exploration of the orchestration model’s latent space.

### How the frontend works

- The frontend is a React + TypeScript + Tailwind app in `frontend/`.
- It presents a **2D latent space canvas** for exploration plus a right-side **inspector panel**.
- Users can:
  - Upload a **piano MusicXML** excerpt.
  - See the excerpt’s **encoded latent position** in a 2D PCA projection of orchestration profiles.
  - Drag or click in the 2D space to request **orchestration previews** at new latent locations.
  - Save **landmarks** (named latent points) and **define paths** through latent space for interpolation.

### How it connects to the backend

The backend is a FastAPI app in `backend/app.py`. It exposes typed JSON APIs used by the frontend:

- `POST /api/encode-excerpt`
  - Input: `multipart/form-data` with `piano_xml` (MusicXML file).
  - Pipeline:
    - Parses MusicXML via `src/io/musicxml_io.py`.
    - Builds piano roll and onset features.
    - Runs the Stage‑1 encoder (`src/inference/stage1.py`) to get framewise instrument activity `(T, 129)`.
    - Averages per-instrument activity to form a **129‑D latent vector**.
    - Projects that vector into 2D via PCA (`src/latent/space.py`).
  - Output: excerpt ID, piano summary, latent coordinates, and an instrument activity summary.

- `POST /api/generate-from-latent`
  - Input: JSON `{ excerpt_id, coords_2d: {x, y} }`.
  - Pipeline:
    - Maps 2D coords back into the PCA subspace to get a **style vector**.
    - Blends the style vector with the excerpt’s own latent profile.
    - Modulates per-frame instrument activity and calls `src/render/assign.assign_events_to_parts`.
    - Writes orchestral MusicXML via `src/render/write_xml.write_orchestral_musicxml`.
    - Builds a simple MIDI preview via `_parts_to_pretty_midi`.
  - Output: orchestration metadata plus `musicxml_base64` and `midi_base64` for playback/export.

- `GET /api/latent/samples`
  - Uses `data/features/meta/features_index.csv` and `src/features/orch.py`‑derived NPZ files to:
    - Compute mean instrument activity vectors for many pieces.
    - Fit a **2D PCA** latent space using `src/latent/space.build_latent_space_from_index`.
  - Returns background sample points for the canvas plus x/y ranges.

- `GET /api/landmarks`, `POST /api/landmarks`, `DELETE /api/landmarks/{id}`
  - In-memory storage for user-defined **landmarks** in 2D latent space (names + coordinates).

- `POST /api/latent/path-samples`
  - Input: control points in 2D + desired number of samples.
  - Output: evenly spaced **interpolated samples** along the polyline path; the frontend can query orchestration along this path.

The frontend never calls `fetch` directly; all HTTP calls go through the typed API layer in `frontend/src/api/`.

### Frontend routes and state

Key frontend pieces:

- `frontend/src/api/types.ts` – TypeScript interfaces mirroring backend response models.
- `frontend/src/api/client.ts` – `apiFetch` wrapper and base64 → Blob URL helper.
- `frontend/src/api/orchestrationApi.ts` – high-level client for the orchestration API.
- `frontend/src/state/useExplorerStore.ts` – Zustand store that:
  - Manages interaction state (`idle`, `dragging`, `loading_generation`, `generation_ready`, `error`).
  - Holds the current excerpt, latent coordinates, landmarks, path, and orchestration preview.
  - Implements **debounced generation** when the active point moves.

Main UI components:

- `AppShell` – top-level layout (top bar, latent explorer, inspector).
- `TopBar` – project heading.
- `LatentSpaceExplorer` – orchestration map area:
  - `LatentCanvas` – 2D canvas with pan/zoom, background samples, active point, landmarks, paths.
  - `ActivePointOverlay` – shows current coords + interaction status.
- `InspectorPanel` – right-side inspector:
  - `InputExcerptCard` – piano input summary.
  - `OrchestrationSummary` – instrument activity bars.
  - `PlaybackPanel` – MusicXML/MIDI download controls.
  - `LandmarkList` – save and restore landmarks.
  - `PathEditor` – define and sample paths in latent space.

### Running the system

1. **Backend**
   - Create/activate the conda environment:
     - `conda env create -f environment.yml`
     - `conda activate projective_orchestration`
   - Ensure a Stage‑1 checkpoint is available (e.g. `checkpoints/stage1_encoder_v2/best.pt`).
     - Optionally set `STAGE1_CHECKPOINT` to override.
   - Start the FastAPI app:
     - `uvicorn backend.app:app --reload`

2. **Frontend**
   - In `frontend/`:
     - `npm install`
     - `npm run dev`
   - The app assumes the backend is at `http://localhost:8000` by default. Override with:
     - `VITE_API_BASE_URL=http://your-backend:8000 npm run dev`

### Where to extend model integration

- **Latent definition and projection**
  - Implemented in `src/latent/space.py`:
    - To change what “latent vector” means (e.g. use a deeper model embedding instead of mean instrument activity), adjust:
      - `_load_instrument_activity_vector` and `build_latent_space_from_index`.
    - To use a different projection (e.g. t‑SNE, UMAP), replace `_fit_pca_2d` and the `LatentSpace.project` / `LatentSpace.invert` methods.

- **Serving orchestration from latent positions**
  - Implemented in `backend/app.py` in `generate_from_latent`:
    - To alter how style mixing works, change the computation of `z_mix` and `gains`.
    - To incorporate additional model stages, re-use or extend the Stage‑1 outputs before passing into `assign_events_to_parts`.

- **Frontend behaviour**
  - All high-level orchestration interactions are mediated through:
    - `frontend/src/state/useExplorerStore.ts` (state transitions and API calls).
    - `frontend/src/api/orchestrationApi.ts` (HTTP integration).
  - For new backend endpoints, add typed functions in `orchestrationApi.ts` and corresponding state/actions in the store before wiring them into UI components.

