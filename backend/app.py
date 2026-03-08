from __future__ import annotations

from dataclasses import dataclass
import base64
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pretty_midi
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.musicxml_io import (  # noqa: E402
    NoteEvent,
    events_to_roll_and_onset,
    infer_tempo_and_timesig,
    load_piano_xml_events,
)
from src.inference.stage1 import (  # noqa: E402
    Stage1ModelBundle,
    load_stage1_model,
    predict_instrument_activity_chunked,
)
from src.latent.space import (  # noqa: E402
    LatentPoint,
    LatentSpace,
    build_latent_space_from_index,
)
from src.render.assign import (  # noqa: E402
    DEFAULT_PARTS,
    PartSpec,
    assign_events_to_parts,
)
from src.render.write_xml import write_orchestral_musicxml  # noqa: E402


FEATURE_INDEX = ROOT / "data" / "features" / "meta" / "features_index.csv"
DEFAULT_HOP_S = 0.05


@dataclass
class ExcerptSession:
    id: str
    piano_xml_path: Path
    bpm: float
    time_signature: str
    events: List[NoteEvent]
    duration_s: float
    hop_s: float
    instrument_activity: np.ndarray  # (T,129)
    latent_vector: np.ndarray  # (129,)
    coords_2d: np.ndarray  # (2,)


@dataclass
class Landmark:
    id: str
    name: str
    x: float
    y: float


LATENT_SPACE: Optional[LatentSpace] = None
STAGE1_BUNDLE: Optional[Stage1ModelBundle] = None
EXCERPTS: Dict[str, ExcerptSession] = {}
LANDMARKS: Dict[str, Landmark] = {}


class LatentCoords(BaseModel):
    x: float
    y: float


class InstrumentTopItem(BaseModel):
    program: int
    score: float


class InstrumentActivitySummary(BaseModel):
    mean: List[float]
    top: List[InstrumentTopItem]


class PianoSummary(BaseModel):
    duration_s: float
    num_events: int
    bpm: float
    time_signature: str
    filename: Optional[str] = None


class LatentSummary(BaseModel):
    coords_2d: LatentCoords
    vector_129: List[float]


class EncodeResponse(BaseModel):
    excerpt_id: str
    piano_summary: PianoSummary
    latent: LatentSummary
    instrument_activity_summary: InstrumentActivitySummary


class GenerateRequest(BaseModel):
    excerpt_id: str
    coords_2d: LatentCoords


class OrchestrationAssets(BaseModel):
    musicxml_base64: str
    midi_base64: Optional[str] = None


class GenerateResponse(BaseModel):
    excerpt_id: str
    coords_2d: LatentCoords
    piano_summary: PianoSummary
    instrument_activity_summary: InstrumentActivitySummary
    orchestration: OrchestrationAssets


class LatentSampleItem(BaseModel):
    id: str
    x: float
    y: float
    label: Optional[str] = None
    meta: Dict[str, str] = {}


class LatentSamplesResponse(BaseModel):
    samples: List[LatentSampleItem]
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]


class LandmarkResponse(BaseModel):
    id: str
    name: str
    x: float
    y: float


class LandmarkCreate(BaseModel):
    name: str
    coords_2d: LatentCoords


class PathSamplesRequest(BaseModel):
    points: List[LatentCoords]
    num_samples: int = 32


class PathSample(BaseModel):
    t: float
    coords_2d: LatentCoords


class PathSamplesResponse(BaseModel):
    samples: List[PathSample]


app = FastAPI(title="Projective Orchestration Explorer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_latent_space() -> LatentSpace:
    global LATENT_SPACE
    if LATENT_SPACE is None:
        if not FEATURE_INDEX.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Feature index CSV not found at {FEATURE_INDEX}",
            )
        try:
            LATENT_SPACE = build_latent_space_from_index(FEATURE_INDEX, root=ROOT)
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=f"Failed to build latent space: {exc}") from exc
    return LATENT_SPACE


def _ensure_stage1_bundle() -> Stage1ModelBundle:
    global STAGE1_BUNDLE
    if STAGE1_BUNDLE is None:
        try:
            STAGE1_BUNDLE = load_stage1_model()
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=f"Failed to load Stage-1 model: {exc}") from exc
    return STAGE1_BUNDLE


def _summarise_instrument_activity(
    activity: np.ndarray,
    topk: int = 8,
) -> InstrumentActivitySummary:
    if activity.size == 0:
        mean = np.zeros((129,), dtype=np.float32)
    else:
        mean = activity.mean(axis=0).astype(np.float32)

    order = np.argsort(-mean)[:topk]
    top_items: List[InstrumentTopItem] = []
    for i in order:
        score = float(mean[int(i)])
        if score <= 0.0:
            continue
        top_items.append(InstrumentTopItem(program=int(i), score=score))

    return InstrumentActivitySummary(mean=mean.tolist(), top=top_items)


def _parts_to_pretty_midi(
    parts_to_events: Dict[str, List[NoteEvent]],
    parts_spec: List[PartSpec],
    bpm: float,
) -> pretty_midi.PrettyMIDI:
    spec_by_name = {p.name: p for p in parts_spec}
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))

    for part_name, events in parts_to_events.items():
        spec = spec_by_name.get(part_name)
        program = int(spec.gm_program) if spec is not None else 0
        inst = pretty_midi.Instrument(program=program, is_drum=False, name=part_name)

        for ev in events:
            if not ev.pitches:
                continue
            pitch = int(ev.pitches[0])
            start = float(ev.start_s)
            end = float(ev.end_s)
            if end <= start:
                end = start + 1e-3
            vel = int(max(1, min(127, round(ev.velocity01 * 127.0))))
            inst.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end))

        if inst.notes:
            pm.instruments.append(inst)

    return pm


@app.on_event("startup")
def _startup() -> None:
    """
    Lazily build latent space and load model when the server starts.
    Errors are logged but exposed to clients on first use.
    """
    try:
        _ensure_latent_space()
    except HTTPException as exc:
        print(f"[warn] latent space init failed: {exc.detail}")
    try:
        _ensure_stage1_bundle()
    except HTTPException as exc:
        print(f"[warn] Stage-1 model init failed: {exc.detail}")


@app.get("/api/latent/samples", response_model=LatentSamplesResponse)
def get_latent_samples(limit: int = 256) -> LatentSamplesResponse:
    space = _ensure_latent_space()
    pts: List[LatentPoint] = space.points[: max(0, int(limit))]
    (xmin, xmax), (ymin, ymax) = space.bounds()

    samples = [
        LatentSampleItem(
            id=p.id,
            x=float(p.coords_2d[0]),
            y=float(p.coords_2d[1]),
            label=p.label,
            meta=p.meta or {},
        )
        for p in pts
    ]
    return LatentSamplesResponse(samples=samples, x_range=(xmin, xmax), y_range=(ymin, ymax))


@app.post("/api/encode-excerpt", response_model=EncodeResponse)
async def encode_excerpt(piano_xml: UploadFile = File(...)) -> EncodeResponse:
    """
    Encode a piano MusicXML excerpt into latent space using the Stage-1 encoder.

    Returns the excerpt's latent coordinates, a summary of the piano input,
    and an instrument activity summary derived from the encoder outputs.
    """
    space = _ensure_latent_space()
    bundle = _ensure_stage1_bundle()

    contents = await piano_xml.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload.")

    with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        bpm, time_signature = infer_tempo_and_timesig(str(tmp_path))
        events, duration_s = load_piano_xml_events(str(tmp_path))
        roll, onset = events_to_roll_and_onset(events, duration_s=duration_s, hop_s=DEFAULT_HOP_S)

        instrument_activity = predict_instrument_activity_chunked(
            bundle=bundle,
            roll=roll,
            onset=onset,
            batch_chunk=512,
        )
        if instrument_activity.shape[0] == 0:
            latent_vec = np.zeros((129,), dtype=np.float32)
        else:
            latent_vec = instrument_activity.mean(axis=0).astype(np.float32)

        coords = space.project(latent_vec)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to encode excerpt: {exc}") from exc

    excerpt_id = uuid.uuid4().hex
    EXCERPTS[excerpt_id] = ExcerptSession(
        id=excerpt_id,
        piano_xml_path=tmp_path,
        bpm=float(bpm),
        time_signature=str(time_signature),
        events=events,
        duration_s=float(duration_s),
        hop_s=float(DEFAULT_HOP_S),
        instrument_activity=instrument_activity,
        latent_vector=latent_vec,
        coords_2d=coords,
    )

    piano_summary = PianoSummary(
        duration_s=float(duration_s),
        num_events=len(events),
        bpm=float(bpm),
        time_signature=str(time_signature),
        filename=piano_xml.filename,
    )
    latent_summary = LatentSummary(
        coords_2d=LatentCoords(x=float(coords[0]), y=float(coords[1])),
        vector_129=latent_vec.tolist(),
    )
    instr_summary = _summarise_instrument_activity(instrument_activity)

    return EncodeResponse(
        excerpt_id=excerpt_id,
        piano_summary=piano_summary,
        latent=latent_summary,
        instrument_activity_summary=instr_summary,
    )


@app.post("/api/generate-from-latent", response_model=GenerateResponse)
def generate_from_latent(req: GenerateRequest) -> GenerateResponse:
    """
    Generate an orchestration preview from a latent 2D position for a given excerpt.

    The 2D point is mapped back into the PCA latent subspace and combined with the
    excerpt's own latent profile to modulate instrument activity over time.
    """
    space = _ensure_latent_space()
    if req.excerpt_id not in EXCERPTS:
        raise HTTPException(status_code=404, detail="Unknown excerpt_id.")

    session = EXCERPTS[req.excerpt_id]

    # Map 2D -> latent, then mix with excerpt latent to get style vector
    z_style = space.invert((req.coords_2d.x, req.coords_2d.y))
    z_excerpt = session.latent_vector

    alpha = 0.6
    z_mix = (1.0 - alpha) * z_excerpt + alpha * z_style

    z_min = float(z_mix.min())
    z_max = float(z_mix.max())
    if z_max - z_min < 1e-6:
        gains = np.ones_like(z_mix, dtype=np.float32)
    else:
        gains = (z_mix - z_min) / (z_max - z_min + 1e-6)
        gains = np.clip(gains, 0.1, 1.0).astype(np.float32)

    instrument_activity_mod = session.instrument_activity * gains[None, :]

    try:
        parts_to_events = assign_events_to_parts(
            events=session.events,
            instrument_activity_hat=instrument_activity_mod,
            hop_s=session.hop_s,
            parts=DEFAULT_PARTS,
        )

        # Write MusicXML to bytes
        with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as tmp_xml:
            write_orchestral_musicxml(
                parts_to_events=parts_to_events,
                out_xml_path=str(tmp_xml.name),
                bpm=session.bpm,
                time_signature=session.time_signature,
                quantize_denom=96,
                non_transposing=True,
            )
            xml_bytes = Path(tmp_xml.name).read_bytes()

        musicxml_b64 = base64.b64encode(xml_bytes).decode("ascii")

        # MIDI preview (optional but useful)
        pm = _parts_to_pretty_midi(parts_to_events, DEFAULT_PARTS, bpm=session.bpm)
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_midi:
            pm.write(tmp_midi.name)
            midi_bytes = Path(tmp_midi.name).read_bytes()
        midi_b64 = base64.b64encode(midi_bytes).decode("ascii")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate orchestration: {exc}") from exc

    instr_summary = _summarise_instrument_activity(instrument_activity_mod)
    piano_summary = PianoSummary(
        duration_s=session.duration_s,
        num_events=len(session.events),
        bpm=session.bpm,
        time_signature=session.time_signature,
        filename=session.piano_xml_path.name,
    )

    return GenerateResponse(
        excerpt_id=session.id,
        coords_2d=req.coords_2d,
        piano_summary=piano_summary,
        instrument_activity_summary=instr_summary,
        orchestration=OrchestrationAssets(
            musicxml_base64=musicxml_b64,
            midi_base64=midi_b64,
        ),
    )


@app.get("/api/landmarks", response_model=List[LandmarkResponse])
def list_landmarks() -> List[LandmarkResponse]:
    return [LandmarkResponse(id=l.id, name=l.name, x=l.x, y=l.y) for l in LANDMARKS.values()]


@app.post("/api/landmarks", response_model=LandmarkResponse)
def create_landmark(body: LandmarkCreate) -> LandmarkResponse:
    lid = uuid.uuid4().hex
    landmark = Landmark(id=lid, name=body.name, x=body.coords_2d.x, y=body.coords_2d.y)
    LANDMARKS[lid] = landmark
    return LandmarkResponse(id=lid, name=landmark.name, x=landmark.x, y=landmark.y)


@app.delete("/api/landmarks/{landmark_id}", response_model=None)
def delete_landmark(landmark_id: str) -> None:
    if landmark_id in LANDMARKS:
        del LANDMARKS[landmark_id]


@app.post("/api/latent/path-samples", response_model=PathSamplesResponse)
def sample_path(body: PathSamplesRequest) -> PathSamplesResponse:
    """
    Given a sequence of control points in 2D, return evenly spaced samples
    along the piecewise-linear path.
    """
    pts = body.points
    num_samples = max(1, int(body.num_samples))

    if len(pts) == 0:
        return PathSamplesResponse(samples=[])
    if len(pts) == 1:
        only = pts[0]
        return PathSamplesResponse(
            samples=[
                PathSample(t=0.0, coords_2d=only),
                PathSample(t=1.0, coords_2d=only),
            ]
        )

    xs = np.array([p.x for p in pts], dtype=np.float32)
    ys = np.array([p.y for p in pts], dtype=np.float32)

    seg_lengths: List[float] = []
    for i in range(len(pts) - 1):
        dx = float(xs[i + 1] - xs[i])
        dy = float(ys[i + 1] - ys[i])
        seg_lengths.append(float(np.hypot(dx, dy)))

    total_len = sum(seg_lengths)
    if total_len <= 1e-8:
        # Degenerate path: all points are the same
        base = pts[0]
        return PathSamplesResponse(
            samples=[
                PathSample(t=float(i) / max(1, num_samples - 1), coords_2d=base)
                for i in range(num_samples)
            ]
        )

    seg_starts = [0.0]
    acc = 0.0
    for L in seg_lengths:
        acc += L
        seg_starts.append(acc)

    samples: List[PathSample] = []
    for i in range(num_samples):
        t = float(i) / max(1, num_samples - 1)
        target_dist = t * total_len

        # Find segment
        seg_idx = len(seg_lengths) - 1
        for j in range(len(seg_lengths)):
            if seg_starts[j] <= target_dist <= seg_starts[j + 1]:
                seg_idx = j
                break

        seg_start = seg_starts[seg_idx]
        seg_len = seg_lengths[seg_idx]
        local_t = 0.0 if seg_len <= 1e-8 else (target_dist - seg_start) / seg_len

        x = float(xs[seg_idx] + local_t * (xs[seg_idx + 1] - xs[seg_idx]))
        y = float(ys[seg_idx] + local_t * (ys[seg_idx + 1] - ys[seg_idx]))

        samples.append(PathSample(t=t, coords_2d=LatentCoords(x=x, y=y)))

    return PathSamplesResponse(samples=samples)


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}

