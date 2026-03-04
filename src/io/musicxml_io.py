# src/io/musicxml_io.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from music21 import converter, note, chord, tempo as m21tempo, meter as m21meter


@dataclass
class NoteEvent:
    start_s: float
    end_s: float
    pitches: List[int]
    velocity01: float = 1.0


def infer_tempo_and_timesig(
    xml_path: str,
    default_bpm: float = 120.0,
    default_time_signature: str = "4/4",
) -> tuple[float, str]:
    """
    Infer first tempo mark + first time signature from MusicXML.
    (If the piece has tempo changes, we still export a single initial tempo for now.)
    """
    score = converter.parse(xml_path)

    bpm = float(default_bpm)
    for el in score.recurse():
        if isinstance(el, m21tempo.MetronomeMark) and el.number is not None:
            bpm = float(el.number)
            break

    ts = default_time_signature
    for el in score.recurse():
        if isinstance(el, m21meter.TimeSignature):
            ts = el.ratioString  # e.g. "4/4"
            break

    return bpm, ts


def load_piano_xml_events(xml_path: str) -> Tuple[List[NoteEvent], float]:
    """
    Parse MusicXML and return events in SECONDS using music21's tempo-aware seconds mapping.
    This handles tempo changes correctly for note timing.
    """
    score = converter.parse(xml_path)

    parts = score.parts
    s = parts[0] if len(parts) > 0 else score

    events: List[NoteEvent] = []
    end_s = 0.0

    flat_notes = list(s.flatten().notes)

    # Build a map from element id -> (offsetSeconds, endTimeSeconds)
    sec_by_obj: dict[int, tuple[float, float]] = {}
    for d in s.secondsMap:
        el = d.get("element", None)
        if el is None:
            continue
        os = d.get("offsetSeconds", None)
        es = d.get("endTimeSeconds", None)
        if os is None or es is None:
            continue
        sec_by_obj[id(el)] = (float(os), float(es))

    for el in flat_notes:
        if id(el) in sec_by_obj:
            start_s, end_s_local = sec_by_obj[id(el)]
        else:
            # fallback (rare); offsets are not truly seconds here
            start_s = float(el.offset)
            end_s_local = start_s + float(el.duration.quarterLength)

        end_s = max(end_s, end_s_local)

        if isinstance(el, note.Note):
            pitches = [int(el.pitch.midi)]
        elif isinstance(el, chord.Chord):
            pitches = [int(p.midi) for p in el.pitches]
        else:
            continue

        events.append(NoteEvent(start_s=start_s, end_s=end_s_local, pitches=pitches, velocity01=1.0))

    return events, float(end_s)


def events_to_roll_and_onset(
    events: List[NoteEvent], duration_s: float, hop_s: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build roll/onset from events (seconds).
    Uses ROUND for frame conversion (consistent with src/grid.py).
    """
    T = int(np.ceil(duration_s / hop_s)) if duration_s > 0 else 0
    roll = np.zeros((T, 128), dtype=np.float32)
    onset = np.zeros((T, 128), dtype=np.float32)

    def t2f(t: float) -> int:
        return int(np.round(t / hop_s))

    for ev in events:
        if T == 0:
            continue
        f0 = t2f(ev.start_s)
        f1 = t2f(ev.end_s)

        f0 = max(0, min(T - 1, f0))
        f1 = max(0, min(T - 1, f1))
        if f1 < f0:
            f1 = f0

        for p in ev.pitches:
            if 0 <= p < 128:
                roll[f0 : f1 + 1, p] = np.maximum(roll[f0 : f1 + 1, p], ev.velocity01)
                onset[f0, p] = max(onset[f0, p], ev.velocity01)

    return roll, onset