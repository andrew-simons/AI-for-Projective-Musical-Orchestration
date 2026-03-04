# src/render/write_xml.py
from __future__ import annotations

from typing import Dict, List
from fractions import Fraction

from music21 import stream, note, instrument, tempo, meter

from src.io.musicxml_io import NoteEvent


def _quantize_ql(x: float, denom: int = 96) -> float:
    """
    Quantize a quarterLength value x to nearest 1/denom.
    """
    if x <= 0:
        return 0.0
    return float(round(x * denom) / denom)


def write_orchestral_musicxml(
    parts_to_events: Dict[str, List[NoteEvent]],
    out_xml_path: str,
    bpm: float = 120.0,
    time_signature: str = "4/4",
    quantize_denom: int = 96,
    non_transposing: bool = True,
) -> None:
    score = stream.Score()
    score.insert(0, tempo.MetronomeMark(number=float(bpm)))
    score.insert(0, meter.TimeSignature(time_signature))

    sec_per_quarter = 60.0 / float(bpm)

    # 64th-note minimum (in quarterLength units)
    min_dur_ql = 1.0 / 16.0

    for part_name, events in parts_to_events.items():
        p = stream.Part()
        p.partName = part_name

        if non_transposing:
            inst_obj = instrument.Instrument()
            inst_obj.partName = part_name
        else:
            inst_obj = instrument.fromString(part_name) if part_name else instrument.Instrument()
        p.insert(0, inst_obj)

        for ev in events:
            start_q = ev.start_s / sec_per_quarter
            dur_q = (ev.end_s - ev.start_s) / sec_per_quarter

            start_q = _quantize_ql(float(start_q), denom=quantize_denom)
            dur_q = _quantize_ql(float(dur_q), denom=quantize_denom)

            if dur_q < _quantize_ql(min_dur_ql, denom=quantize_denom):
                dur_q = _quantize_ql(min_dur_ql, denom=quantize_denom)

            n = note.Note(int(ev.pitches[0]))
            n.offset = start_q
            n.quarterLength = dur_q
            p.insert(n.offset, n)

        # Let music21 lay out measures
        p.makeMeasures(inPlace=True)
        score.append(p)

    score.write("musicxml", fp=out_xml_path)