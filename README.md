# Projective Orchestration — Quick Guide

This guide explains how to:

1. Train the Stage-1 model  
2. Generate orchestral MusicXML from a piano score

---

# 1. Setup

Clone the repository and create a Python virtual environment.

```bash
git clone <repo-url>
cd projective_orchestration

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If PyTorch is not included in requirements:

```bash
pip install torch torchvision
```

---

# 2. Train the Model

Stage-1 learns to predict **instrument activity over time** from a piano roll representation.

Run training from the project root:

```bash
python training/train_stage1.py \
  --use_onset \
  --subset_frac 1.0 \
  --chunk_len 512 \
  --batch_size 8 \
  --epochs 5 \
  --run_name stage1_encoder_v2
```

### Training Output

Checkpoints will be saved in:

```
checkpoints/<run_name>/
```

Example:

```
checkpoints/stage1_encoder_v2/
    best.pt
```

Use **best.pt** for orchestration.

---

# 3. Generate Orchestration

To orchestrate a piano score (MusicXML):

```bash
python scripts/20_xml_demo_orchestrate.py tests/input/piano.musicxml \
  --checkpoint checkpoints/stage1_encoder_v2/best.pt \
  --out_xml tests/output/out_orch.musicxml \
  --activity_thresh 0.20
```

This pipeline will:

1. Parse the piano MusicXML file  
2. Convert the score to piano-roll features  
3. Predict instrument activity with the trained model  
4. Assign notes to orchestral instruments  
5. Write a new orchestral MusicXML file

The resulting file can be opened in notation software such as **MuseScore, Dorico, Sibelius, or Finale**.

---

# 4. Useful Parameters

### Activity Threshold

Controls how easily instruments activate.

Higher values → fewer instruments.

Example:

```bash
--activity_thresh 0.25
```

---

### Continuity

Controls how stable the orchestration is.

```
0.0  → instruments change frequently
1.0  → instruments stay active longer
```

Example:

```bash
--continuity 1.0
```

---

### Limit Instrument Density

To reduce the number of simultaneous instruments:

```bash
--topk 2
--max_active_parts_per_frame 4
```

---

# 5. Example Full Run

```bash
python scripts/20_xml_demo_orchestrate.py tests/input/piano.musicxml \
  --checkpoint checkpoints/stage1_encoder_v2/best.pt \
  --out_xml tests/output/out_orch.musicxml \
  --activity_thresh 0.25 \
  --topk 2 \
  --max_active_parts_per_frame 4 \
  --continuity 1.0
```

---

# Input Format

The script expects **MusicXML** files:

```
.musicxml
.xml
```

If your notation program exports `.mxl`, export **uncompressed MusicXML** instead.
