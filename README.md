# Wheat Leaf Disease Detection — Web Application

FastAPI backend + vanilla HTML/CSS/JS frontend implementing **Step 6 (web integration)**
and **Step 7 (privacy)** of Abbasi et al. (2026), *Food Science & Nutrition*.

Two independent capabilities, exactly as the paper describes:

| | Covers | Powered by |
|---|---|---|
| **Image detection** | Brown Rust · Healthy · Yellow Rust | 6-conv CNN, val accuracy **0.9891** |
| **Symptom → treatment** | 13 diseases + Healthy | Regex knowledge base, no model needed |

The knowledge base is deliberately broader than the classifier. The paper's Section 3.3
lists 13 diseases for the recommendation module while the CNN only distinguishes 3, so
the Browse tab badges the three that are "detectable by photo".

## Setup

TensorFlow has **no Python 3.14 wheels**. Build the venv with 3.13 explicitly — using the
default `python3` will fail to resolve.

```bash
cd neural-web-project
/opt/homebrew/bin/python3.13 -m venv venv
./venv/bin/pip install -r requirements.txt
./run.sh
```

Open <http://localhost:8000>. Interactive API docs at <http://localhost:8000/docs>.

`wheat_disease_cnn.keras` must sit in the project root (or set `WHEAT_MODEL_PATH`). It is
gitignored — 26 MB does not belong in version control.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + model status. Always 200, `status` is `ok` or `degraded`. |
| GET | `/diseases` | All 14 knowledge-base entries. |
| POST | `/predict` | multipart, field `image` → class, confidence, per-class probabilities, treatment. |
| POST | `/recommend` | JSON `{"text": "..."}` → matching diseases with treatments. |

```bash
curl -s localhost:8000/health | python3 -m json.tool
curl -s -F image=@leaf.jpg localhost:8000/predict | python3 -m json.tool
curl -s -X POST localhost:8000/recommend \
  -H 'Content-Type: application/json' -d '{"text":"yellow stripes on the leaves"}'
```

Errors: `400` missing image field · `413` over 10 MB · `422` undecodable image or text over
2000 chars · `503` model not loaded.

## Design notes

**Graceful degradation.** A missing knowledge base is fatal (it ships in the repo, so a
broken one means a broken deploy). A missing model is *not* — the app still starts, `/health`
reports `degraded`, `/predict` returns a clean 503, and the Symptom and Browse tabs keep
working. The frontend disables the upload control and explains why. The original Flask
version loaded the model at import and took the whole process down instead.

**Model loading.** Once, in the FastAPI lifespan handler, with a warmup inference so no user
request pays the graph-tracing cost. Inference is serialized behind a lock (a single Keras
model is not safe under concurrent `predict`) and dispatched to a threadpool so it never
blocks the event loop.

**Preprocessing is frozen.** `RGB → resize → /255 → batch`, matching the notebook exactly.
Verified numerically identical: the API and the notebook produce bit-for-bit the same
probabilities (max delta `0.00e+00`). Any change here silently degrades accuracy without
raising, so re-run that comparison if you touch `inference.py`.

## Two deliberate deviations from the notebook

**1. Shared symptoms return every candidate.** The notebook indexed `term → disease` in a
plain dict, so a term claimed by several diseases kept only the last. `'stunting'` belongs to
BYDV, WSBMV *and* WSMV, but only WSMV survived. The index is now `term → [diseases]`, and the
UI flags the result as ambiguous rather than implying a diagnosis.

**2. Short acronyms must be uppercase.** The knowledge base contains the paper's acronyms
(`yr`, `br`, `pm`, `slb`, `ts`, `hb`, `ls`, `kb`, `rr`, `lb`). Matched case-insensitively they
fire on ordinary prose — *"we applied 50 lb of seed"* read as Leaf Blight. Terms of 3
characters or fewer now require uppercase in the original text, so `YR` matches and `50 lb`
does not.

## Known limitation: no out-of-distribution rejection

The model returns a confident answer for **any** image. Feeding it pure random noise yields
`Healthy` at **100.0%** confidence — the `uncertain` flag does not fire, because softmax
confidence is uncalibrated off-distribution.

The `CONFIDENCE_THRESHOLD` catches genuinely *ambiguous wheat photos*. It is **not** a
"is this a wheat leaf?" detector and must not be presented as one. That is why the UI carries
a permanent caveat instead of relying on the threshold. Real fixes, in ascending cost:
temperature scaling on the val set → an energy/Mahalanobis OOD score → a fourth
"not a wheat leaf" training class.

## Privacy (paper Step 7)

Enforced in two places, both marked in the source:

- **Images are never persisted.** `/predict` holds the upload in a local `bytes` — no
  `UploadFile.save`, no temp file, no disk write. It goes out of scope when the handler
  returns. The frontend revokes its object URL on clear/replace.
- **Query text is never logged.** `/predict` logs the class only; `/recommend` logs a match
  count only.

Verified: file count unchanged across requests, and a unique marker string in a query never
appeared in the log.

Before any public deployment, add authentication, rate limiting, and HTTPS, and tighten
`ALLOWED_ORIGINS`. `--reload` is dev-only (it reloads the 26 MB model on every save), and the
server should stay at **one worker** — each additional worker loads its own model and full TF
runtime for no throughput gain, since inference is already serialized.
