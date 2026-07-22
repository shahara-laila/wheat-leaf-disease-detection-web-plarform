"""Wheat leaf disease detection - web API.

Two features:
  1. A photo of a leaf is classified by the trained CNN (Brown Rust, Healthy,
     Yellow Rust).
  2. Typed symptoms are matched against the disease list in diseases.json.

Start the server with:  uvicorn app:app --reload
"""

import json
import re
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

ROOT = Path(__file__).parent

# The CNN was trained on 256x256 RGB images with pixels scaled to 0-1, and its
# three outputs come out in this order. Both must match the training notebook.
IMAGE_SIZE = (256, 256)
CLASS_NAMES = ["Brown_Rust", "Healthy", "Yellow_Rust"]

# Below this confidence we warn the user instead of trusting the answer.
CONFIDENCE_THRESHOLD = 0.70

diseases = json.loads((ROOT / "diseases.json").read_text(encoding="utf-8"))

# The model file is not in the repository (26 MB), so loading is allowed to
# fail: the symptom search and the disease list work without it.
try:
    from tensorflow import keras

    model = keras.models.load_model(ROOT / "wheat_disease_cnn.keras")
except Exception as error:
    model = None
    print(f"Model not loaded, /predict is disabled: {error}")

app = FastAPI(title="Wheat Leaf Disease Detection")


class SymptomText(BaseModel):
    text: str


@app.get("/health")
def health():
    """Used by the web page to see whether the model is available."""
    return {"model_loaded": model is not None, "disease_count": len(diseases)}


@app.get("/diseases")
def list_diseases():
    """The whole knowledge base: 13 diseases plus Healthy."""
    return diseases


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """Classify one leaf photo.

    The upload is kept in memory only - it is never written to disk.
    """
    if model is None:
        raise HTTPException(503, "The model is not loaded on the server.")

    uploaded_bytes = await image.read()
    try:
        photo = Image.open(BytesIO(uploaded_bytes)).convert("RGB").resize(IMAGE_SIZE)
    except Exception:
        raise HTTPException(400, "That file could not be read as an image.")

    # Same preprocessing as training, then wrapped in a batch of one.
    pixels = np.asarray(photo, dtype="float32") / 255.0
    batch = np.expand_dims(pixels, axis=0)

    probabilities = model.predict(batch, verbose=0)[0]
    best = int(np.argmax(probabilities))
    name = CLASS_NAMES[best]
    confidence = float(probabilities[best])

    return {
        "prediction": name,
        "confidence": confidence,
        "uncertain": confidence < CONFIDENCE_THRESHOLD,
        "probabilities": {
            label: float(value) for label, value in zip(CLASS_NAMES, probabilities)
        },
        "recommendation": diseases[name]["recommendation"],
    }


def contains_word(text, term):
    """True if the term appears in the text as a whole word.

    Whole words only, so "rust" does not match inside "trust".
    """
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


@app.post("/recommend")
def recommend(query: SymptomText):
    """Find every disease whose name or symptoms appear in the typed text."""
    text = query.text.lower()
    matches = []

    for name, info in diseases.items():
        terms = info["common_names"] + info["symptoms"]
        found = [term for term in terms if contains_word(text, term.lower())]
        if found:
            matches.append(
                {
                    "disease": name,
                    "matched_terms": found,
                    "symptoms": info["symptoms"],
                    "recommendation": info["recommendation"],
                }
            )

    return {"matches": matches}


# Mounted last so it cannot hide the routes above.
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
