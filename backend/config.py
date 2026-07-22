"""Configuration for the wheat disease detection API.

Every tunable lives here so the rest of the package never hardcodes a path or a
threshold. Paths are resolved relative to the project root (the parent of this
package) so the app works regardless of the current working directory.
"""

import os
from pathlib import Path

# backend/config.py -> backend/ -> project root
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# The trained CNN. Overridable so the model can live outside the repo.
MODEL_PATH = Path(os.getenv("WHEAT_MODEL_PATH", PROJECT_ROOT / "wheat_disease_cnn.keras"))

# Knowledge base exported from the notebook (13 diseases + Healthy).
DB_PATH = Path(os.getenv("WHEAT_DB_PATH", BACKEND_DIR / "data" / "recommendation_db.json"))

# Model output index i maps to CLASS_NAMES[i]. This order is frozen: it matches the
# `classes=` argument the notebook passed to flow_from_directory, so changing it
# would silently relabel every prediction.
CLASS_NAMES = ["Brown_Rust", "Healthy", "Yellow_Rust"]

# Below this top-probability the prediction is flagged `uncertain`.
#
# NOTE: this catches genuinely ambiguous wheat images. It does NOT provide
# out-of-distribution rejection -- the model returns ~100% confidence on inputs it
# has never seen (random noise classifies as "Healthy" with probability 1.0). The
# UI carries a standing caveat about this; the threshold alone is not a safety net.
CONFIDENCE_THRESHOLD = float(os.getenv("WHEAT_CONFIDENCE_THRESHOLD", "0.70"))

# Upload / input limits.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_TEXT_CHARS = 2000
MAX_IMAGE_PIXELS = 50_000_000  # decompression-bomb guard for Pillow

# The bundled frontend is served from the same origin, so it needs no CORS entry.
# These origins exist only for running a separate dev frontend.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "WHEAT_ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
    ).split(",")
    if o.strip()
]
