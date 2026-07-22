"""CNN inference (paper Steps 2-4).

TensorFlow is imported lazily inside `load()` rather than at module scope, so the
rest of the app -- and every endpoint except /predict -- stays importable and
functional on a machine where TF is missing or the model file has not been placed.
"""

import logging
import threading
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from . import config

log = logging.getLogger("wheat-api")

# Guard against decompression bombs: Pillow raises instead of allocating.
Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS


class InvalidImage(ValueError):
    """Uploaded bytes could not be decoded as an image."""


@dataclass(frozen=True)
class Prediction:
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    uncertain: bool
    margin: float
    message: str | None


class Classifier:
    """Wraps the trained Keras model. Loaded once at startup, never per request."""

    def __init__(self, model_path: Path, class_names: list[str]) -> None:
        self.model_path = Path(model_path)
        self.class_names = list(class_names)
        self._model = None
        self._input_size: tuple[int, int] | None = None
        self._load_error: str | None = None
        # A single Keras model is not safe under concurrent predict() calls, so
        # inference is serialized. Throughput is irrelevant here (one 256x256
        # image is milliseconds); correctness is not.
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- loading
    def load(self) -> None:
        """Load the model. Never raises -- failure is recorded, not propagated.

        The old Flask app loaded at import time and took the whole process down
        when the file was absent. Here the API still starts and /predict returns
        a clean 503 while the text endpoints keep working.
        """
        if not self.model_path.exists():
            self._load_error = (
                f"Model file not found at {self.model_path}. "
                "Copy wheat_disease_cnn.keras there, or set WHEAT_MODEL_PATH."
            )
            log.warning("model unavailable: %s", self._load_error)
            return

        try:
            import tensorflow as tf  # noqa: PLC0415 - deliberately lazy

            model = tf.keras.models.load_model(self.model_path)

            shape = model.input_shape
            if len(shape) != 4 or shape[-1] != 3:
                raise ValueError(f"expected (None, H, W, 3) input, got {shape}")
            n_out = model.output_shape[-1]
            if n_out != len(self.class_names):
                raise ValueError(
                    f"model emits {n_out} classes but CLASS_NAMES has "
                    f"{len(self.class_names)} -- labels would be wrong"
                )

            _, height, width, _ = shape
            self._input_size = (int(width), int(height))
            self._model = model

            # First predict() triggers graph tracing and costs a second or two.
            # Spend it here so no user request pays for it.
            self._model.predict(
                np.zeros((1, height, width, 3), dtype=np.float32), verbose=0
            )
            log.info(
                "model loaded: %s, input %dx%d, %d classes",
                self.model_path.name, width, height, n_out,
            )
        except Exception as exc:  # noqa: BLE001 - startup must not die
            self._load_error = f"Failed to load model: {exc}"
            self._model = None
            log.error("model load failed: %s", exc)

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def input_size(self) -> tuple[int, int] | None:
        return self._input_size

    @property
    def load_error(self) -> str | None:
        return self._load_error

    # ------------------------------------------------------------ inference
    def preprocess(self, raw: bytes) -> np.ndarray:
        """Match the notebook's pipeline exactly: RGB -> resize -> /255 -> batch.

        Any deviation here (a different rescale, an added normalization) silently
        degrades accuracy without raising, so this must not drift.
        """
        if not raw:
            raise InvalidImage("empty file")
        try:
            img = Image.open(BytesIO(raw)).convert("RGB").resize(self._input_size)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            raise InvalidImage(str(exc)) from exc

        arr = np.asarray(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, 0)

    def predict(self, raw: bytes) -> Prediction:
        """Classify raw image bytes. Bytes are never written to disk."""
        if not self.ready:
            raise RuntimeError(self._load_error or "model not loaded")

        arr = self.preprocess(raw)
        with self._lock:
            probs = self._model.predict(arr, verbose=0)[0]

        order = np.argsort(probs)[::-1]
        top, second = int(order[0]), int(order[1])
        confidence = float(probs[top])
        margin = confidence - float(probs[second])
        uncertain = confidence < config.CONFIDENCE_THRESHOLD

        return Prediction(
            prediction=self.class_names[top],
            confidence=confidence,
            probabilities={
                name: float(probs[i]) for i, name in enumerate(self.class_names)
            },
            uncertain=uncertain,
            margin=margin,
            message=(
                "Low confidence — this may not be a clear wheat-leaf photo. "
                "Retake it in even light with a single leaf filling the frame."
                if uncertain
                else None
            ),
        )
