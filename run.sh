#!/usr/bin/env bash
# Launch the wheat disease detection API + frontend.
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x venv/bin/uvicorn ]]; then
  echo "No venv found. Create it with Python 3.13 (TensorFlow has no 3.14 wheels):"
  echo "  /opt/homebrew/bin/python3.13 -m venv venv"
  echo "  ./venv/bin/pip install -r requirements.txt"
  exit 1
fi

if [[ ! -f wheat_disease_cnn.keras ]]; then
  echo "WARNING: wheat_disease_cnn.keras not found."
  echo "  /predict will return 503; the symptom and browse tabs still work."
fi

# One worker on purpose: each would load its own ~26 MB model plus a full TF
# runtime, and inference is already serialized. --reload is dev-only (it reloads
# the model on every file save).
exec ./venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000 "$@"
