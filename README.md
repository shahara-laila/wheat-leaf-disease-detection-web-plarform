# Wheat Leaf Disease Detection

A small web application with two parts:

1. **Photo check** - a trained CNN classifies a wheat leaf photo as
   Brown Rust, Yellow Rust or Healthy.
2. **Symptom search** - typed symptoms are matched against a list of
   13 diseases plus Healthy, each with a treatment recommendation.

The second part does not use the model, so it works on its own.

## Files

| File | What it does |
|---|---|
| `app.py` | The whole backend: four endpoints and the model loading. |
| `diseases.json` | The disease knowledge base (names, symptoms, treatment). |
| `frontend/index.html` | The page. |
| `frontend/app.js` | Calls the endpoints and shows the results. |
| `frontend/style.css` | Styling. |

The trained model `wheat_disease_cnn.keras` (26 MB) is not in the repository.
Copy it into the project folder before running. Without it the app still
starts and the symptom search works.

## Run it

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open <http://localhost:8000>.
Automatic API documentation is at <http://localhost:8000/docs>.

TensorFlow needs Python 3.9-3.13. It has no wheels for Python 3.14 yet.

## Endpoints

| Method | Path | What it returns |
|---|---|---|
| GET | `/health` | Whether the model loaded and how many diseases are known. |
| GET | `/diseases` | The whole knowledge base. |
| POST | `/predict` | Send an image in the `image` field, get the class, confidence, all three probabilities and the treatment. |
| POST | `/recommend` | Send `{"text": "..."}`, get every matching disease with its treatment. |

## How the prediction works

The uploaded photo goes through the same steps used during training:
convert to RGB, resize to 256x256, scale the pixel values to 0-1, then
wrap it in a batch of one. The model returns three probabilities and the
highest one is the answer. Below 70% the result is marked uncertain.

The photo is kept in memory only and is never written to disk.

## Limitation

The model only knows three classes, so it gives a confident answer for any
image, including photos that are not wheat leaves. The confidence value
tells you how sure the model is between those three classes, nothing more.
