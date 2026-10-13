# sps_genai

FastAPI service for the Columbia SPS GenAI course (Yihan Wang, yw4669).

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/` | Health check |
| POST | `/generate` | Bigram text generation. Body: `{"start_word": "the", "length": 10}` |
| POST | `/embedding` | spaCy word vector (300-dim). Body: `{"word": "king"}` |
| GET | `/similarity?word1=king&word2=queen` | Cosine similarity of two words |
| POST | `/classify` | **Assignment 2.** Upload an image, returns the predicted CIFAR-10 class |

## Assignment 2: CNN image classifier

- `app/cnn_model.py`: the required CNN architecture (`AssignmentCNN`)
  - Input 64x64x3 → Conv(16, 3x3, s1, p1) → ReLU → MaxPool(2, 2) → Conv(32, 3x3, s1, p1) → ReLU → MaxPool(2, 2) → Flatten (8192) → FC(100) → ReLU → FC(10)
  - CIFAR-10 images (32x32) are resized to 64x64 to match the required input size
- `train_cnn.py`: trains the model on CIFAR-10 and saves `app/models/cnn_cifar10.pth`
- `app/cnn_classifier.py`: loads the trained weights and runs inference
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

The trained weights are already included, so the API works without retraining. To retrain:

```bash
uv run python train_cnn.py --epochs 10
```

## Run locally

```bash
uv sync
uv run fastapi dev app/main.py
```

Open http://127.0.0.1:8000/docs, or:

```bash
curl -X POST http://127.0.0.1:8000/classify -F "file=@cat.jpg"
```

Example response:

```json
{
  "filename": "cat.jpg",
  "predicted_class": "cat",
  "confidence": 0.71,
  "top_k": [
    {"class": "cat", "probability": 0.71},
    {"class": "dog", "probability": 0.18},
    {"class": "frog", "probability": 0.04}
  ]
}
```

## Run with Docker

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Then open http://127.0.0.1:8000/docs.
