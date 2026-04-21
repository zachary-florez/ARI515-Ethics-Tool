# EthicsAITool 🛰️

A Java-based detection platform that identifies AI-generated satellite imagery using a TensorFlow machine learning 
pipeline alongside with LLM-based analysis.

---

## Overview

As generative AI tools make synthetic satellite imagery hard to distinguish from real images, the risk of fake 
geospatial content being used in misinformation and conflict reporting has grown in today’s society and there’s 
no widely adopted automated detection framework. This project is motivated by AI-generated image detection, 
including work on the growing field of deepfake detection.
EthicsAITool provides a software framework that:

- Classifies satellite images as **real** or **AI-generated** using a fine-tuned MobileNetV2 model
- Fetches live satellite tiles from **NASA EPIC** and **Mapbox** for real-world testing
- Escalates uncertain detections (40–80% confidence) for deeper LLM analysis
- Tracks detection history and statistics across batches of images
- Exports results with confidence scores and LLM verdicts

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Java 17 |
| Build | Maven |
| ML Inference | TensorFlow Java 1.1.0 |
| Image Processing | OpenCV 4.9.0 |
| Model Training | Python 3.11 / TensorFlow / Keras |
| LLM Analysis | Anthropic Claude API |
| Satellite Data | Mapbox Static Images API, NASA EPIC API |
| Image Generation | Stable Diffusion (stabilityai/stable-diffusion-2-1-base) |
| HTTP Client | OkHttp 4.12.0 |
| CSV Processing | OpenCSV 5.8 |
| Logging | SLF4J + Logback |

---

## Prerequisites

- Java 17+
- Maven 3.8+
- Python 3.11 (via Homebrew: `brew install python@3.11`)
- Apple Silicon Mac (M1/M2/M3) or Linux x86_64

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/EthicsAITool.git
cd EthicsAITool
```

### 2. Set up the Python virtual environment
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install tensorflow pillow scikit-learn diffusers torch accelerate transformers
```

### 3. Set up your data directory
```
data/
├── raw/          ← place your satellite images here
└── metadata.csv  ← CSV with columns: image_id, filename, source, label, resolution, date_collected, llm_analyzed
```

The `label` column must be either `real` or `ai_generated`.

### 4. Train the model
```bash
source venv/bin/activate
python3 scripts/train_model.py \
  --data_dir ./data/raw \
  --csv      ./data/metadata.csv \
  --output   ./models/satellite_detector \
  --epochs   40
```

Target metrics after training:
- `val_accuracy` > 0.85
- `val_auc` > 0.90

### 5. Build the Java project
```bash
mvn clean install
```

### 6. Run the platform
```bash
mvn exec:java -Dexec.mainClass="org.example.Main"
```

Or run `Main.java` directly from IntelliJ.

---

## Dataset

The model is trained on a balanced dataset of **~2000 satellite images**:

| Class | Count | Source                                     |
|---|---|--------------------------------------------|
| `real` | ~1000 | Web-scraped satellite imagery              |
| `ai_generated` | ~1000 | Web-scraped ai_generated satellite imagery |

### Generating AI training images
```bash
source venv/bin/activate
python3 scripts/generate_ai_satellite_images.py \
  --output_dir ./data/raw \
  --csv        ./data/metadata.csv \
  --count      1000 \
  --start_id   1001
```

---

## Real-World Testing

The platform can fetch and analyze live satellite imagery:

```java
SatelliteImageFetcher fetcher = new SatelliteImageFetcher(
    inferenceEngine, llmIntegrator, "./data/realworld_cache"
);

// NASA EPIC — no API key required
fetcher.analyzeNASAEpicImage();

// Mapbox — requires MAPBOX_TOKEN
fetcher.analyzeMapboxTile(40.7128, -74.0060, 16, mapboxToken); // New York City
fetcher.analyzeMapboxTile(51.5074, -0.1278,  16, mapboxToken); // London
fetcher.analyzeMapboxTile(35.6762, 139.6503, 16, mapboxToken); // Tokyo

// Local file
fetcher.analyzeLocalFile(Paths.get("./data/raw/test_image.jpg"));
```

Example output:
```
=== DETECTION RESULT ===
  Image:       mapbox_40.7128_-74.0060_z16.jpg
  Source:      Mapbox
  Verdict:     REAL
  Confidence:  91.4%
========================
```

---

## Sample Results

After training on the balanced 2000-image dataset:

| Test | Verdict | Confidence |
|---|---|---|
| Known AI image 1 | AI-GENERATED | 97.4% |
| Known AI image 2 | AI-GENERATED | 100.0% |
| Known AI image 3 | AI-GENERATED | 99.1% |
| Known real image 1 | REAL | 79.7% |
| Known real image 2 | REAL | 82.3% |
| NASA EPIC (live) | REAL | 91.4% |

---

## Configuration

All thresholds are configurable at the top of `Main.java`:

| Parameter | Default | Description |
|---|---|---|
| `inputSize` | 224 | Image resize dimension |
| `confidenceThreshold` | 0.5 | AI-generated cutoff |
| `llmLowerBound` | 0.4 | LLM escalation lower bound |
| `llmUpperBound` | 0.8 | LLM escalation upper bound |
| `modelPath` | `./models/satellite_detector` | SavedModel directory |
| `dataPath` | `./data` | Data root directory |



## Project Structure Notes

Training is done entirely in Python (`scripts/train_model.py`) and exported as a TensorFlow SavedModel. The Java side only performs inference — it never trains. This is by design since the TF Java 1.1.0 API does not support building or training CNN graphs from scratch.

---

# ARI515-Ethics-Tool
