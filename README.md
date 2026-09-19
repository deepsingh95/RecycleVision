# ♻️ RecycleVision — Garbage Image Classification Using Deep Learning

Deep learning system that classifies waste images into six categories —
**cardboard, glass, metal, paper, plastic, and trash** — to help automate
recycling and reduce manual sorting effort.

**Domain:** Waste Management · Environmental Tech · Deep Learning · Computer Vision

---

## 📊 Results

The final model (**EfficientNetB0, fine-tuned**) achieves **85.42% accuracy**
on a held-out test set of 384 unseen images.

| Model | Test Accuracy | Test F1-Score |
|---|---|---|
| MobileNetV2 (frozen backbone) | 80.21% | 80.53% |
| MobileNetV2 (fine-tuned) | 83.33% | 83.54% |
| EfficientNetB0 (frozen backbone) | 84.90% | 85.14% |
| **EfficientNetB0 (fine-tuned) — final model** | **85.42%** | **85.63%** |

**Per-class performance (final model):**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Cardboard | 0.92 | 0.95 | 0.94 |
| Glass | 0.91 | 0.79 | 0.85 |
| Metal | 0.75 | 0.87 | 0.81 |
| Paper | 0.96 | 0.86 | 0.91 |
| Plastic | 0.82 | 0.81 | 0.81 |
| Trash | 0.65 | 0.91 | 0.75 |

See [`src/reports/`](src/reports/) for the full evaluation report and
confusion matrix, and [`src/reports/class_distribution.png`](src/reports/class_distribution.png)
/ [`src/reports/sample_images.png`](src/reports/sample_images.png) for the EDA plots.

---

## 🚀 Demo

The Streamlit app lets you upload any waste image and get an instant
prediction with confidence scores and top-3 alternatives.

```bash
streamlit run app/streamlit_app.py
```

---

## 🗂️ Project Structure
```
RecycleVision/
├── src/
│   ├── config.py           # Dataset variant configs (6/12/10-class)
│   ├── download_data.py    # Kaggle dataset download script
│   ├── preprocessing.py    # Resize, normalize, split (train/val/test)
│   ├── eda.py               # Exploratory data analysis + plots
│   ├── model.py             # Transfer-learning model architectures
│   ├── train.py             # Base training (frozen backbone)
│   ├── fine_tune.py         # Fine-tuning (unfreezes top backbone layers)
│   ├── evaluate.py          # Test-set metrics, confusion matrix
│   ├── data/                # (gitignored) downloaded + processed images
│   ├── models/              # Saved models + class-index mappings
│   └── reports/             # EDA plots, evaluation reports, confusion matrices
├── app/
│   └── streamlit_app.py     # Deployment UI
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up Kaggle API credentials
# Download kaggle.json from kaggle.com -> Settings -> API
# Place it at ~/.kaggle/kaggle.json (or %USERPROFILE%\.kaggle\kaggle.json on Windows)
```

## 🏗️ Pipeline — Reproducing the Results

Run these from inside `src/`:

```bash
# 1. Download the dataset (6-class Garbage Classification, ~2,467 images)
python download_data.py

# 2. Preprocess: resize to 224x224, split 70/15/15 into train/val/test
python preprocessing.py

# 3. Exploratory Data Analysis
python eda.py

# 4. Train the base model (frozen backbone)
python train.py --backbone EfficientNetB0

# 5. Fine-tune (unfreeze top layers, lower learning rate)
python fine_tune.py --backbone EfficientNetB0 --unfreeze 30 --epochs 15

# 6. Evaluate on the test set
python evaluate.py --backbone EfficientNetB0 --suffix _finetuned
```

Then launch the app from the project root:
```bash
streamlit run app/streamlit_app.py
```

---

## 🧠 Approach

1. **Data Preparation** — Garbage Classification dataset (Kaggle), 6 classes.
2. **Preprocessing** — resize to 224×224, normalize, stratified 70/15/15 split.
3. **EDA** — class distribution, sample images, pixel intensity analysis
   (revealed class imbalance: `trash` has far fewer images than other classes).
4. **Model Development** — transfer learning with a frozen ImageNet backbone
   (MobileNetV2 and EfficientNetB0 were both tried) + a custom classification
   head (GlobalAveragePooling → Dropout → Dense → Dropout → Softmax).
   Class weights were used during training to counter the imbalance.
5. **Fine-tuning** — unfroze the top layers of the backbone and continued
   training at a much lower learning rate, which improved test accuracy by
   several points over the frozen-backbone baseline.
6. **Evaluation** — Accuracy, Precision, Recall, F1-Score, and a confusion
   matrix on the held-out test set (never seen during training or fine-tuning).
7. **Model Selection** — EfficientNetB0 (fine-tuned) was chosen as the final
   model: highest test accuracy and F1-score, and the most balanced
   per-class performance.
8. **Deployment** — a Streamlit app for interactive, real-time predictions.

---

## 🛠️ Tech Stack
Python · TensorFlow / Keras · OpenCV · Pillow · scikit-learn ·
Matplotlib / Seaborn · Streamlit · Kaggle API

---

## 📌 Notes
- Trained and evaluated on CPU (no native GPU support on Windows for
  TensorFlow ≥ 2.11); training the final model took roughly 1–2 hours total
  across base training + two fine-tuning rounds.
- Large files (raw/processed dataset images, `.keras` model weights) are
  excluded from this repository via `.gitignore` — regenerate them by
  running the pipeline above.
