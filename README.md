# 🧠 Uncertainty-Aware Emotion Classifier

> A production-grade NLP system that goes beyond emotion prediction by estimating its own uncertainty using Bayesian Deep Learning techniques.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

# 📌 Overview

Traditional text classifiers provide a prediction and a confidence score. However, confidence alone does not indicate whether a model genuinely understands its prediction.

This project introduces an **Uncertainty-Aware Emotion Classification System** built using **DistilBERT** and **Monte Carlo Dropout (MC Dropout)**. Instead of producing a single deterministic prediction, the model performs multiple stochastic forward passes during inference to estimate predictive uncertainty.

The result is an intelligent NLP system capable of:

- Predicting emotions from text
- Measuring prediction confidence
- Quantifying model uncertainty
- Identifying ambiguous inputs
- Providing a full probability distribution across all emotion classes

By combining modern transformer architectures with Bayesian approximation techniques, the system delivers more trustworthy predictions for real-world AI applications.

---

# 🎯 Key Features

✅ Multi-class emotion classification

✅ Bayesian uncertainty estimation using Monte Carlo Dropout

✅ Full probability distribution across all emotion classes

✅ Confidence and uncertainty scoring

✅ Calibration analysis using Expected Calibration Error (ECE)

✅ Interactive Streamlit web interface

✅ Production-ready inference pipeline

✅ Comprehensive evaluation and visualization tools

---

# 😊 Emotions Detected

| Emotion | Description |
|----------|----------|
| 😊 Joy | Happiness, excitement, optimism |
| 😢 Sadness | Grief, disappointment, sorrow |
| 😠 Anger | Frustration, irritation, hostility |
| 😨 Fear | Anxiety, concern, nervousness |
| ❤️ Love | Affection, care, attachment |
| 😲 Surprise | Shock, amazement, astonishment |

---

# 🧠 Model Architecture

### Transformer Backbone

- DistilBERT
- Pre-trained on large-scale language corpora
- Fine-tuned for emotion classification

### Uncertainty Estimation

- Monte Carlo Dropout
- 20 stochastic forward passes during inference
- Bayesian approximation for uncertainty quantification

### Input

- Raw English text

### Output

- Predicted emotion
- Confidence score
- Uncertainty score
- Probability distribution across all classes

---

# 📊 Performance Metrics

| Metric | Score |
|----------|----------|
| Test Accuracy | **~92%** |
| Weighted F1 Score | **~92%** |
| Expected Calibration Error (ECE) | **~0.05** |
| MC Dropout Passes | **20** |
| Training Epochs | **3** |
| Dataset Size | **20,000 Samples** |

### Why Calibration Matters

A model can be accurate while still being poorly calibrated.

For example:

- A model that predicts with 95% confidence but is correct only 75% of the time is overconfident.
- A well-calibrated model ensures that confidence scores align closely with actual performance.

An ECE of approximately **0.05** indicates strong calibration and reliable confidence estimates.

---

# 📈 Evaluation Results

## Confusion Matrix

![Confusion Matrix](results/figures/confusion_matrix.png)

---

## Uncertainty Distribution

![Uncertainty Distribution](results/figures/uncertainty_distribution.png)

---

## Reliability Diagram

![Reliability Diagram](results/figures/reliability_diagram.png)

---

# 🚀 Quick Start

## 1. Clone the Repository

```bash
git clone https://github.com/Eddiegah/uncertainty-emotion-classifier.git
cd uncertainty-emotion-classifier
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Train the Model

```bash
python src/train.py
```

## 5. Run Inference

```bash
python src/predict.py
```

## 6. Generate Evaluation Reports

```bash
python src/evaluate.py
```

## 7. Launch the Web Application

```bash
streamlit run app.py
```

---

# 📁 Project Structure

```text
uncertainty-emotion-classifier/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── models/
│   └── best_model.pt
│
├── results/
│   ├── figures/
│   │   ├── confusion_matrix.png
│   │   ├── uncertainty_distribution.png
│   │   └── reliability_diagram.png
│   │
│   └── metrics.json
│
└── dataset/
```

---

# 🧰 Technology Stack

| Technology | Purpose |
|------------|----------|
| PyTorch | Deep Learning Framework |
| Hugging Face Transformers | DistilBERT Architecture |
| Hugging Face Datasets | Emotion Dataset |
| Monte Carlo Dropout | Uncertainty Quantification |
| scikit-learn | Evaluation Metrics |
| Streamlit | Interactive Web Interface |
| NumPy | Numerical Computation |
| Matplotlib | Visualization & Reporting |

---

# 💡 How It Works

Unlike traditional classifiers that generate a single prediction, this system keeps dropout layers active during inference.

For every input text:

1. The model performs 20 stochastic forward passes.
2. Each pass generates slightly different predictions.
3. Predictions are averaged to obtain final class probabilities.
4. Variance across predictions is used to estimate uncertainty.
5. Predictive entropy is normalized to produce an uncertainty score between 0 and 1.

This allows the system to distinguish between:

- Predictions it understands well
- Predictions it is uncertain about
- Ambiguous inputs that may require human review

---

# 🔬 Research Significance

Reliable AI systems must know when they might be wrong.

Uncertainty estimation is increasingly important in:

- Healthcare AI
- Financial forecasting
- Autonomous systems
- Human-AI decision support
- Risk-sensitive NLP applications

Rather than forcing a prediction for every input, uncertainty-aware systems provide an additional layer of trust and interpretability.

---

# 📦 Dataset

### Emotion Dataset

The model was trained using the **dair-ai/emotion** dataset available through Hugging Face.

Dataset Characteristics:

- Approximately 20,000 English text samples
- Six emotion categories
- Balanced benchmark dataset for emotion classification
- Widely used in NLP research

Emotion Classes:

- Joy
- Sadness
- Anger
- Fear
- Love
- Surprise

---

# 🚀 Future Improvements

- Deep Ensemble Methods
- Temperature Scaling
- Out-of-Distribution Detection
- Active Learning Integration
- Multi-label Emotion Recognition
- Real-Time REST API Deployment
- Large Language Model Benchmarking
- Advanced Bayesian Transformer Architectures

---

# 🔗 About Nexora

This project aligns with the broader vision of **Nexora**, a healthcare technology initiative focused on:

- Explainable AI
- Uncertainty Quantification
- Predictive Healthcare
- Precision Medicine
- Clinical Decision Support Systems
- Trustworthy Artificial Intelligence

---

# 📄 License

This project is licensed under the MIT License.

See the `LICENSE` file for details.

---

<div align="center">

### Built with 🧠, Transformers, and Bayesian Deep Learning

**Edmund Eric Gah (Eddiegah)**

[GitHub Profile](https://github.com/Eddiegah)

</div>
