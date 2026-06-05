🧠 Uncertainty-Aware Emotion Classifier

«An advanced NLP system that not only predicts emotions from text but also quantifies its own uncertainty using Bayesian Deep Learning techniques.»

"Python" (https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
"PyTorch" (https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch)
"HuggingFace" (https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface)
"License" (https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

📌 Overview

Most machine learning classifiers produce a prediction and a confidence score, but confidence alone does not indicate whether a model truly understands its prediction.

This project introduces Uncertainty-Aware Emotion Classification, leveraging Monte Carlo Dropout, a Bayesian approximation technique, to estimate predictive uncertainty alongside emotion predictions.

Built on a fine-tuned DistilBERT transformer and trained on over 20,000 labeled text samples, the system performs robust emotion recognition while identifying cases where the model lacks certainty.

The result is a more trustworthy and interpretable NLP system suitable for real-world AI applications where reliability matters.

---

🎯 Key Features

✅ Multi-class emotion classification

✅ Predictive uncertainty estimation

✅ Bayesian Deep Learning with Monte Carlo Dropout

✅ Probability distribution across all emotion classes

✅ Confidence and uncertainty reporting

✅ Calibration analysis and reliability evaluation

✅ Interactive Streamlit web application

✅ Production-ready inference pipeline

---

😊 Emotions Detected

Emotion| Description
😊 Joy| Happiness, excitement, optimism
😢 Sadness| Grief, disappointment, sorrow
😠 Anger| Frustration, hostility, irritation
😨 Fear| Anxiety, concern, nervousness
❤️ Love| Affection, care, attachment
😲 Surprise| Shock, amazement, astonishment

---

🧠 Model Architecture

Transformer Backbone

- DistilBERT
- Pre-trained on large-scale language corpora
- Fine-tuned for emotion classification

Uncertainty Estimation

- Monte Carlo Dropout
- 20 stochastic forward passes per prediction
- Bayesian approximation of model uncertainty

Input

- Raw English text

Output

- Emotion prediction
- Confidence score
- Uncertainty score
- Full class probability distribution

---

📊 Performance Metrics

Metric| Score
Test Accuracy| ~92%
Weighted F1 Score| ~92%
Expected Calibration Error (ECE)| ~0.05
Monte Carlo Passes| 20
Training Epochs| 3
Dataset Size| 20,000 Samples

Why Calibration Matters

A classifier may be highly accurate yet poorly calibrated.

For example:

- A model claiming 95% confidence but being correct only 75% of the time is overconfident.
- A well-calibrated model aligns confidence with real-world accuracy.

An ECE of approximately 0.05 indicates strong calibration performance and reliable confidence estimates.

---

📈 Evaluation Results

Confusion Matrix

"Confusion Matrix" (results/figures/confusion_matrix.png)

---

Uncertainty Distribution

"Uncertainty Distribution" (results/figures/uncertainty_distribution.png)

---

Reliability Diagram

"Reliability Diagram" (results/figures/reliability_diagram.png)

---

🚀 Quick Start

1. Clone the Repository

git clone https://github.com/Eddiegah/uncertainty-emotion-classifier.git
cd uncertainty-emotion-classifier

2. Create a Virtual Environment

python -m venv venv

Windows

venv\Scripts\activate

Linux / macOS

source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

4. Train the Model

python src/train.py

5. Run Predictions

python src/predict.py

6. Generate Evaluation Reports

python src/evaluate.py

7. Launch the Web Application

streamlit run app.py

---

📁 Project Structure

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
├── results/
│   ├── figures/
│   │   ├── confusion_matrix.png
│   │   ├── uncertainty_distribution.png
│   │   └── reliability_diagram.png
│   │
│   └── metrics.json
│
└── models/
    └── best_model.pt

---

🧰 Technology Stack

Technology| Purpose
PyTorch| Deep Learning Framework
Hugging Face Transformers| DistilBERT Architecture
Hugging Face Datasets| Emotion Dataset
Monte Carlo Dropout| Uncertainty Quantification
scikit-learn| Metrics & Evaluation
Streamlit| Interactive Web Interface
NumPy| Numerical Computation
Matplotlib| Visualization & Reporting

---

💡 How It Works

Traditional classifiers generate a single deterministic prediction.

This system keeps dropout layers active during inference, creating a stochastic neural network.

For every input text:

1. The model performs 20 forward passes.
2. Each pass produces slightly different predictions.
3. Predictions are averaged to obtain the final class probabilities.
4. Variance across predictions is used to estimate uncertainty.
5. Predictive entropy is normalized to generate an uncertainty score between 0 and 1.

This allows the model to distinguish between:

- Predictions it understands well
- Predictions it is genuinely uncertain about

---

🔬 Research Significance

Uncertainty estimation is becoming increasingly important in:

- Healthcare AI
- Financial forecasting
- Autonomous systems
- Risk-sensitive NLP applications
- Human-AI decision support systems

By quantifying uncertainty, models can defer difficult cases for human review rather than making unreliable predictions.

---

📦 Dataset

dair-ai/emotion

A benchmark emotion classification dataset containing approximately 20,000 English-language text samples annotated with six emotional categories.

Classes include:

- Joy
- Sadness
- Anger
- Fear
- Love
- Surprise

---

🚀 Future Improvements

- Deep Ensembles
- Temperature Scaling
- Out-of-Distribution Detection
- Active Learning Integration
- Multi-label Emotion Recognition
- Real-time API Deployment
- Transformer Uncertainty Benchmarking

---

🔗 About Nexora

This project reflects the broader vision behind Nexora — building trustworthy AI systems that combine predictive performance with transparency, uncertainty awareness, and real-world reliability.

Key focus areas include:

- Explainable AI
- Uncertainty Quantification
- Predictive Healthcare
- Clinical Decision Support
- Human-Centered AI Systems

---

📄 License

This project is licensed under the MIT License.

See the "LICENSE" file for more information.

---

<div align="center">Built with 🧠, Transformers, and Bayesian Deep Learning

Edmund Eric Gah (Eddiegah)

GitHub: https://github.com/Eddiegah

</div>
