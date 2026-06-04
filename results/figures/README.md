# Uncertainty-Aware Emotion Classifier

A fine-tuned DistilBERT model that classifies text into 6 emotions with **MC Dropout uncertainty estimation** — it doesn't just predict, it tells you how confident and uncertain it is.

## Emotions
`joy` · `sadness` · `anger` · `fear` · `love` · `surprise`

## Method
Uses **Monte Carlo Dropout** at inference time: 20 stochastic forward passes with dropout active, averaged to produce a mean prediction, confidence score, and uncertainty score (predictive entropy).

## Results

| Metric | Value |
|---|---|
| Test Accuracy | 93.67% |
| Weighted F1 | 93.5% |
| ECE | 3.01%s |

## Plots



![Confusion Matrix](results/figures/confusion_matrix.png)




![Uncertainty Distribution](results/figures/uncertainty_distribution.png)




![Reliability Diagram](results/figures/reliability_diagram.png)



## Quick Start

```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py
python src/evaluate.py

Tech stack: Python, PyTorch, Hugging Face Transformers, scikit-learn, matplotlib, DistilBERT, MC Dropout, ECE, reliability diagrams