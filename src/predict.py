import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH  = "./results/model"
LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
MC_SAMPLES  = 20
MAX_LENGTH  = 128


def enable_mc_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def predict_with_uncertainty(text, model, tokenizer, n_samples=MC_SAMPLES):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True
    )

    model.eval()
    enable_mc_dropout(model)

    all_probs = []
    with torch.no_grad():
        for _ in range(n_samples):
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().numpy()
            all_probs.append(probs)

    all_probs  = np.array(all_probs)
    mean_probs = all_probs.mean(axis=0)

    predicted_idx   = int(np.argmax(mean_probs))
    predicted_label = LABEL_NAMES[predicted_idx]
    confidence      = float(mean_probs[predicted_idx])

    entropy     = float(-np.sum(mean_probs * np.log(mean_probs + 1e-8)))
    max_entropy = float(np.log(len(LABEL_NAMES)))
    uncertainty = entropy / max_entropy

    return {
        "text": text,
        "predicted_emotion": predicted_label,
        "confidence_pct": round(confidence * 100, 2),
        "uncertainty_pct": round(uncertainty * 100, 2),
        "all_emotion_probs": {
            label: round(float(prob) * 100, 2)
            for label, prob in zip(LABEL_NAMES, mean_probs)
        }
    }


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return model, tokenizer


if __name__ == "__main__":
    print("Loading model...")
    model, tokenizer = load_model()

    test_texts = [
        "I am so happy today, everything is going great!",
        "I feel really sad and empty inside.",
        "This makes me so angry I could scream!",
        "I love spending time with my family.",
        "I'm terrified of what might happen next.",
        "Wow, I did not see that coming at all.",
        "I'm not sure how I feel about this situation.",
    ]

    print("\n" + "="*60)
    print("PREDICTIONS WITH UNCERTAINTY")
    print("="*60)
    for text in test_texts:
        result = predict_with_uncertainty(text, model, tokenizer)
        print(f"\nText       : {result['text'][:65]}")
        print(f"Emotion    : {result['predicted_emotion'].upper()}")
        print(f"Confidence : {result['confidence_pct']}%")
        print(f"Uncertainty: {result['uncertainty_pct']}%")