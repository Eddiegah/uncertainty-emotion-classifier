import torch
import numpy as np
import os
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME    = "distilbert-base-uncased"
DATASET_NAME  = "dair-ai/emotion"
OUTPUT_DIR    = "./results/model"
MAX_LENGTH    = 128
BATCH_SIZE    = 16
NUM_EPOCHS    = 3
LEARNING_RATE = 2e-5

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def load_data():
    return load_dataset(DATASET_NAME)


def tokenize_data(dataset, tokenizer):
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False
        )
    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("./results/figures", exist_ok=True)

    print("Loading dataset...")
    dataset = load_data()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizing dataset...")
    tokenized_dataset = tokenize_data(dataset, tokenizer)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_NAMES),
        id2label={i: label for i, label in enumerate(LABEL_NAMES)},
        label2id={label: i for i, label in enumerate(LABEL_NAMES)}
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./results/logs",
        logging_steps=50,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("\nStarting training...")
    trainer.train()

    print("\nSaving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nFinal evaluation on test set...")
    results = trainer.evaluate(tokenized_dataset["test"])
    print(f"\nTest Results: {results}")

    with open("./results/train_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nTraining complete! Model saved to ./results/model/")


if __name__ == "__main__":
    main()