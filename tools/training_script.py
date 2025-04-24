import json
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate
from datasets import Dataset


# the LLM model we are going to be using:
# google's BERT model
MODEL = "bert-base-uncased"

ACCURACY_METRIC = evaluate.load("accuracy")
F1_METRIC = evaluate.load("f1")
PRECISION_METRIC = evaluate.load("precision")
RECALL_METRIC = evaluate.load("recall")


def compute_metrics(eval_pred):

    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)

    # weighted averages
    f1_w = F1_METRIC.compute(
        predictions=preds, references=labels, average="weighted"
    )["f1"]
    prec_w = PRECISION_METRIC.compute(
        predictions=preds, references=labels, average="weighted"
    )["precision"]
    rec_w = RECALL_METRIC.compute(
        predictions=preds, references=labels, average="weighted"
    )["recall"]

    # macro averages
    f1_m = F1_METRIC.compute(
        predictions=preds, references=labels, average="macro"
    )["f1"]
    prec_m = PRECISION_METRIC.compute(
        predictions=preds, references=labels, average="macro"
    )["precision"]
    rec_m = RECALL_METRIC.compute(
        predictions=preds, references=labels, average="macro"
    )["recall"]

    return {
        "accuracy": ACCURACY_METRIC.compute(
            predictions=preds, references=labels
        )["accuracy"],
        "f1_weighted": f1_w,
        "precision_weighted": prec_w,
        "recall_weighted": rec_w,
        "f1_macro": f1_m,
        "precision_macro": prec_m,
        "recall_macro": rec_m,
    }


# creates a dataset object from the training flat_data
def main() -> None:

    flat_data = None
    aggregate_data = None
    context = None

    flat_source = "./flattened_data_new.json"
    aggregate_source = "./aggregate_data_new.json"

    with open(flat_source, "r", encoding="utf-8") as f:
        flat_data = json.load(f)
    with open(aggregate_source, "r", encoding="utf-8") as f:
        aggregate_data = json.load(f)

    # builds context onto each entry
    # very basic and ultimately not very controlled
    try:
        for rec in flat_data:
            rec["context"] = " ".join(
                str(v) for k, v in rec.items() if k not in ("text", "label")
            ).strip()

        ds = Dataset.from_list(flat_data)
    except:
        raise (Exception("Error creating dataset from list"))

    labels = list(aggregate_data.keys())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}

    if context and "context" in flat_data[0]:
        ds = ds.map(
            lambda x: {"input_text": x["context"] + " " + x["text"]},
            batched=False,
        )
        text_field = "input_text"
    else:
        ds = ds.map(lambda x: {"input_text": x["text"]}, batched=False)
        text_field = "input_text"

    # maps labels to integers
    ds = ds.map(
        lambda x: {"labels": label2id[x["label"]]},
        remove_columns=(
            ["label", "text", "context"]
            if "context" in flat_data[0]
            else ["label", "text"]
        ),
    )

    # quickly write the label/id mappings to files, just in case
    with open("label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)
    with open("id2label.json", "w", encoding="utf-8") as f:
        json.dump(id2label, f, indent=2)

    # this creates a datadict with two keys, "train" and "test"
    # each has a subset of flat_data, one for testing and one for training
    # ratio of 80/20 train/test
    split = ds.train_test_split(0.2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    tokenized = split.map(
        lambda x: tokenizer(
            x[text_field], padding="max_length", truncation=True
        ),
        batched=True,
    )
    tokenized.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # these are the training arguments. these should be ok for testing
    # but not a full fledged run. once dataset is larger, num_train_epochs should be raised
    training_args = TrainingArguments(
        output_dir="./BERTley",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,  # simulate a 64‑batch without OOM
        num_train_epochs=5,  # for a full run, more epochs may be needed
        weight_decay=0.01,
        dataloader_num_workers=4,
        eval_strategy="epoch",  # evaluate every few steps instead of per epoch
        fp16=True,
        logging_strategy="epoch",  # log based on epoch
        logging_dir="./logs",
        save_strategy="epoch",
        save_total_limit=1,  # save checkpoints based on steps
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[
            "tensorboard"
        ],  # report metrics to TensorBoard, for example
    )

    # arguments for training the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # training the model...
    trainer.train()

    # saving the model
    trainer.save_model()

    # evaluate after training
    evals = trainer.evaluate()
    with open("evals.json", "w", encoding="utf-8") as f:
        json.dump(evals, f, indent=2)
    print("Evaluation results: ")
    print(evals)
    print("Accuracy, F1, Precision, and Recall metrics: ")
    for key, value in evals.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
