import argparse
import json
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
from pathlib import Path


def chunk_and_classify(text, classifier, tokenizer, max_len=512, stride=50):
    """
    Splits a given text into overlapping chunks, classifies each chunk using a
    provided classifier, and computes the average classification scores for
    each label across all chunks.

    Args:
        text (str): The input text to be chunked and classified.
        classifier (Callable): A function or model that takes a text input and
            returns a list of dictionaries containing classification labels and scores.
        tokenizer (Callable): A tokenizer function or model that tokenizes the input
            text and provides token IDs.
        max_len (int, optional): The maximum length of each chunk in tokens. Defaults to 512.
        stride (int, optional): The number of tokens to overlap between consecutive chunks.
            Defaults to 50.

    Returns:
        dict: A dictionary where keys are classification labels and values are the
        average scores for each label across all chunks.
    """
    # tokenize entire doc once
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    chunks = []
    for i in range(0, tokens.size(0), max_len - stride):
        chunk_ids = tokens[i : i + max_len]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        if i + max_len >= tokens.size(0):
            break

    # classify each chunk
    chunk_scores = []
    for chunk in chunks:
        scores = classifier(chunk)[0]  # list of {label, score}
        chunk_scores.append({d["label"]: d["score"] for d in scores})

    # average scores per label
    avg_scores = {
        label: sum(s[label] for s in chunk_scores) / len(chunk_scores)
        for label in chunk_scores[0]
    }
    return avg_scores


def main():

    # This initial set of lines defines the command line arguments this
    # program uses

    default_dir = Path("./BERTley/checkpoint-3486/").resolve()
    parser = argparse.ArgumentParser(
        description="Run inference on a trained BERT metadata classifier"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=default_dir,
        help="Directory where your trained model and config live",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Raw text string to classify")
    group.add_argument(
        "--input_file",
        type=str,
        help="Path to a .txt file containing the document to classify",
    )
    args = parser.parse_args()

    # 1) Load tokenizer + model (config.json should have the id2label/label2id baked in
    # thru training script)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    # 2) Build the pipeline...
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=4,
    )

    # 3) Read your document
    if args.input_file:
        text = open(args.input_file, "r", encoding="utf-8").read()
    else:
        text = args.text

    # If it’s longer than 512 tokens, needs to be chunked + classified
    # otherwise single call
    tokens = tokenizer(text, return_tensors="pt")["input_ids"]
    if tokens.size(1) <= 512:
        result = classifier(text)[0]
        scores = {d["label"]: d["score"] for d in result}
    else:
        scores = chunk_and_classify(text, classifier, tokenizer)

    # print scores
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
