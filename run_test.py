# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# This script was provided as part of a university project.
# It is used by the course instructor to evaluate the model's F1 score.
# You don’t need to run this to use or understand the project.
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
from transformers import pipeline, AutoTokenizer
import pandas as pd
import evaluate
import argparse
import os

LABEL_MAPPING = {
    "LABEL_0": "negative",
    "LABEL_1": "positive",
}

LABEL2ID = {
    "negative": 0,
    "positive": 1
}

def predict_sentiment(pipeline_object, text):
    prediction = pipeline_object(text)
    prediction = prediction[0]
    return {"label" : LABEL_MAPPING[prediction["label"]], "score" : prediction["score"]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running experiments for the pretrained model")
    parser.add_argument("--valid", action="store_true", help="Evaluate validation data")
    parser.add_argument("--test",  action="store_true", help="Evaluate test data")
    args = parser.parse_args()

    if args.test and not os.path.exists("data/test.csv"):
        raise FileNotFoundError("Test data not found. Try it with --valid flag")

    if not os.path.exists("saved_objects/tokenizer.pkl"):
        raise FileNotFoundError("Either tokenizer is not found or named wrong. Save your tokenizer object first and name it tokenizer.pkl")

    if not os.path.exists("saved_objects/embedder.pkl"):
        raise FileNotFoundError("Either embedder is not found or named wrong. Save your embedder object first and name it embedder.pkl")

    if not os.path.exists("saved_objects/model.pkl"):
        raise FileNotFoundError("Either model is not found or named wrong. Save your model object first and name it model.pkl")

    if args.valid and not args.test:
        data = pd.read_csv("data/valid.csv")
    elif args.test:
        data = pd.read_csv("data/test.csv")
    else:
        raise ValueError("Please provide either --valid or --test flag")

    # Load the model
    model_checkpoint = "emiracun/mymodel"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, max_length=512, truncation=True)
    classifier = pipeline("text-classification", model=model_checkpoint, tokenizer=tokenizer, max_length=512, truncation=True)

    # Inference
    predictions = [predict_sentiment(classifier, sample)["label"] for sample in data["text"]]
    predictions = [LABEL2ID[label] for label in predictions]
    groundtruth = [LABEL2ID[label] for label in data["label"]]

    # Calculating the performance
    metric = evaluate.load("f1")
    result = metric.compute(predictions=predictions, references=groundtruth)["f1"]

    print("Evaluation result: ", result)
