from datasets import Dataset, DatasetDict
import pandas as pd
from preprocessing import Preprocessor, Tokenizer, Embedder, LabelEncoder
from model import Model
from transformers import DataCollatorWithPadding, AutoTokenizer
import os
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    logger.info("Loading data...")
    train_df = pd.read_csv("data/train.csv")
    valid_df = pd.read_csv("data/valid.csv")

    if 'sentence' in train_df.columns:
        train_df = train_df.rename(columns={'sentence': 'text', 'sentiment': 'label'})
    if 'sentence' in valid_df.columns:
        valid_df = valid_df.rename(columns={'sentence': 'text', 'sentiment': 'label'})

    label_map = {0: 'LABEL_0', 1: 'LABEL_1', 'negative': 'LABEL_0', 'positive': 'LABEL_1'}
    train_df['label'] = train_df['label'].map(label_map)
    valid_df['label'] = valid_df['label'].map(label_map)

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(valid_df)
    })
    logger.info("Data loaded successfully")
    return dataset

def preprocess_dataset(dataset, tokenizer, label_encoder):
    logger.info("Preprocessing dataset...")
    def tokenize_and_encode(examples):
        tokenized = tokenizer.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        tokenized["labels"] = [label_encoder.label2id[label] for label in examples["label"]]
        return tokenized
    preprocessed = dataset.map(tokenize_and_encode, batched=True, remove_columns=["text", "label"])
    logger.info("Dataset preprocessed successfully")
    return preprocessed

def main():
    try:
        logger.info("Starting main execution...")
        os.makedirs("saved_objects", exist_ok=True)
        tokenizer = Tokenizer(model_name="distilbert-base-multilingual-cased")
        label_encoder = LabelEncoder(labels={"LABEL_0": 0, "LABEL_1": 1})
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer.tokenizer)
        preprocessor = Preprocessor(tokenizer=tokenizer, collator=data_collator)

        dataset = load_data()
        preprocessed_dataset = preprocess_dataset(dataset, tokenizer, label_encoder)
        model = Model(model_name="distilbert-base-multilingual-cased")
        model.train((preprocessed_dataset["train"], preprocessed_dataset["validation"]))
        logger.info("Model training completed")

        logger.info("Pushing tokenizer to Hugging Face Hub...")
        tokenizer.push_to_hub("emiracun/mymodel")
        logger.info("Tokenizer pushed successfully")

        logger.info("Pushing model to Hugging Face Hub...")
        model.push_to_hub("emiracun/mymodel")
        logger.info("Model pushed successfully")

        embedder = Embedder()
        embedder.save("saved_objects/embedder.pkl")
        logger.info("Embedder saved successfully")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    os.makedirs("saved_objects", exist_ok=True)
    main()
