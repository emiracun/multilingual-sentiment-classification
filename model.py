# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ! This is the file for the model class.
# !     You need to implement the train and predict methods.
# !     You can also add any other methods as required.
# !     You can also add required parameters to the methods.
# !     You can also use additional packages in this file.
# !
# ! Make sure that the final implementation is compatible with the
# !     Model class. Be careful about the input and output types of
# !     the methods.
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
from datasets import DatasetDict
from typing import List, Dict, Tuple
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, AutoTokenizer, EarlyStoppingCallback
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Model:
    def __init__(self, pretrained_path: str = None, **kwargs):
        self.pretrained_path = pretrained_path
        self.__dict__.update(kwargs)

        if pretrained_path is not None:
            self.__dict__.update(self.load(self.pretrained_path).__dict__)
        else:
            self.init_model(**kwargs)

    def init_model(self, **kwargs):
        model_name = kwargs.get("model_name", "distilbert-base-multilingual-cased")
        logger.info(f"Initializing model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            id2label={0: "LABEL_0", 1: "LABEL_1"},
            label2id={"LABEL_0": 0, "LABEL_1": 1},
        )


    def train(self, datasets: Tuple[DatasetDict, DatasetDict], training_args: Dict = None):
        train_dataset, eval_dataset = datasets
        logger.info("Setting up training arguments...")
        args = TrainingArguments(
            output_dir="emiracun/mymodel",
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            learning_rate=2e-5,
            num_train_epochs=3,
            weight_decay=0.01,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            metric_for_best_model="f1",
            report_to="none",
            push_to_hub=True,
            logging_strategy="steps",
            logging_steps=100,
        )

        if training_args:
            for key, value in training_args.items():
                setattr(args, key, value)


        logger.info("Initializing trainer...")
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer=AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        trainer.train()
        self.save(os.path.join(args.output_dir, "model.pkl"))

    def predict(self, x: List[Dict]) -> List[int]:
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        logger.info("Making predictions...")

        predictions = []
        with torch.no_grad():
            for encoding in x:
                inputs = {
                    key: torch.tensor(val, dtype=torch.long if key == "input_ids" else torch.float).unsqueeze(0).to(device)
                    for key, val in encoding.items() if key in ["input_ids", "attention_mask"]
                }
                outputs = self.model(**inputs)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
                predictions.append(pred)
        logger.info("Predictions completed")
        return predictions

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        f1 = f1_score(labels, predictions, average="weighted")
        accuracy = accuracy_score(labels, predictions)
        return {"f1": f1, "accuracy": accuracy}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")

    def push_to_hub(self, path: str):
        save_path = os.path.join("saved_objects", "model.pkl")
        os.makedirs("saved_objects", exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {save_path}")
        self.tokenizer.push_to_hub(path, commit_message="Push tokenizer to hub")
        self.model.push_to_hub(path, commit_message="Push model to hub")
        logger.info(f"Model and tokenizer pushed to Hugging Face Hub: {path}")

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)
