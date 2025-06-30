# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ! This is the file for preprocessing objects and their utilities.
# !     Do not change Preprocessor, LabelEncoder, PreprocessorObject 
# !         classes.
# !     You need to update Tokenizer and Embedder classes.
# !     You can add any other classes or methods as needed.
# !     You can also add required parameters to the methods.
# !     You can also implement additional methods as required.
# !     You can also use additional packages in this file.
# !
# ! Make sure that the final implementation is compatible with the
# !     Preprocessor class and its methods. Be careful about the 
# !     input and output types of the methods.
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
from datasets import DatasetDict
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModel
import torch
import pickle
import os
import numpy as np

class Preprocessor:
    def __init__(self, tokenizer, collator, **kwargs):
        self.tokenizer = tokenizer
        self.collator = collator
        self.label_encoder = LabelEncoder(labels={"negative": 0, "positive": 1})
        self.__dict__.update(kwargs)

    def prepare_data(self, dataset: DatasetDict) -> DatasetDict:
        raise NotImplementedError

class LabelEncoder:
    def __init__(self, labels, **kwargs):
        self.id2label = {v: k for k, v in labels.items()}
        self.label2id = {k: v for k, v in labels.items()}
        self.__dict__.update(kwargs)

######################################################
# ! Update Tokenizer and Embedder classes
######################################################

class Tokenizer:
    def __init__(self, model_name="distilbert-base-multilingual-cased", **kwargs):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.__dict__.update(kwargs)

    def train(self, texts: List[str]):
        pass

    def tokenize(self, text: str) -> Dict:
        return self.tokenizer(text, padding="max_length", truncation=True, max_length=128)

    def push_to_hub(self, path: str):
        save_path = os.path.join("saved_objects", "tokenizer.pkl")
        os.makedirs("saved_objects", exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
        self.tokenizer.push_to_hub(path, commit_message="Push tokenizer to hub")

    def from_pretrained(self, path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        return self

class Embedder:
    def __init__(self, model_name="distilbert-base-multilingual-cased", **kwargs):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(self.model_name)
        self.embedding_layer = self.model.get_input_embeddings()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.__dict__.update(kwargs)

    def embed(self, tokenized_text: Dict) -> List[float]:
        input_ids = torch.tensor(tokenized_text["input_ids"]).to(self.device)
        attention_mask = torch.tensor(tokenized_text["attention_mask"]).to(self.device)

        with torch.no_grad():
            embeddings = self.embedding_layer(input_ids)
            masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
            mean_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1).clamp(min=1e-9)

        return mean_embeddings.squeeze().cpu().numpy().tolist()

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)
