from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import lightning as L
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, target_model = None):
    if target_model is not None:
        input_ids_padded = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=0)
        scores_list = [item["scores"].squeeze(0) for item in batch]
        scores_padded = pad_sequence(scores_list, batch_first=True, padding_value=0)
        return {
            "input_ids": input_ids_padded,
            "scores": scores_padded
        } 
    return torch.utils.data.default_collate(batch)

class WikiTextV2Datamodule(L.LightningDataModule):
    def __init__(self, min_len: int, max_len: int, target_model = None, device="cuda:4", num_workers: int = 0, batch_size: int = 16) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.min_len = min_len 
        self.max_len = max_len 
        self.num_workers = num_workers
        self.target_model = target_model
        self.device = device
   
    def setup(self, stage) -> None:
        train_data = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
        test_data = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="test")
        # If model is not None, batch has structure {"text": [N strings]} where N is batchsize
        self.train_dataset = self.filter_dataset(train_data, self.min_len, self.max_len)
        self.test_dataset = self.filter_dataset(test_data, self.min_len, self.max_len)
        
        if self.target_model is not None:
            self.prepare_dataset_for_draft_model_finetuning()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=lambda batch: collate_fn(batch, self.target_model),
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda batch: collate_fn(batch, self.target_model),
        )
    
    def prepare_dataset_for_draft_model_finetuning(self):
        cache_dir = "data"
        os.makedirs(cache_dir, exist_ok=True)
        train_cache_path = os.path.join(cache_dir, "train.pt")
        test_cache_path = os.path.join(cache_dir, "test.pt")
        
        # Check if cache exists
        if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
            self.train_dataset = torch.load(train_cache_path)
            self.test_dataset = torch.load(test_cache_path)
            print(f"Loaded preprocessed data from cache")
        else:
            # Trying to get tokenizer for the model
            if hasattr(self.target_model, 'tokenizer'):
                tokenizer = self.target_model.tokenizer
            else:
                try:
                    model_name = self.target_model.config._name_or_path
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                except:
                    raise ValueError("Could not determine appropriate tokenizer for the target model")
            
            def process_dataset(dataset, desc):
                processed_data = []
                for item in tqdm(dataset, desc=desc):
                    input_text = item["text"]
                    
                    tokenized_input = tokenizer(input_text, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        scores = self.target_model(tokenized_input.input_ids[0]).logits[:, -1, :]   
                    processed_data.append({
                        "input_ids": tokenized_input.input_ids[0],
                        "scores": scores
                    })
                
                return processed_data
            
            self.train_dataset = process_dataset(self.train_dataset, "Processing train dataset")
            self.test_dataset = process_dataset(self.test_dataset, "Processing test dataset")
            
            # Save to cache
            torch.save(self.train_dataset, train_cache_path)
            torch.save(self.test_dataset, test_cache_path)
            print(f"Processed datasets and saved to cache")

    @staticmethod
    def filter_dataset(dataset, min_len: int, max_len: int) -> list[dict[str, str]]:
        return dataset.filter(lambda row: min_len < len(row["text"].split()) <= max_len)