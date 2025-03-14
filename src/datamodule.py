from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import lightning as L
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import os
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, target_model = None):
    if target_model is not None:
        input_ids_padded = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=0)
        logits_list = [item["logits"].squeeze(0) for item in batch]
        logits_padded = pad_sequence(logits_list, batch_first=True, padding_value=0)
        return {
            "input_ids": input_ids_padded,
            "logits": logits_padded
        } 
    return torch.utils.data.default_collate(batch)

class WikiTextV2Datamodule(L.LightningDataModule):
    def __init__(self, min_len: int, max_len: int, target_model = None, target_model_tokenizer = None, device="cuda:4", num_workers: int = 0, batch_size: int = 16, check_cache=True) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.min_len = min_len 
        self.max_len = max_len 
        self.num_workers = num_workers
        self.target_model = target_model
        self.target_model_tokenizer = target_model_tokenizer
        self.device = device
        self.check_cache = check_cache
   
    def setup(self, stage) -> None:
        train_data = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
        test_data = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="test")
        
        # If model is not None, batch has structure {"text": [N strings]} where N is batchsize    
        self.train_dataset = self.filter_dataset(train_data, self.min_len, self.max_len)
        self.val_dataset = self.filter_dataset(test_data, self.min_len, self.max_len)
        
        if self.target_model is not None:
            if self.target_model_tokenizer is None:
                raise Exception("You should provide target model tokenizer for fine-tuning too!")
            self.prepare_dataset_for_draft_model_finetuning()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=lambda batch: collate_fn(batch, self.target_model),
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
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
        if self.check_cache and os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
            self.train_dataset = torch.load(train_cache_path)
            self.val_dataset = torch.load(test_cache_path)
            print(f"Loaded preprocessed data from cache")
        else:
            def process_dataset(dataset, desc):
                processed_data = []
                for item in tqdm(dataset, desc=desc):
                    input_text = item["text"]
                    
                    tokenized_input = self.target_model_tokenizer(input_text, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        logits = self.target_model(tokenized_input.input_ids[0]).logits   
                    processed_data.append({
                        "input_ids": tokenized_input.input_ids[0].cpu(),
                        "logits": logits.cpu()
                    })
                
                return processed_data
            
            self.train_dataset = process_dataset(self.train_dataset, "Processing train dataset")
            self.val_dataset = process_dataset(self.val_dataset, "Processing test dataset")
            
            # Save to cache
            torch.save(self.train_dataset, train_cache_path)
            torch.save(self.val_dataset, test_cache_path)
            print(f"Processed datasets and saved to cache")

    @staticmethod
    def filter_dataset(dataset, min_len: int, max_len: int) -> list[dict[str, str]]:
        return dataset.filter(lambda row: min_len < len(row["text"].split()) <= max_len)