from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import lightning as L
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

class WikiTextV2Dataset(L.LightningDataModule):
    train_dataset: Dataset
    test_dataset: Dataset
    batch_size: int
    maxlen: int

    def __init__(self, min_len: int, max_len: int, batch_size: int = 16) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.min_len = min_len 
        self.max_len = max_len 

    # Batch has structure {"text": [N strings]} where N is batchsize
    def setup(self, stage: str) -> None:
        train_data = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
        test_data = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="test")

        self.train_dataset = self.filter_dataset(train_data, self.min_len, self.max_len)
        self.test_dataset = self.filter_dataset(test_data, self.min_len,self.max_len)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        
    @staticmethod
    def filter_dataset(dataset, min_len: int, max_len: int) -> list[dict[str, str]]:
        return dataset.filter(lambda row: min_len < len(row["text"].split()) <= max_len)
