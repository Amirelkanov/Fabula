# dataloader.py
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
"""
class ShakespeareDataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.samples = []
        
        dataset = load_dataset("tiny_shakespeare", split="train")
        text = dataset["text"]
        # TODO:
        print(f"Loaded {len(text)} characters of Shakespeare text")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def get_dataloader(tokenizer, batch_size=1):
    dataset = ShakespeareDataset(tokenizer)
    print(f"Created Shakespeare dataset with {len(dataset)} samples")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)"""