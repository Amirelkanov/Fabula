# dataloader.py
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

class ShakespeareDataset(Dataset):
    def __init__(self, tokenizer, max_length=512, chunk_size=1000, overlap=100):
        self.tokenizer = tokenizer
        self.samples = []
        
        dataset = load_dataset("tiny_shakespeare", split="train")
        text = dataset["text"]
        print(f"Loaded {len(text)} characters of Shakespeare text")
        
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk) > 50: 
                encodings = tokenizer(chunk, truncation=True, max_length=max_length, 
                                     return_tensors="pt", padding="max_length")
                self.samples.append({
                    'input_ids': encodings['input_ids'],
                    'attention_mask': encodings['attention_mask']
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def get_dataloader(tokenizer, batch_size=1):
    dataset = ShakespeareDataset(tokenizer)
    print(f"Created Shakespeare dataset with {len(dataset)} samples")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)