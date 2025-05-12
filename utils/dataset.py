# utils/dataset.py
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        
        # Process each example
        for item in data:
            # Extract text based on format
            if isinstance(item, dict):
                if "instruction" in item and "response" in item:
                    text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}"
                elif "input" in item and "output" in item:
                    text = f"### User:\n{item['input']}\n\n### Assistant:\n{item['output']}"
                elif "prompt" in item and "completion" in item:
                    text = f"{item['prompt']}{item['completion']}"
                else:
                    continue
            else:
                # Skip non-dict items
                continue
                
            # Encode and add to dataset
            self.inputs.append(text)

    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, idx):
        # Tokenize input text
        encoded = self.tokenizer(
            self.inputs[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Extract and format tensors
        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()
        labels = input_ids.clone()
        
        # Apply label masking to padding tokens
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def load_dataset(filepath, format="auto"):
    """Load dataset from file based on format"""
    # Auto-detect format from file extension
    if format == "auto":
        if filepath.endswith(".json"):
            format = "json"
        elif filepath.endswith(".jsonl"):
            format = "jsonl"
        elif filepath.endswith(".csv"):
            format = "csv"
        else:
            raise ValueError(f"Cannot determine format for {filepath}")
    
    # Load based on format
    if format == "json":
        with open(filepath, 'r') as f:
            data = json.load(f)
    elif format == "jsonl":
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    elif format == "csv":
        df = pd.read_csv(filepath)
        data = df.to_dict('records')
    else:
        raise ValueError(f"Unsupported format: {format}")
        
    return data