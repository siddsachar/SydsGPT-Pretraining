import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class Dataset(Dataset):
    def __init__(self, text, tokenizer, max_length, step_size):
        self.input_ids = []
        self.target_ids = []

        encoded_text = tokenizer.encode(text)

        for i in range(0, len(encoded_text) - max_length, step_size):
            input_chunk = encoded_text[i:i + max_length]
            target_chunk = encoded_text[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

def create_dataloader(text, max_length = 512, step_size = 256, batch_size = 8, drop_last = True, shuffle = True, num_workers = 0):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = Dataset(text,tokenizer, max_length, step_size)
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers
    )

    return dataloader