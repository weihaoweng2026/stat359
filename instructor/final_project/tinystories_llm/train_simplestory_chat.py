import os
import torch
import argparse
import json
import random
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
from tqdm import tqdm

class BPETokenizer:
    def __init__(self, vocab_size=5000, special_tokens=None):
        self.vocab_size = vocab_size
        self.token2id = {}
        self.id2token = {}
        self.bpe_codes = {}
        self.special_tokens = special_tokens or ['<pad>', '<unk>', '<bos>', '<eos>', '<user>', '<assistant>', '<system>']

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        tokenizer = cls(vocab_size=state['vocab_size'], special_tokens=state.get('special_tokens'))
        tokenizer.bpe_codes = state['bpe_codes']
        tokenizer.token2id = state['token2id']
        tokenizer.id2token = state['id2token']
        return tokenizer

    def encode(self, text, add_special_tokens=True):
        tokens = []
        if add_special_tokens: tokens.append('<bos>')
        return [self.token2id.get(t, self.token2id.get('<unk>', 0)) for t in tokens]

class SimpleStoryChatDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        print(f"Loading local dataset: {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.examples.append(data['text'])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.examples[idx], add_special_tokens=True)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            pad_id = self.tokenizer.token2id.get('<pad>', 0)
            tokens += [pad_id] * (self.max_length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="simplestory_text/my_simplestories.jsonl")
    parser.add_argument("--tokenizer_path", type=str, default="simplestory_model/bpe_tokenizer_simplestories.pkl")
    parser.add_argument("--output_dir", type=str, default="simplestory_model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--amp", action="store_true", help="Enable Mixed Precision training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BPETokenizer.load(args.tokenizer_path)
    vocab_size = len(tokenizer.token2id)
    dataset = SimpleStoryChatDataset(args.input_file, tokenizer, args.max_seq_len)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    from transformer_model import TinyStoriesConfig, TinyStoriesForCausalLM
    config = TinyStoriesConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=args.max_seq_len
    )
    model = TinyStoriesForCausalLM(config).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, 
        pct_start=0.1, anneal_strategy='linear'
    )

    scaler = torch.amp.GradScaler(device="cuda", enabled=(args.amp and device.type == "cuda"))
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token2id.get('<pad>', 0))

    print(f"Start Professional Training | Device: {device} | Steps: {total_steps}")

    

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            

            with torch.amp.autocast(device_type="cuda", enabled=(args.amp and device.type == "cuda")):
                logits = model(input_ids=inputs)["logits"]
                loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            
            optimizer.zero_grad()
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.6f}"})


        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                inputs, targets = batch[:, :-1], batch[:, 1:]
                logits = model(input_ids=inputs)["logits"]
                loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
                val_loss += loss.item()
        
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")

        torch.save(model.state_dict(), os.path.join(args.output_dir, f"chat_model_epoch_{epoch+1}.pth"))

    final_path = os.path.join(args.output_dir, "final_chat_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f" Training finished! Model saved at: {final_path}")

if __name__ == "__main__":
    train()