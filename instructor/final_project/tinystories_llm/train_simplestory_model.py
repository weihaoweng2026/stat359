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
import matplotlib.pyplot as plt

class BPETokenizer:
    def __init__(self, vocab_size=10000, special_tokens=None):
        self.vocab_size = vocab_size
        self.bpe_codes = {}
        self.vocab = {}
        self.token2id = {}
        self.id2token = {}
        if special_tokens is None:
            self.special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>', '<user>', '<assistant>', '<system>']
        else:
            self.special_tokens = special_tokens

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        tokenizer = cls(vocab_size=state['vocab_size'], special_tokens=state.get('special_tokens'))
        tokenizer.bpe_codes = state['bpe_codes']
        tokenizer.vocab = state['vocab']
        tokenizer.token2id = state['token2id']
        tokenizer.id2token = state['id2token']
        return tokenizer

    def encode(self, text, add_special_tokens=False):
        tokens = []
        if add_special_tokens:
            tokens.append('<bos>')
        for word in text.strip().split():
            if word in self.special_tokens:
                tokens.append(word)
                continue
            word_chars = list(word) + ['</w>']
            while True:
                pairs = [(word_chars[i], word_chars[i+1]) for i in range(len(word_chars)-1)]
                pair_ranks = {pair: self.bpe_codes.get(pair, float('inf')) for pair in pairs}
                if not pair_ranks or min(pair_ranks.values()) == float('inf'):
                    break
                best_pair = min(pair_ranks, key=pair_ranks.get)
                i, new_word = 0, []
                while i < len(word_chars):
                    if i < len(word_chars) - 1 and (word_chars[i], word_chars[i+1]) == best_pair:
                        new_word.append(word_chars[i] + word_chars[i+1]); i += 2
                    else:
                        new_word.append(word_chars[i]); i += 1
                word_chars = new_word
            tokens.extend(word_chars)
        if add_special_tokens:
            tokens.append('<eos>')
        return [self.token2id.get(t, self.token2id.get('<unk>', 0)) for t in tokens]

class SimpleStoryDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=256, max_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print(f"Loading data from {jsonl_path}...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if max_samples:
                lines = lines[:max_samples]
            for line in lines:
                data = json.loads(line)
                self.examples.append(data['text'])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.examples[idx], add_special_tokens=True)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens += [self.tokenizer.token2id.get('<pad>', 0)] * (self.max_length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

class WarmupLinearScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr_scale = float(self.current_step) / float(max(1, self.warmup_steps))
        else:
            lr_scale = max(0.0, 1.0 - (self.current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps)))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group.get('initial_lr', 5e-4) * lr_scale

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="simplestory_text/my_simplestories.jsonl")
    parser.add_argument("--tokenizer_path", type=str, default="simplestory_model/bpe_tokenizer_simplestories.pkl")
    parser.add_argument("--output_dir", type=str, default="simplestory_model")
    parser.add_argument("--epochs", type=int, default=15) 
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_seq_len", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BPETokenizer.load(args.tokenizer_path)
    vocab_size = len(tokenizer.token2id)

    full_dataset = SimpleStoryDataset(args.input_file, tokenizer, args.max_seq_len, args.max_samples)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    from transformer_model import TinyStoriesConfig, TinyStoriesForCausalLM
    config = TinyStoriesConfig(
        vocab_size=vocab_size, hidden_size=256, num_hidden_layers=4,
        num_attention_heads=8, max_position_embeddings=args.max_seq_len
    )
    model = TinyStoriesForCausalLM(config).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    for pg in optimizer.param_groups: pg['initial_lr'] = args.lr
    
    total_steps = len(train_loader) * args.epochs
    scheduler = WarmupLinearScheduler(optimizer, 500, total_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token2id.get('<pad>', 0))

    train_loss_history = []
    val_loss_history = []

    print(f"Starting Training on {device} | Epochs: {args.epochs} | Vocab Size: {vocab_size}")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            
            logits = model(input_ids=inputs)["logits"]
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                inputs, targets = batch[:, :-1], batch[:, 1:]
                logits = model(input_ids=inputs)["logits"]
                loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        print(f"Epoch {epoch+1} Done. Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth"))

    history = {"train_loss": train_loss_history, "val_loss": val_loss_history}
    with open(os.path.join(args.output_dir, "loss_history.json"), "w") as f:
        json.dump(history, f)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), train_loss_history, label='Train Loss', marker='o')
    plt.plot(range(1, args.epochs + 1), val_loss_history, label='Val Loss', marker='x')
    plt.title("SimpleStory Training Curve (15 Epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "loss_curve.png"))
    print(f"Loss curve saved to {args.output_dir}/loss_curve.png")

    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_chat_model.pth"))
    print("Training Complete!")

if __name__ == "__main__":
    train()