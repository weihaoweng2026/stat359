import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 512  # change it to fit your memory constraints, e.g., 256, 128 if you run out of memory
EPOCHS = 5
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        return torch.tensor(self.pairs[idx][0]), torch.tensor(self.pairs[idx][1])
    


# Simple Skip-gram Module
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.in_em = nn.Embedding(vocab_size, embedding_dim)
        self.out_em = nn.Embedding(vocab_size, embedding_dim)
        init_range = 0.5 / embedding_dim
        self.in_em.weight.data.uniform_(-init_range, init_range)
        self.out_em.weight.data.zero_()

    def forward(self, center, context):
        v = self.in_em(center)
        u = self.out_em(context)
        v = v.unsqueeze(2)
        score = torch.bmm(u, v).squeeze(2)
        return score
    
    def get_embeddings(self):
        return self.in_em.weight.detach().cpu().numpy()

 
# Load processed data
with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)
# Precompute negative sampling distribution below
pairs_list = list(data['skipgram_df'].itertuples(index=False, name=None))
word2idx = data['word2idx']
idx2word = data['idx2word']
word_counts = data['counter']
vocab_size = len(word2idx)

counts = torch.tensor([word_counts[idx2word[i]] for i in range(vocab_size)])
noise_dist = counts.pow(0.75)
noise_dist = noise_dist/noise_dist.sum()

# Device selection: CUDA > MPS > CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset and DataLoader
dataset = SkipGramDataset(pairs_list)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, Loss, Optimizer
model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()


# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    for center, context in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        center = center.to(device)
        context = context.to(device)
        curr_batch_size = center.size(0)

        neg_samples = torch.multinomial(noise_dist, curr_batch_size * NEGATIVE_SAMPLES, replacement=True)
        neg_samples = neg_samples.view(curr_batch_size, NEGATIVE_SAMPLES).to(device)
        combined_indices = torch.cat([context.unsqueeze(1), neg_samples], dim=1)
        labels = torch.zeros(curr_batch_size, 1 + NEGATIVE_SAMPLES).to(device)
        labels[:, 0] = 1.0

        logits = model(center, combined_indices)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

# Save embeddings and mappings
embeddings = model.get_embeddings()
with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': data['word2idx'], 'idx2word': data['idx2word']}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
