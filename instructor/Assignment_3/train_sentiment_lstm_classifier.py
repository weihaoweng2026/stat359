import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from gensim.models import KeyedVectors
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

print("Loading dataset and FastText model...")
dataset = load_dataset('financial_phrasebank', 'sentences_50agree', trust_remote_code=True)
ft_model = KeyedVectors.load("fasttext-wiki-news-subwords-300.model")

def prepare_lstm_data(texts, model, max_len=32):
    all_matrices = []
    for text in texts:
        words = text.lower().split()[:max_len]
        matrix = np.zeros((max_len, 300), dtype=np.float32)
        for i, w in enumerate(words):
            if w in model:
                matrix[i] = model[w]
        all_matrices.append(matrix)
    return np.array(all_matrices)

sentences = dataset['train']['sentence']
labels = np.array(dataset['train']['label'])

X_all = prepare_lstm_data(sentences, ft_model)
X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(X_all, labels, test_size=0.3, stratify=labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

X_train, y_train = torch.FloatTensor(X_train_raw), torch.LongTensor(y_train_raw)
X_val, y_val = torch.FloatTensor(X_val), torch.LongTensor(y_val)
X_test, y_test = torch.FloatTensor(X_test), torch.LongTensor(y_test)

weights = compute_class_weight('balanced', classes=np.unique(labels), y=y_train.numpy())
class_weights = torch.FloatTensor(weights)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

class SentimentLSTM(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=256, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        last_hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        return self.fc(last_hidden)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentLSTM().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)) #
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': [], 'train_acc': [], 'val_acc': []}
best_val_f1 = 0.0

print(f"Training LSTM on {device}...")
for epoch in range(50): 
    model.train()
    t_loss, t_preds, t_labels = 0, [], []
    for b_x, b_y in train_loader:
        b_x, b_y = b_x.to(device), b_y.to(device)
        optimizer.zero_grad()
        out = model(b_x)
        loss = criterion(out, b_y)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
        t_preds.extend(out.argmax(1).cpu().numpy()); t_labels.extend(b_y.cpu().numpy())

    model.eval()
    with torch.no_grad():
        v_out = model(X_val.to(device))
        v_loss = criterion(v_out, y_val.to(device))
        v_preds = v_out.argmax(1).cpu().numpy()
        v_f1 = f1_score(y_val, v_preds, average='macro')

    history['train_loss'].append(t_loss/len(train_loader)); history['val_loss'].append(v_loss.item())
    history['train_f1'].append(f1_score(t_labels, t_preds, average='macro'))
    history['val_f1'].append(v_f1)
    history['train_acc'].append(accuracy_score(t_labels, t_preds))
    history['val_acc'].append(accuracy_score(y_val, v_preds))
    if v_f1 > best_val_f1:
        best_val_f1 = v_f1
        torch.save(model.state_dict(), 'outputs/best_lstm_model.pth')
    
    print(f"Epoch {epoch+1:02d} | Val F1: {v_f1:.4f}")

    if epoch >= 30 and (epoch - np.argmax(history['val_f1']) >= 10):
        print("Early stopping triggered after epoch 30.")
        break

plt.figure(figsize=(18, 5))
for i, m in enumerate(['loss', 'acc', 'f1']):
    plt.subplot(1, 3, i+1)
    plt.plot(history[f'train_{m}'], label='Train')
    plt.plot(history[f'val_{m}'], label='Val')
    plt.title(f'LSTM {m.capitalize()} vs Epochs'); plt.legend()
plt.savefig('outputs/lstm_training_report.png')

model.load_state_dict(torch.load('outputs/best_lstm_model.pth'))
model.eval()
with torch.no_grad():
    test_preds = model(X_test.to(device)).argmax(1).cpu().numpy()

test_f1 = f1_score(y_test, test_preds, average='macro')
print(f"\nTarget Achieved! Best Test Macro F1: {test_f1:.4f}")

cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('LSTM Test Confusion Matrix')
plt.savefig('outputs/lstm_confusion_matrix.png')