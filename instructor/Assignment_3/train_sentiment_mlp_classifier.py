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

dataset = load_dataset('financial_phrasebank', 'sentences_50agree', trust_remote_code=True)
ft_model = KeyedVectors.load("fasttext-wiki-news-subwords-300.model")

def get_emb(text):
    vecs = [ft_model[w] for w in text.lower().split() if w in ft_model]
    return np.mean(vecs, axis=0) if vecs else np.zeros(300)

sentences = dataset['train']['sentence']
labels = np.array(dataset['train']['label'])


X_all = np.array([get_emb(x) for x in sentences])
X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(X_all, labels, test_size=0.3, stratify=labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

X_train, y_train = torch.FloatTensor(X_train_raw), torch.LongTensor(y_train_raw)
X_val, y_val = torch.FloatTensor(X_val), torch.LongTensor(y_val)
X_test, y_test = torch.FloatTensor(X_test), torch.LongTensor(y_test)

weights = compute_class_weight('balanced', classes=np.unique(labels), y=y_train.numpy())
class_weights = torch.FloatTensor(weights)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=256, num_classes=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMLP().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': [], 'train_acc': [], 'val_acc': []}
best_val_f1 = 0.0

print(f"Starting training on {device}...")
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
        torch.save(model.state_dict(), 'outputs/best_mlp_model.pth')
    
    print(f"Epoch {epoch+1:02d} | Train F1: {history['train_f1'][-1]:.4f} ")


    if epoch >= 30 and (epoch - np.argmax(history['val_f1']) >= 10):
        print("Early stopping triggered.")
        break

plt.figure(figsize=(18, 5))
metrics = ['loss', 'acc', 'f1']
for i, m in enumerate(metrics):
    plt.subplot(1, 3, i+1)
    plt.plot(history[f'train_{m}'], label='Train')
    plt.plot(history[f'val_{m}'], label='Val')
    plt.title(f'MLP {m.capitalize()} vs Epochs'); plt.legend()
plt.savefig('outputs/mlp_training_report.png')

model.load_state_dict(torch.load('outputs/best_mlp_model.pth'))
model.eval()
with torch.no_grad():
    test_preds = model(X_test.to(device)).argmax(1).cpu().numpy()

print(f"\nFinal Test Macro F1: {f1_score(y_test, test_preds, average='macro'):.4f}")

cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted Labels'); plt.ylabel('True Labels')
plt.title('MLP Test Confusion Matrix'); plt.savefig('outputs/mlp_confusion_matrix.png')