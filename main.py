import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

df = pd.read_csv("content/IMDB Dataset.csv", names=["text", "label"], skiprows=[0])

df['text'] = df['text'].str.lower().str.split()

le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

vocab = {word for phrase in df['text'] for word in phrase}
word_to_idx = {word: idx for idx, word in enumerate(vocab, start=1)}

max_length = df['text'].str.len().max()

def encode_and_pad(text):
    encoded = [word_to_idx[word] for word in text]
    return [0] * (max_length - len(encoded)) + encoded

train_data['text'] = train_data['text'].apply(encode_and_pad)
test_data['text'] = test_data['text'].apply(encode_and_pad)

class SentimentDataset(Dataset):
    def __init__(self, data):
        self.texts = data['text'].values
        self.labels = data['label'].values

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return torch.tensor(text, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    
train_dataset = SentimentDataset(train_data)
test_dataset = SentimentDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, middle_size, output_size):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, middle_size)
        self.fc2 = nn.Linear(middle_size, output_size)


    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc1(out[:, -1, :])
        out = F.dropout(F.relu(out))
        out = self.fc2(out)
        return out
    
vocab_size = len(vocab) + 1
embed_size = 128
hidden_size = 128
middle_size = 32
output_size = 2
model = SentimentRNN(vocab_size, embed_size, hidden_size, middle_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

num_epochs = 10

def train():
    model.train()
    epoch_loss = 0
    for texts, labels in train_loader:
        
        outputs = model(texts)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        epoch_loss = 0
        for texts, labels in test_loader:
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            
        accuracy = 100.0 * correct / total
        print(f'Accuracy: {accuracy:.2f}%, Loss: {epoch_loss / len(test_loader):.4f}')


test()
for epoch in range(num_epochs):
    train()
    test()
