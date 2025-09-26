import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 文本数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels, word2vec_model, max_len=100):
        self.texts = texts
        self.labels = labels
        self.word2vec_model = word2vec_model
        self.max_len = max_len
        self.embedding_dim = word2vec_model.vector_size

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 将文本转换为词向量序列
        embeddings = []
        for word in text.split():
            if word in self.word2vec_model:
                embeddings.append(torch.tensor(self.word2vec_model[word], dtype=torch.float32))
            if len(embeddings) >= self.max_len:
                break

        # 填充或截断到固定长度
        if len(embeddings) < self.max_len:
            padding = torch.zeros(self.max_len - len(embeddings), self.embedding_dim, dtype=torch.float32)
            embeddings = torch.cat([torch.stack(embeddings), padding])
        else:
            embeddings = torch.stack(embeddings[:self.max_len])

        return embeddings, torch.tensor(label, dtype=torch.long)

# 双向LSTM + 注意力机制的文本特征提取模块
class BiLSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1):
        super(BiLSTMAttention, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        attn_weights = self.attention(lstm_out)  # attn_weights shape: (batch_size, seq_len, 1)
        context = torch.bmm(attn_weights.transpose(1, 2), lstm_out)  # context shape: (batch_size, 1, hidden_dim * 2)
        return context.squeeze(1)  # (batch_size, hidden_dim * 2)

# 多层感知机 + BatchNorm的深度分类模型
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super(MLPClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

# 完整模型（特征提取 + 分类）
class TextClassificationModel(nn.Module):
    def __init__(self, embedding_dim, lstm_hidden_dim, mlp_hidden_dims, num_classes):
        super(TextClassificationModel, self).__init__()
        self.feature_extractor = BiLSTMAttention(embedding_dim, lstm_hidden_dim)
        self.classifier = MLPClassifier(lstm_hidden_dim * 2, mlp_hidden_dims, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        train_acc = train_correct / train_total
        train_loss = train_loss / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_acc = val_correct / val_total
        val_loss = val_loss / val_total

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    return best_val_acc

# 预测函数
def predict(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
    return predictions

# 主函数
def main():
    # 加载预训练Word2Vec模型
    print("Loading pre-trained Word2Vec model...")
    word2vec_model = api.load("word2vec-google-news-300")  # 可根据需要更换为其他预训练模型或自定义训练的模型

    # 假设从CSV文件加载数据，这里用示例数据模拟
    data = {
        'text': [
            'user needs model responses matching',
            'dialogue text matching degree high',
            'user needs and model responses low matching',
            'key semantics in dialogue important',
            'tie labels few samples present'
        ],
        'label': ['class1', 'class2', 'class3', 'class1', 'class3']
    }
    df = pd.DataFrame(data)

    # 标签编码
    label_encoder = LabelEncoder()
    df['encoded_label'] = label_encoder.fit_transform(df['label'])
    num_classes = len(label_encoder.classes_)

    # 划分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['encoded_label'].tolist(), test_size=0.2, random_state=42
    )

    # 创建数据集和数据加载器
    train_dataset = TextDataset(train_texts, train_labels, word2vec_model)
    val_dataset = TextDataset(val_texts, val_labels, word2vec_model)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 模型初始化
    embedding_dim = word2vec_model.vector_size
    lstm_hidden_dim = 128
    mlp_hidden_dims = [64, 32]
    model = TextClassificationModel(embedding_dim, lstm_hidden_dim, mlp_hidden_dims, num_classes)

    # 定义损失函数（带类别权重，这里简单示例，实际可根据数据类别分布计算）
    class_weights = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32)  # 假设'tie'标签（class3）样本少，权重设为2
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_acc = train_model(model, train_loader, val_loader, criterion, optimizer, 10, device)
    print(f'Best validation accuracy: {best_acc:.4f}')

    # 加载最佳模型进行预测（这里用训练好的模型直接预测验证集示例）
    model.load_state_dict(torch.load('best_model.pth'))
    predictions = predict(model, val_loader, device)
    print("Predictions:", predictions)
    print("True labels:", val_labels)

if __name__ == "__main__":
    main()