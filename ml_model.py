import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import AutoTokenizer, AutoModel


class BertEncoder:
    """
    Класс-обёртка, извлекающий эмбеддинги (pooler_output или скрытые состояния) из BERT.
    По умолчанию берём 'bert-base-uncased' c выходом ~768.
    """

    def __init__(self, model_name='bert-base-uncased', device='cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_embeddings(self, texts, max_length=128):
        """
        Возвращает тензор эмбеддингов (batch_size, 768).
        Усредняем last_hidden_state по всем токенам.
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
            # Берём среднее по seq_len:
            embeddings = torch.mean(last_hidden, dim=1)
        return embeddings


class AutoEncoder(nn.Module):
    """
    Простой MLP-автоэнкодер (768 -> 64 -> 768), как в вашем примере.
    """

    def __init__(self, input_dim=768, hidden_dim=256):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def train_autoencoder(
        texts,
        bert_encoder,
        autoencoder,
        num_epochs=3,
        batch_size=32,
        lr=1e-4,
        device='cpu'
):
    """
    Тренируем автоэнкодер на эмбеддингах, вычисляемых для списка texts (строк).
    """
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Чтобы не вызывать get_embeddings для каждой строки, будем обрабатывать батчами.
    autoencoder.train()
    n = len(texts)
    for epoch in range(num_epochs):
        # Перемешиваем индексы:
        indices = np.random.permutation(n)
        total_loss = 0.0
        count = 0

        for i in range(0, n, batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_texts = [texts[j] for j in batch_idx]

            # Генерируем BERT-эмбеддинги
            emb = bert_encoder.get_embeddings(batch_texts)  # shape: (B, 768)
            emb = emb.to(device)

            optimizer.zero_grad()
            outputs = autoencoder(emb)
            loss = loss_fn(outputs, emb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * emb.size(0)
            count += emb.size(0)

        avg_loss = total_loss / count
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}")


def compute_reconstruction_errors(
        texts,
        bert_encoder,
        autoencoder,
        device='cpu',
        batch_size=32
):
    """
    Считает ошибку реконструкции (MSE) для каждой строки из texts.
    Возвращает список (или массив) ошибок.
    """
    autoencoder.eval()
    n = len(texts)
    errors = np.zeros(n, dtype=np.float32)

    idx = 0
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_texts = texts[i:i + batch_size]
            emb = bert_encoder.get_embeddings(batch_texts).to(device)
            recon = autoencoder(emb)
            # построчно MSE
            mse_batch = torch.mean((recon - emb) ** 2, dim=1).cpu().numpy()
            errors[idx:idx + len(mse_batch)] = mse_batch
            idx += len(mse_batch)

    return errors
