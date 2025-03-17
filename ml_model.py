import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel

class BertEncoder:
    """
    Класс-обёртка, извлекающий эмбеддинги (pooler_output или скрытые состояния) из BERT.
    """
    def __init__(self, model_name='bert-base-uncased', device='cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_embeddings(self, texts, max_length=128):
        """
        Возвращает тензор эмбеддингов для списка текстов.
        На выходе: torch.Tensor размера (batch_size, hidden_dim).
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
            # Можно использовать pooled_output либо среднее по скрытым состояниям
            # pooled_output = outputs.pooler_output
            last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
            # усредним по всем токенам (или берём CLS-токен – по задаче)
            embeddings = torch.mean(last_hidden_state, dim=1)
        return embeddings


class AutoEncoder(nn.Module):
    """
    Простой MLP-автоэнкодер для демонстрации.
    Допустим, вход 768-мерный (BERT base).
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
        num_epochs=5,
        batch_size=16,
        lr=1e-4,
        device='cpu'
):
    """
    Упрощённая тренировка автоэнкодера на текстовых логах.
    texts – список строк (логов)
    """
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

    # MSELoss
    mse_loss_fn = nn.MSELoss(reduction='mean')
    # Для MAE
    mae_loss_fn = nn.L1Loss(reduction='mean')

    autoencoder.train()
    for epoch in range(num_epochs):
        total_mse = 0.0
        total_mae = 0.0
        count = 0

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            embeddings = bert_encoder.get_embeddings(batch_texts).to(device)

            optimizer.zero_grad()
            outputs = autoencoder(embeddings)

            # Считаем MSE
            mse_loss = mse_loss_fn(outputs, embeddings)
            # Считаем MAE
            mae_loss = mae_loss_fn(outputs, embeddings)

            # Для обучения используем только MSE, например
            mse_loss.backward()
            optimizer.step()

            total_mse += mse_loss.item() * embeddings.size(0)  # умножаем на кол-во образцов в батче
            total_mae += mae_loss.item() * embeddings.size(0)
            count += embeddings.size(0)

        avg_mse = total_mse / count if count else 0.0
        avg_mae = total_mae / count if count else 0.0

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"MSE: {avg_mse:.4f}, MAE: {avg_mae:.4f}")


def compute_reconstruction_errors(texts, bert_encoder, autoencoder):
    """
    Считает ошибку реконструкции для каждого лога.
    Возвращает список (или тензор) ошибок.
    """
    autoencoder.eval()
    errors = []
    with torch.no_grad():
        for txt in texts:
            emb = bert_encoder.get_embeddings([txt])  # shape (1, 768)
            recon = autoencoder(emb)
            # MSE по батчу размером 1
            mse = torch.mean((recon - emb)**2, dim=1).item()
            errors.append(mse)
    return errors

def compute_reconstruction_errors_batch(
    texts,
    bert_encoder,
    autoencoder,
    device='cpu',
    batch_size=4
):
    """
    Батчевая версия. За один проход обрабатываем up to `batch_size` строк.
    """
    autoencoder.eval()
    errors = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            emb = bert_encoder.get_embeddings(batch_texts).to(device)  # (batch, hidden_dim)
            recon = autoencoder(emb)
            # mean((recon - emb)^2) построчно:
            batch_mse = torch.mean((recon - emb)**2, dim=1)
            # batch_mse shape: (batch,)
            errors.extend(batch_mse.cpu().numpy())
    return list(errors)

