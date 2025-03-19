import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import random_split, DataLoader, Dataset
import tqdm
import os
import gin
from rave.model import RAVE


class AudioChunksDataset(Dataset):
    def __init__(self, audio_dir, chunk_size=513, sample_rate=48000):
        self.audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")]
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.data = []
        # Podział dźwięku na fragmenty podczas inicjalizacji
        for file_path in self.audio_files:
            waveform, sr = torchaudio.load(file_path)

            # Konwersja do mono, jeśli dźwięk jest stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)  # [1, T]

            # Dzielimy plik na fragmenty o rozmiarze chunk_size
            num_chunks = waveform.shape[1] // chunk_size  # Ile pełnych segmentów zmieści się w pliku

            for i in range(num_chunks):
                chunk = waveform[:, i * chunk_size: (i + 1) * chunk_size]  # Wyciągamy fragment
                self.data.append(chunk.requires_grad_(False))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]  # Pobranie fragmentu
        return chunk


class LitAutoEncoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=64, sparsity_target=0.05, sparsity_weight=0.001):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(nn.Linear(input_dim, latent_dim), nn.ReLU())

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
        )
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


def performSAE(output, sae, sae_diff, bottlneck):
    z, out = sae(output)
    sae_diff.append((out, output))
    bottlneck.append(z)


def loadGinConfig(config_paths: list[str]):
    gin.clear_config()  # Czyścimy wcześniejszą konfigurację
    for config_path in config_paths:
        gin.parse_config_file(config_path)  # Wczytanie konfiguracji


def createRaveModelFormGinConfigFile(config_paths: list[str]):
    loadGinConfig(config_paths)  # Wczytanie konfiguracji
    model = RAVE()  # RAVE pobierze wartości z Gin Config
    return model


def main():
    model_path = "./darbouka_onnx.ts"
    model_with_weights = torch.jit.load(model_path)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model_with_weights.to(DEVICE)
    model_with_weights.eval()
    # state_dict_weights = model_with_weights.state_dict()
    # model = createRaveModelFormGinConfigFile(["configs/v2.gin", "configs/onnx.gin"])
    # model.to(DEVICE)
    # model.eval()
    # model.load_state_dict(state_dict_weights, strict=True)
    # matching_keys = [layer_name for layer_name in state_dict_weights.keys() if layer_name in model.state_dict().keys()]
    # unmatching_keys = [layer_name for layer_name in state_dict_weights.keys() if layer_name not in model.state_dict().keys()]
    # print(matching_keys)
    # print(unmatching_keys)
    dataset = AudioChunksDataset(audio_dir="./")
    dl = lambda x, s: DataLoader(x, batch_size=16, shuffle=s, pin_memory=True if torch.cuda.is_available() else
                                 False, generator=torch.Generator().manual_seed(42))
    train_ds, val_ds = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
    train_dl, val_dl = dl(train_ds, True), dl(val_ds, False)
    last_encoder_layer_name, last_encoder_layer_param = list(model_with_weights.encoder.named_parameters())[-1]
    n = last_encoder_layer_param.shape[1]
    sae = LitAutoEncoder(input_dim=n, latent_dim=5 * n).to(DEVICE)
    sae_diff = []
    bottlneck = []
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
    a_coef = 1e-3
    epochs = 200
    with tqdm.tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            sae.train()
            sae_diff, bottlneck, total_loss = [], [], 0
            for batch in train_dl:
                training_batch = batch.to(DEVICE)
                output1 = model_with_weights.pqmf(training_batch).detach()
                output2 = model_with_weights.encoder(output1).detach()
                performSAE(output2.squeeze(-1), sae, sae_diff, bottlneck)
                loss = torch.norm(sae_diff[-1][0]-sae_diff[-1][1]) + a_coef*torch.norm(bottlneck[-1], p=1)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            sae.eval()
            with torch.no_grad():
                sae_diff, bottlneck, val_loss = [], [], 0
                for batch in val_dl:
                    training_batch = batch.to(DEVICE)
                    output1 = model_with_weights.pqmf(training_batch)
                    output2 = model_with_weights.encoder(output1)
                    performSAE(output2.squeeze(-1), sae, sae_diff, bottlneck)
                    val_loss += torch.norm(sae_diff[-1][0]-sae_diff[-1][1]) + a_coef*torch.norm(bottlneck[-1], p=1).item()
            pbar.set_postfix_str(f'epoch: {epoch}, loss: {total_loss:.3f} val_los::{val_loss:.3f}')
            pbar.update(1)


if __name__ == "__main__":
    main()
