import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Dataset, TensorDataset
import tqdm
import os
import librosa


# class AudioChunksDataset(Dataset):
#     def __init__(self, audio_dir, sample_rate=44100):
#         self.audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")]
#         self.sample_rate = sample_rate
#         self.data = []
#         # Podział dźwięku na fragmenty podczas inicjalizacji
#         for file_path in self.audio_files:
#             waveform, sr = librosa.load(file_path, sr=sample_rate, mono=True)
#             n_fft = 1024  # Rozmiar FFT (zgodny z 513 wartościami w modelu)
#             hop_length = 256  # Przesunięcie okna
#             fourier = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
#             magnitude = torch.abs(torch.Tensor(fourier))
#             spectrogram_frames = magnitude.T  # Transponujemy (teraz shape to [n_frames, 513])
#             self.data.extend(spectrogram_frames)
#         self.data = self.data[:len(self.data)//2]

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         frame = self.data[idx]
#         return frame.unsqueeze(0)


class AudioChunksDataset(Dataset):
    def __init__(self, audio_dir, chunk_size=513, sample_rate=44100):
        self.audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")]
        self.sample_rate = sample_rate
        self._chunk_size = chunk_size
        self._data = []
        # Podział dźwięku na fragmenty podczas inicjalizacji
        for file_path in self.audio_files:
            waveform, sr = librosa.load(file_path, sr=sample_rate, mono=True)
            waveform = torch.tensor(waveform).unsqueeze(0)
            num_samples = waveform.shape[1]
            waveform = waveform[:, :num_samples - (num_samples % self._chunk_size)]
            chunks = waveform.view(-1, chunk_size)
            self._data.extend(chunks)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        frame = self._data[idx]
        return frame.unsqueeze(0)


class LitAutoEncoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=64, sparsity_target=0.05, sparsity_weight=0.001):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(input_dim, latent_dim), nn.ReLU())

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
        )
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


def performSAE(batch, sae, sae_diff, bottlneck):
    z, out = sae(batch)
    sae_diff.append((out, batch))
    bottlneck.append(z)


def getDevice():
    return torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def prepareModel(path: str, device: str):
    model_with_weights = torch.jit.load(path)
    model_with_weights.to(device)
    model_with_weights.eval()
    return model_with_weights


def createDataloader(dataset, shuffle: bool):
    return DataLoader(dataset, batch_size=16, shuffle=shuffle, pin_memory=True if torch.cuda.is_available() else
                      False, generator=torch.Generator().manual_seed(42), drop_last=True)


def prepareDataloaders(dataset: TensorDataset):
    train_ds, val_ds = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
    train_dl, val_dl = createDataloader(train_ds, False), createDataloader(val_ds, False)
    return train_dl, val_dl


def prepareSAE(train_dl: DataLoader, device: str, input_dim: int | None = None):
    in_channels = next(iter(train_dl)).shape[1]
    if input_dim:
        sae = LitAutoEncoder(input_dim=input_dim, latent_dim=5 * input_dim).to(device)
    else:
        sae = LitAutoEncoder(input_dim=in_channels, latent_dim=5 * in_channels).to(device)
    return sae


def sae_loss(sae_diff: list[torch.Tensor], bottlneck: list[torch.Tensor], a_coef: float):
    return torch.norm(sae_diff[-1][0]-sae_diff[-1][1]) + a_coef*torch.norm(bottlneck[-1], p=1)


def train(sae: nn.Module, hiperparams: dict, train_dl: list[torch.Tensor], val_dl: list[torch.Tensor],
          device: str,
          optimizer: torch.optim.Optimizer, output_path: str):
    with tqdm.tqdm(total=hiperparams["epochs"]) as pbar:
        for epoch in range(hiperparams["epochs"]):
            sae.train()
            hiperparams["loss_params"]["sae_diff"], hiperparams["loss_params"]["bottlneck"], total_loss = [], [], 0
            for batch in train_dl:
                training_batch = batch.to(device).detach()
                performSAE(training_batch, sae, hiperparams["loss_params"]["sae_diff"],
                           hiperparams["loss_params"]["bottlneck"])
                loss = hiperparams["loss"](**hiperparams["loss_params"])
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            sae.eval()
            with torch.no_grad():
                hiperparams["loss_params"]["sae_diff"], hiperparams["loss_params"]["bottlneck"], val_loss = [], [], 0
                for batch in val_dl:
                    training_batch = batch.to(device).detach()
                    performSAE(training_batch, sae, hiperparams["loss_params"]["sae_diff"],
                               hiperparams["loss_params"]["bottlneck"])
                    val_loss += hiperparams["loss"](**hiperparams["loss_params"]).item()
            pbar.set_postfix_str(f'epoch: {epoch}, loss: {total_loss:.3f} val_los::{val_loss:.3f}')
            pbar.update(1)
        torch.save(sae.state_dict(), output_path)


def prepareTrainingHiperparams():
    hiperparams = {"loss": sae_loss, "loss_params": {"sae_diff": [], "bottlneck": [], "a_coef": 1e-3},
                   "epochs": 500, "lr": 1e-3}
    return hiperparams


def experiment(activations_path: str, output_path: str, hiperparams: dict, sae_input_channels: int | None = None):
    DEVICE = getDevice()
    activations = torch.load(activations_path)
    train_dl, val_dl = random_split(activations, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
    sae = prepareSAE(train_dl, DEVICE, sae_input_channels)
    hiperparams = hiperparams
    optimizer = torch.optim.Adam(sae.parameters(), lr=hiperparams["lr"])
    train(sae, hiperparams, train_dl, val_dl, DEVICE, optimizer, output_path)
