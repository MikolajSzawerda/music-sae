import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Dataset, TensorDataset
import tqdm
import os
import librosa
import random
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import List


class AudioChunksDataset(Dataset):
    def __init__(self, audio_dir, chunk_size=513, sample_rate=48000, max_length: int = float("inf"), seed: int = 42):
        self._audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")]
        random.seed(seed)
        random.shuffle(self._audio_files)
        random.seed(None)
        self.sample_rate = sample_rate
        self._chunk_size = chunk_size
        self._data = []
        # Podział dźwięku na fragmenty podczas inicjalizacji
        for file_path in self._audio_files:
            waveform, sr = librosa.load(file_path, sr=sample_rate, mono=True)
            waveform = torch.tensor(waveform).unsqueeze(0)
            num_samples = waveform.shape[1]
            waveform = waveform[:, : num_samples - (num_samples % self._chunk_size)]
            chunks = waveform.view(-1, chunk_size)
            self._data.extend(chunks)
            if len(self._data) >= max_length:
                break
        if max_length != float("inf"):
            self._data = self._data[:max_length]
        random.seed(seed)
        random.shuffle(self._data)
        random.seed(None)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        frame = self._data[idx]
        return frame.unsqueeze(0)


class SparseAutoEncoder(nn.Module):
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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepareModel(path: str, device: str):
    model_with_weights = torch.jit.load(path)
    model_with_weights.to(device)
    model_with_weights.eval()
    return model_with_weights


def createDataloader(dataset, shuffle: bool, batch_size: int = 16):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True if torch.cuda.is_available() else
                      False, generator=torch.Generator().manual_seed(42), drop_last=True)


def prepareDataloaders(dataset: TensorDataset):
    train_ds, val_ds = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
    train_dl, val_dl = createDataloader(train_ds, False), createDataloader(val_ds, False)
    return train_dl, val_dl


def prepareSAE(train_dl: DataLoader, device: str, input_dim: int | None = None):
    in_channels = next(iter(train_dl)).shape[1]
    if input_dim:
        sae = SparseAutoEncoder(input_dim=input_dim, latent_dim=3 * input_dim).to(device)
    else:
        sae = SparseAutoEncoder(input_dim=in_channels, latent_dim=3 * in_channels).to(device)
    return sae


def sae_loss(sae_diff: list[torch.Tensor], bottlneck: list[torch.Tensor], a_coef: float):
    l1_norm = a_coef * torch.sum(torch.abs(bottlneck[-1]))
    return torch.norm(sae_diff[-1][0] - sae_diff[-1][1])**2 + l1_norm


def train(
    sae: nn.Module,
    hiperparams: dict,
    train_dl: list[torch.Tensor],
    val_dl: list[torch.Tensor],
    device: str,
    optimizer: torch.optim.Optimizer,
    train_losses: list,
    val_losses: list
):
    with tqdm.tqdm(total=hiperparams["epochs"]) as pbar:
        for epoch in range(hiperparams["epochs"]):
            sae.train()
            (
                hiperparams["loss_params"]["sae_diff"],
                hiperparams["loss_params"]["bottlneck"],
                total_loss,
            ) = [], [], 0
            for batch in train_dl:
                training_batch = batch.to(device).detach()
                performSAE(training_batch, sae, hiperparams["loss_params"]["sae_diff"],
                           hiperparams["loss_params"]["bottlneck"])
                loss = hiperparams["loss"](**hiperparams["loss_params"])
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_losses.append(total_loss)
            sae.eval()
            with torch.no_grad():
                (
                    hiperparams["loss_params"]["sae_diff"],
                    hiperparams["loss_params"]["bottlneck"],
                    val_loss,
                ) = [], [], 0
                for batch in val_dl:
                    training_batch = batch.to(device).detach()
                    performSAE(training_batch, sae, hiperparams["loss_params"]["sae_diff"],
                               hiperparams["loss_params"]["bottlneck"])
                    val_loss += hiperparams["loss"](**hiperparams["loss_params"]).item()
            val_losses.append(val_loss)
            pbar.set_postfix_str(f"epoch: {epoch}, loss: {total_loss:.3f} val_los::{val_loss:.3f}")
            pbar.update(1)


def prepareTrainingHiperparams():
    hiperparams = {"id": 0, "loss": sae_loss, "loss_params": {"sae_diff": [], "bottlneck": [], "a_coef": 1e-3},
                   "epochs": 25, "lr": 0.00001, "max_activations": float('inf')}
    return hiperparams


def saveLossPlots(train_losses: list[float], val_losses: list[float], filename: str) -> None:
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(19.2, 10.8))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()


def saveLossesToJson(losses: list[float], losses_name: str, filename: str):
    data = {
        losses_name: losses
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def findPtFiles(folder_path: str) -> List[Path]:
    folder = Path(folder_path)
    return sorted(list(folder.glob("*.pt")))


def experiment(activations_path: str, base_name: str, output_path: str, hiperparams: dict,
               sae_input_channels: int | None = None):
    DEVICE = getDevice()
    files_paths = findPtFiles(activations_path)
    train_losses = []
    val_losses = []
    for file_path in files_paths:
        activations = torch.load(file_path)
        if len(activations) > hiperparams["max_activations"]:
            activations = activations[:hiperparams["max_activations"]]
        train_dl, val_dl = random_split(activations, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
        sae = prepareSAE(train_dl, DEVICE, sae_input_channels)
        hiperparams = hiperparams
        optimizer = torch.optim.Adam(sae.parameters(), lr=hiperparams["lr"])
        train(sae, hiperparams, train_dl, val_dl, DEVICE, optimizer, train_losses, val_losses)
    torch.save(sae.state_dict(), output_path)
    saveLossPlots(train_losses, val_losses, "diagrams/" + base_name +
                  f"_loss_params_id_{hiperparams['id']}_sae.png")
    saveLossesToJson(train_losses, "train_losses", "diagrams_data/" + base_name +
                     f"_train_losses_params_id_{hiperparams['id']}_sae.json")
    saveLossesToJson(val_losses, "val_losses", "diagrams_data/" + base_name +
                     f"_val_losses_params_id_{hiperparams['id']}_sae.json")
