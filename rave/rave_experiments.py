import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import random_split, DataLoader, Dataset
import tqdm
import os


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


def getDevice():
    return torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def prepareModel(path: str, device: str):
    model_with_weights = torch.jit.load(path)
    model_with_weights.to(device)
    model_with_weights.eval()
    return model_with_weights


def createDataloader(dataset, shuffle: bool):
    return DataLoader(dataset, batch_size=16, shuffle=shuffle, pin_memory=True if torch.cuda.is_available() else
                      False, generator=torch.Generator().manual_seed(42))


def prepareDataloaders(audio_dir: str):
    dataset = AudioChunksDataset(audio_dir)
    train_ds, val_ds = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
    train_dl, val_dl = createDataloader(train_ds, True), createDataloader(val_ds, False)
    return train_dl, val_dl


def prepareSAE(model, train_dl: DataLoader, device: str):
    output = model.encode(next(iter(train_dl)).to(device)).detach()
    sae = LitAutoEncoder(input_dim=output.shape[1], latent_dim=3 * output.shape[1]).to(device)
    output = output.to("cpu")
    output = None
    return sae


def loss(sae_diff: list[torch.Tensor], bottlneck: list[torch.Tensor], a_coef: float):
    return torch.norm(sae_diff[-1][0]-sae_diff[-1][1]) + a_coef*torch.norm(bottlneck[-1], p=1)


def getActivation(model, training_batch):
    output = model.encode(training_batch).detach()
    return output.squeeze(-1)


def prepareTrainingHiperparams():
    hiperparams = {"loss": loss, "loss_params": {"sae_diff": [], "bottlneck": [], "a_coef": 1e-3},
                   "epochs": 180, "lr": 1e-3, "activation": getActivation}
    return hiperparams


def train(model, sae: nn.Module, hiperparams: dict, train_dl: DataLoader, val_dl: DataLoader, device: str,
          optimizer: torch.optim.Optimizer):
    with tqdm.tqdm(total=hiperparams["epochs"]) as pbar:
        sae_diff = hiperparams["loss_params"]["sae_diff"]
        bottlneck = hiperparams["loss_params"]["bottlneck"]
        a_coef = hiperparams["loss_params"]["a_coef"]
        for epoch in range(hiperparams["epochs"]):
            sae.train()
            sae_diff, bottlneck, total_loss = [], [], 0
            for batch in train_dl:
                training_batch = batch.to(device).detach()
                output = hiperparams["activation"](model, training_batch)
                performSAE(output, sae, sae_diff, bottlneck)
                loss = hiperparams["loss"](sae_diff, bottlneck, a_coef)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            sae.eval()
            with torch.no_grad():
                sae_diff, bottlneck, val_loss = [], [], 0
                for batch in val_dl:
                    training_batch = batch.to(device).detach()
                    output = hiperparams["activation"](model, training_batch)
                    performSAE(output, sae, sae_diff, bottlneck)
                    val_loss += hiperparams["loss"](sae_diff, bottlneck, a_coef).item()
            pbar.set_postfix_str(f'epoch: {epoch}, loss: {total_loss:.3f} val_los::{val_loss:.3f}')
            pbar.update(1)


def experiment(model_path: str, audio_folder_path: str):
    DEVICE = getDevice()
    model = prepareModel(model_path, DEVICE)
    train_dl, val_dl = prepareDataloaders(audio_folder_path)
    sae = prepareSAE(model, train_dl, DEVICE)
    hiperparams = prepareTrainingHiperparams()
    optimizer = torch.optim.Adam(sae.parameters(), lr=hiperparams["lr"])
    train(model, sae, hiperparams, train_dl, val_dl, DEVICE, optimizer)
