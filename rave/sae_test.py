import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split, DataLoader, Dataset
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
                chunk = waveform[:, i * chunk_size : (i + 1) * chunk_size]  # Wyciągamy fragment
                self.data.append(chunk)

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

def perform_sae(output):
    z, out = sae(output)
    sae_diff.append((out, output))
    bottlneck.append(z)

model_path = "./darbouka_onnx.ts"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

model = torch.jit.load(model_path)

model.to(DEVICE)

model.train()

layers_dict = {}

for name, module in model.named_modules():
    layers_dict[name] = module

# waveform, sr = torchaudio.load("./drums-loop-darbuka_91bpm.wav")

# # Konwersja stereo → mono
# waveform = waveform.mean(dim=0, keepdim=True)  # [1, 349884]

# # Dopasowanie długości do 513 próbek
# waveform = F.interpolate(waveform.unsqueeze(0), size=513, mode="linear")  # [1, 1, 513]

# # Powielenie batch size do 16
# waveform = waveform.expand(16, -1, -1)  # [16, 1, 513]

# waveform = waveform.to(DEVICE)

dataset = AudioChunksDataset(audio_dir="./")
dl = lambda x, s: DataLoader(x, batch_size=16, shuffle=s, pin_memory=True if torch.cuda.is_available() else False)
train_ds, val_ds = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
train_dl, val_dl = dl(train_ds, True), dl(val_ds, False)
print(len(train_dl))
print(len(val_dl))

first_layer_name, first_layer_param = list(layers_dict["pqmf"].named_parameters())[0]
print(first_layer_param.shape)
last_encoder_layer_name, last_encoder_layer_param = list(layers_dict["encoder"].named_parameters())[-1]

n = last_encoder_layer_param.shape[1]
sae = LitAutoEncoder(input_dim=n, latent_dim=5 * n).to(DEVICE)
sae_diff = []
bottlneck = []

optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
a_coef = 1e-3
epochs = 150
with tqdm.tqdm(total=epochs) as pbar:
    for epoch in range(epochs):
        sae.train()
        model.train()
        sae_diff, bottlneck, total_loss = [], [], 0
        for batch in train_dl:
            training_batch = batch.to(DEVICE)
            output1 = layers_dict["pqmf"](training_batch).detach()
            output2 = layers_dict["encoder"](output1).detach()
            perform_sae(output2.squeeze(-1))
            loss = torch.norm(sae_diff[-1][0]-sae_diff[-1][1]) + a_coef*torch.norm(bottlneck[-1], p=1)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        sae.eval()
        with torch.no_grad():
            sae_diff, bottlneck, val_loss = [], [], 0
            for batch in val_dl:
                # batch = batch['encoded_music'].to(DEVICE)
                training_batch = batch.to(DEVICE)
                output1 = layers_dict["pqmf"](training_batch)
                output2 = layers_dict["encoder"](output1)
                perform_sae(output2.squeeze(-1))
                val_loss += torch.norm(sae_diff[-1][0]-sae_diff[-1][1]) + a_coef*torch.norm(bottlneck[-1], p=1).item()
                
        pbar.set_postfix_str(f'epoch: {epoch}, loss: {total_loss:.3f} val_los::{val_loss:.3f}')
        pbar.update(1)

# with torch.no_grad():
#     output1 = layers_dict["pqmf"](waveform)
#     output2 = layers_dict["encoder"](output1)
#     perform_sae(output2.squeeze(-1))
    
# print("Output 1 shape:", output1.shape)
# print("Output 2 shape:", output2.shape)