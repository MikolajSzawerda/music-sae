import argparse
import torch
import numpy as np
from rave_experiments import SparseAutoEncoder
import sys
from tqdm import tqdm


def getCMDArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("activations_path", type=str, help="Path to the file with saved tensors")
    parser.add_argument("batches_path", type=str,
                        help="Path to the file with batches corresponding to saved activation tensors")
    parser.add_argument("weights_path", type=str,
                        help="Path to the file with btrained sae wights")
    parser.add_argument("output_path", type=str, help="Path to the file with encoded tensors")
    args = parser.parse_args()
    return args


def main():
    args = getCMDArgs()
    activations = torch.load(args.activations_path)
    batches = torch.load(args.batches_path)
    new_activations = []
    for activation in activations:
        tensors = list(activation)
        for tensor in tensors:
            new_activations.append(tensor)
    new_batches = []
    for batch in batches:
        tensors = list(batch)
        for tensor in tensors:
            new_batches.append(tensor)
    # max_length = 800
    # if len(new_activations) > max_length:
    #     new_activations = new_activations[:max_length]
    # if len(new_batches) > max_length:
    #     new_batches = new_batches[:max_length]
    X = [activation.cpu() for activation in new_activations]
    batches = [batch.cpu().numpy() for batch in new_batches]
    sae = SparseAutoEncoder(input_dim=X[0].shape[0], latent_dim=5*X[0].shape[0])
    sae.to("cpu")
    sae.load_state_dict(torch.load(args.weights_path))
    X_coded = []
    with torch.no_grad():
        for input in tqdm(X):
            X_coded.append(sae(input)[0].numpy())
    X_coded = np.asarray(X_coded)
    torch.save(X_coded, args.output_path)


if __name__ == "__main__":
    sys.argv = ["dictionary_learning.py",
                "./darbouka_decoder_5.pt",
                "./darbouka_decoder_batches_5.pt",
                "./sae_darbouka_decoder_5.pth",
                "darbouka_decoder_5_encoded.pt"]
    main()
