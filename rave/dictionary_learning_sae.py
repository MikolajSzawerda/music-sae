import argparse
import torch
import numpy as np
from rave_experiments import SparseAutoEncoder, findPtFiles
from tqdm import tqdm


def getCMDArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("activations_path", type=str, help="Path to the folder with saved tensors")
    parser.add_argument("weights_path", type=str,
                        help="Path to the file with trained sae wights")
    parser.add_argument("multiply_factor", type=int,
                        help="Latent dim multiply factor")
    parser.add_argument("output_path", type=str, help="First part of the path to the file with encoded tensors")
    args = parser.parse_args()
    return args


def main():
    args = getCMDArgs()
    files_paths = findPtFiles(args.activations_path)
    for iterator, file_path in enumerate(files_paths):
        activations = torch.load(file_path)
        sae = SparseAutoEncoder(input_dim=activations[0].shape[1],
                                latent_dim=args.multiply_factor*activations[0].shape[1])
        sae = sae.to("cuda")
        sae.load_state_dict(torch.load(args.weights_path))
        X_coded = []
        with torch.no_grad():
            for input in tqdm(activations):
                temp = sae(input)[0]
                for tensor in temp:
                    X_coded.append(tensor.cpu().numpy())
        X_coded = np.asarray(X_coded)
        torch.save(X_coded, args.output_path + f"_id_{iterator}.npy")


if __name__ == "__main__":
    main()
