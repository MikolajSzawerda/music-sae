import argparse
from rave_experiments import prepareTrainingHiperparams, experiment
import sys


def getCMDArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("activations_path", type=str, help="Path to the folder with saved tensors")
    parser.add_argument("base_name", type=str, help="Base name for saving training statistics")
    parser.add_argument("output_path", type=str, help="Path to the file with saved sae")
    parser.add_argument('--pretrained_weights_path',
                        type=str, default=None, help='Path to file with pretrained weigths')
    args = parser.parse_args()
    return args


def main():
    args = getCMDArgs()
    experiment(activations_path=args.activations_path, base_name=args.base_name, output_path=args.output_path,
               hiperparams=prepareTrainingHiperparams(), pretrained_weights_path=args.pretrained_weights_path)


if __name__ == "__main__":
    sys.argv = ["sae_training.py",
                "./activations_test",
                "sae_darbouka_decoder_BN_7",
                "./weights/sae_darbouka_decoder_BN_7_0.pth"]
    main()
