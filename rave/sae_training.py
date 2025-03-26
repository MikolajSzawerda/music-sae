import argparse
from rave_experiments import prepareTrainingHiperparams, experiment


def getCMDArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("activations_path", type=str, help="Path to the file with saved tensors")
    parser.add_argument("output_path", type=str, help="Path to the file with saved sae")
    args = parser.parse_args()
    return args


def main():
    args = getCMDArgs()
    experiment(activations_path=args.activations_path, output_path=args.output_path,
               hiperparams=prepareTrainingHiperparams())


if __name__ == "__main__":
    main()
