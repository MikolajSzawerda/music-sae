import argparse
from rave_experiments import experiment, sae_loss, getLearningParamsDictsList


def getCMDArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("activations_path", type=str, help="Path to the folder with saved tensors")
    parser.add_argument("params_path", type=str, help="Path to the file with hiperparams")
    parser.add_argument("params_id", type=int, help="ID of the chosen hiperparams")
    parser.add_argument("base_name", type=str, help="Base name for saving training statistics")
    parser.add_argument("output_path", type=str, help="Path to the file with saved sae")
    parser.add_argument('--pretrained_weights_path',
                        type=str, default=None, help='Path to file with pretrained weigths')
    args = parser.parse_args()
    return args


def main():
    args = getCMDArgs()
    params_list = getLearningParamsDictsList(args.params_id, args.params_path)
    id_, hiperparams = params_list[0]
    hiperparams["loss"] = sae_loss
    hiperparams["id"] = id_
    if hiperparams["max_activations"] == "Infinity":
        hiperparams["max_activations"] = float("inf")
    experiment(activations_path=args.activations_path, base_name=args.base_name, output_path=args.output_path,
               hiperparams=hiperparams, pretrained_weights_path=args.pretrained_weights_path)


if __name__ == "__main__":
    main()
