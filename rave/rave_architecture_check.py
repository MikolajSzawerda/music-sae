import torch


def getModulesStructure(module):
    children = []
    for name, child in module.named_children():
        children.append(
            {
                "children": getModulesStructure(child),
                "model": child,
                "module name": name,
            }
        )
    return children


def main():
    model_path = "./darbouka_onnx.ts"
    model_with_weights = torch.jit.load(model_path)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_with_weights.to(DEVICE)
    model_with_weights.eval()
    structure = getModulesStructure(model_with_weights.decoder)
    print("Główne moduły decoder", [dict["module name"] for dict in structure])
    print(
        "Glówne moduły decoder net",
        [dict["module name"] for dict in structure[0]["children"]],
    )
    print(
        "Główne moduły decoder synth",
        [dict["module name"] for dict in structure[1]["children"]],
    )
    structure = getModulesStructure(model_with_weights)
    print("Główne moduły", [dict["module name"] for dict in structure])
    print("Główne moduły pqmf", [dict["module name"] for dict in structure[0]["children"]])
    print(
        "Główne moduły encoder",
        [dict["module name"] for dict in structure[1]["children"][0]["children"][0]["children"]],
    )
    print(
        "Wymiary wejścia sieci",
        list(structure[0]["model"].named_parameters())[0][1].shape,
    )


if __name__ == "__main__":
    main()
