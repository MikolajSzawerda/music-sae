import torch


def prepareLatentTensor(model, output: torch.Tensor, device: str = None):
    stereo = model.stereo
    if stereo:
        z0 = torch.cat([output, output])
    else:
        z0 = output
    mode = model.mode
    if mode == "variational":
        _0 = (torch.Tensor.size(z0))[0]
        full_latent_size = model.full_latent_size
        latent_size = model.latent_size
        _1 = torch.sub(full_latent_size, latent_size)
        noise = torch.randn([_0, _1, (torch.Tensor.size(z0))[-1]])
        if device:
            noise = noise.to(device)
        else:
            noise = noise.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        z2 = torch.cat([z0, noise], 1)
        latent_pca = model.latent_pca
        _2 = torch.unsqueeze(latent_pca, -1)
        z3 = torch.conv1d(z2, _2)
        latent_mean = model.latent_mean
        z4 = torch.add(z3, torch.unsqueeze(latent_mean, -1))
        z1 = z4
    else:
        z1 = z0
    return z1


def iterateThroughNet(net, output: torch.Tensor, layer_name: str):
    for name, module in net:
        output = module(output).detach()
        if name == layer_name:
            break
    return output.detach()


def getActivationEncoderLayer(model, batch, layer_name: str):
    encoder_net = list(list(list(model.encoder.named_children())[0][1].named_children())[0][1].named_children())
    output = model.pqmf(batch).detach()
    output = iterateThroughNet(encoder_net, output, layer_name)
    return output.detach()


def getActivationDecoderLayer(model, batch, layer_name: str):
    output = model.encode(batch).detach()
    output = prepareLatentTensor(model, output, device=next(model.parameters()).device)
    decoder_net = list(list(model.decoder.named_children())[0][1].named_children())
    output = iterateThroughNet(decoder_net, output, layer_name)
    return output.detach()


def getCallbacks():
    callbacks_dict = {}
    for layer_number in range(15):
        callbacks_dict["darbouka_encoder_" + str(layer_number)] = {}
        callbacks_dict["darbouka_encoder_" + str(layer_number)]["callback"] = getActivationEncoderLayer
        callbacks_dict["darbouka_encoder_" + str(layer_number)]["args"] = {}
        callbacks_dict["darbouka_encoder_" + str(layer_number)]["args"]["layer_name"] = str(layer_number)
    for layer_number in range(9):
        callbacks_dict["darbouka_decoder_" + str(layer_number)] = {}
        callbacks_dict["darbouka_decoder_" + str(layer_number)]["callback"] = getActivationDecoderLayer
        callbacks_dict["darbouka_decoder_" + str(layer_number)]["args"] = {}
        callbacks_dict["darbouka_decoder_" + str(layer_number)]["args"]["layer_name"] = str(layer_number)
    return callbacks_dict
