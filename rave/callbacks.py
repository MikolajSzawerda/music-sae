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


def iterateThroughNetWithIntervention(net, output: torch.Tensor, layer_name: str):
    for name, module in net:
        output = module(output).detach()
        if name == layer_name:
            output = output
    return output.detach()


def mod_sigmoid(x):
    return 2 * torch.sigmoid(x)**2.3 + 1e-7


def forwardThroughSynth(decoder, x: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
    x0 = x
    if decoder.use_noise:
        waveform, loudness, noise = decoder.synth(x0)
    else:
        waveform, loudness = decoder.synth(x0)
        noise = torch.zeros_like(waveform)
    if decoder.loud_stride != 1:
        loudness = torch.repeat_interleave(loudness, decoder.loud_stride)
    loudness = torch.reshape(loudness, (x0.size(0), 1, -1))
    modulated_waveform = torch.tanh(waveform) * mod_sigmoid(loudness)
    if add_noise:
        output = modulated_waveform + noise
    else:
        output = modulated_waveform
    return output


def postEncode(model, x: torch.Tensor) -> torch.Tensor:
    if model.mode == "variational":
        z_reparam = model.encoder.reparametrize(x)[0]

        z_centered = z_reparam - model.latent_mean.unsqueeze(-1)

        z_projected = torch.conv1d(z_centered, model.latent_pca.unsqueeze(-1))

        z_final = z_projected[:, 1:1+model.latent_size, :]
    else:
        z_final = x
    return z_final


def preDecode(model, z: torch.Tensor) -> torch.Tensor:
    # Jeśli model jest stereo, duplikujemy latenty
    if model.stereo:
        z0 = torch.cat([z, z], dim=0)  # 2x batch
    else:
        z0 = z

    # Jeśli używamy trybu wariacyjnego (VAE)
    if model.mode == "variational":
        batch_size = z0.size(0)
        time_steps = z0.size(-1)

        # Uzupełnij latenty o szum
        noise_dims = model.full_latent_size - model.latent_size
        noise = torch.randn(batch_size, noise_dims, time_steps)

        # Sklej latenty ze szumem
        z2 = torch.cat([z0, noise], dim=1)

        # PCA: zastosuj odwrotną projekcję przez conv1d
        pca_weights = model.latent_pca.numpy().T  # transpozycja PCA
        pca_weights_tensor = torch.from_numpy(pca_weights).unsqueeze(-1)
        z3 = torch.conv1d(z2, pca_weights_tensor)

        # Dodaj średnią latentów
        z4 = z3 + model.latent_mean.unsqueeze(-1)
        z1 = z4
    else:
        z1 = z0
    return z1


def forwardWithIntervention(model, batch, layer_name: str):
    encoder_net = list(list(list(model.encoder.named_children())[0][1].named_children())[0][1].named_children())
    output = model.pqmf(batch).detach()
    output = iterateThroughNetWithIntervention(encoder_net, output, layer_name)
    output = postEncode(model, output)
    output = preDecode(model, output)
    decoder_net = list(list(model.decoder.named_children())[0][1].named_children())
    output = iterateThroughNetWithIntervention(decoder_net, output, layer_name)
    output = forwardThroughSynth(model.decoder, output, add_noise=False)
    y0 = model.pqmf.inverse(output)
    if model.stereo:
        y1 = torch.cat(torch.chunk(y0, 2, dim=0), dim=1)
    else:
        y1 = y0
    return y1


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
