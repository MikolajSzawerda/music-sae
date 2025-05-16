import torch as t
import pytest
from typing import Dict

from musicsae.usae import UniversalAutoEncoder, USAETopKTrainer


@pytest.fixture
def model_dims() -> Dict[str, int]:
    """Fixture providing model dimensions for testing."""
    return {"model1": 512, "model2": 768, "model3": 1024}


@pytest.fixture
def input_data(model_dims) -> Dict[str, t.Tensor]:
    """Fixture providing model dimensions for testing."""
    return {name: t.randn(32, dim) for name, dim in model_dims.items()}


@pytest.fixture
def universal_ae(model_dims):
    """Fixture providing a UniversalAutoEncoder instance."""
    return UniversalAutoEncoder(model_activation_dims=model_dims, dict_size=2048, k=40)


@pytest.fixture
def universal_trainer(model_dims):
    """Fixture providing a UniversalSAETrainer instance."""
    return USAETopKTrainer(
        steps=10000, activation_dims=model_dims, dict_size=2048, k=40, layer=0, lm_name="test", device="cpu"
    )


def test_universal_ae_initialization(universal_ae, model_dims):
    """Test proper initialization of UniversalAutoEncoder."""
    # Check model dimensions
    assert universal_ae.model_activation_dims == model_dims
    assert universal_ae.dict_size == 2048

    # Check that all encoders and decoders exist
    for model_name in model_dims:
        assert model_name in universal_ae.encoders
        assert model_name in universal_ae.decoders
        assert model_name in universal_ae.b_dec

    # Check encoder-decoder shapes
    for model_name, dim in model_dims.items():
        assert universal_ae.encoders[model_name].weight.shape == (2048, dim)
        assert universal_ae.decoders[model_name].weight.shape == (dim, 2048)
        assert universal_ae.b_dec[model_name].shape == (dim,)


def test_universal_ae_forward(universal_ae, model_dims):
    """Test forward pass through UniversalAutoEncoder."""
    batch_size = 32

    for model_name, dim in model_dims.items():
        # Create random input
        x = t.randn(batch_size, dim)

        # Test forward pass without features
        x_hat = universal_ae(x, model_name)
        assert x_hat.shape == x.shape

        # Test forward pass with features
        x_hat, features = universal_ae(x, model_name, output_features=True)
        assert x_hat.shape == x.shape
        assert features.shape == (batch_size, 2048)

        # Check sparsity constraint
        assert (features != 0).sum(dim=-1).max() <= 40  # k=40


def test_universal_ae_encode_decode(universal_ae, model_dims):
    """Test encode and decode methods separately."""
    batch_size = 32

    for model_name, dim in model_dims.items():
        x = t.randn(batch_size, dim)

        # Test encode
        encoded = universal_ae.encode(x, model_name)
        assert encoded.shape == (batch_size, 2048)
        assert (encoded != 0).sum(dim=-1).max() <= 40

        # Test encode with topk
        encoded, topk_vals, topk_indices, post_relu = universal_ae.encode(x, model_name, return_topk=True)
        assert topk_vals.shape == (batch_size, 40)
        assert topk_indices.shape == (batch_size, 40)
        assert post_relu.shape == (batch_size, 2048)

        # Test decode
        decoded = universal_ae.decode(encoded, model_name)
        assert decoded.shape == x.shape


def test_universal_ae_encode_decode_all(universal_ae, model_dims):
    """Test encode and decode all methods separately."""
    batch_size = 32

    for model_name, dim in model_dims.items():
        x = t.randn(batch_size, dim)

        # Test encode
        encoded, topk_vals, topk_indices, post_relu = universal_ae.encode(x, model_name, return_topk=True)

        # Test decode
        decoded = universal_ae.decode_all(encoded)
        for decode_name, decoded_value in decoded.items():
            assert decoded_value.shape == (batch_size, model_dims[decode_name])


def test_universal_ae_threshold(universal_ae, model_dims):
    """Test threshold-based sparsity."""
    batch_size = 32

    for model_name, dim in model_dims.items():
        x = t.randn(batch_size, dim)

        # Test with threshold
        encoded = universal_ae.encode(x, model_name, use_threshold=True)
        assert encoded.shape == (batch_size, 2048)

        # Set threshold and test again
        universal_ae.thresholds[model_name].data = t.tensor(0.5)
        encoded = universal_ae.encode(x, model_name, use_threshold=True)
        assert t.min(encoded[encoded != 0]) > t.tensor(0.5)  # No values above threshold


def test_universal_ae_scale_biases(universal_ae, model_dims):
    """Test bias scaling functionality."""
    scale = 2.0

    # Store original biases
    original_biases = {
        model_name: {
            "encoder": universal_ae.encoders[model_name].bias.clone(),
            "decoder": universal_ae.b_dec[model_name].clone(),
        }
        for model_name in model_dims
    }

    # Scale biases
    universal_ae.scale_biases(scale)

    # Check that all biases were scaled
    for model_name in model_dims:
        t.testing.assert_close(universal_ae.encoders[model_name].bias, original_biases[model_name]["encoder"] * scale)
        t.testing.assert_close(universal_ae.b_dec[model_name], original_biases[model_name]["decoder"] * scale)


def test_universal_trainer_initialization(universal_trainer, model_dims):
    """Test proper initialization of UniversalSAETrainer."""
    assert universal_trainer.ae.model_activation_dims == model_dims
    assert universal_trainer.ae.dict_size == 2048
    assert universal_trainer.ae.k.item() == 40
    assert universal_trainer.steps == 10000
    assert universal_trainer.layer == 0
    assert universal_trainer.lm_name == "test"


def test_universal_trainer_update(universal_trainer, model_dims, input_data):
    """Test training update step."""
    for model_name, dim in model_dims.items():
        loss = universal_trainer.update(0, input_data, model_name)
        assert isinstance(loss, float)
        assert loss >= 0


def test_universal_trainer_loss(universal_trainer, model_dims, input_data):
    """Test loss computation."""
    for model_name, dim in model_dims.items():
        loss = universal_trainer.loss(input_data, model_name, step=0)
        assert isinstance(loss, t.Tensor)
        assert loss.shape == ()
        assert loss >= 0

        # Test loss with logging
        loss_log = universal_trainer.loss(input_data, model_name, step=0, logging=True)
        assert hasattr(loss_log, "x")
        assert hasattr(loss_log, "x_hat")
        assert hasattr(loss_log, "f")
        assert hasattr(loss_log, "losses")
        assert "l2_loss" in loss_log.losses
        assert "auxk_loss" in loss_log.losses
        assert "loss" in loss_log.losses


#
#
def test_universal_trainer_threshold_update(universal_trainer, model_dims):
    """Test threshold update mechanism."""
    batch_size = 32

    for model_name, dim in model_dims.items():
        x = t.randn(batch_size, dim)

        # Get activations
        f, top_acts, top_indices, post_relu = universal_trainer.ae.encode(x, model_name, return_topk=True)

        # Test threshold update
        universal_trainer.update_threshold(top_acts, model_name)
        assert universal_trainer.ae.thresholds[model_name] >= 0


#
#
# def test_universal_trainer_save_load(universal_trainer, model_dims, tmp_path):
#     """Test saving and loading the model."""
#     # Save model
#     save_path = tmp_path / "universal_ae.pt"
#     t.save(universal_trainer.ae.state_dict(), save_path)
#
#     # Load model
#     loaded_ae = UniversalAutoEncoder.from_pretrained(
#         save_path,
#         k=40,
#         device="cpu"
#     )
#
#     # Check that loaded model matches original
#     assert loaded_ae.model_activation_dims == model_dims
#     assert loaded_ae.dict_size == 2048
#     assert loaded_ae.k.item() == 40
#
#     # Check that weights match
#     for model_name in model_dims:
#         t.testing.assert_close(
#             loaded_ae.encoders[model_name].weight,
#             universal_trainer.ae.encoders[model_name].weight
#         )
#         t.testing.assert_close(
#             loaded_ae.decoders[model_name].weight,
#             universal_trainer.ae.decoders[model_name].weight
#         )
