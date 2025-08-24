"""
Example script for training Universal SAEs using the new cleaner implementation.
"""

import torch as t
from musicsae.usae import UniversalAutoEncoder, USAETopKTrainer
from scripts.train_usae import trainUSAE


def create_dummy_data(activation_dims: dict[str, int], batch_size: int = 32, num_batches: int = 100):
    """Create dummy activation data for multiple models."""
    for _ in range(num_batches):
        batch = {}
        for model_name, dim in activation_dims.items():
            batch[model_name] = t.randn(batch_size, dim)
        yield batch


def main():
    # Define the activation dimensions for different models
    activation_dims = {
        "model_a": 768,  # e.g., BERT-style model
        "model_b": 1024,  # e.g., larger transformer
        "model_c": 512,  # e.g., smaller model
    }

    # USAE hyperparameters
    dict_size = 8192  # Shared feature dictionary size
    k = 64  # Top-k sparsity
    steps = 1000

    # Create training configuration
    trainer_config = {
        "trainer": USAETopKTrainer,
        "steps": steps,
        "activation_dims": activation_dims,
        "dict_size": dict_size,
        "k": k,
        "layer": 12,  # Example layer
        "lm_name": "multi_model_usae",
        "lr": 1e-3,
        "auxk_alpha": 0.03125,
        "warmup_steps": 100,
        "threshold_start_step": 100,
        "device": "cuda" if t.cuda.is_available() else "cpu",
        "wandb_name": "UniversalSAE_Example",
    }

    # Create dummy data generator
    data = create_dummy_data(activation_dims, batch_size=32, num_batches=steps)

    # Train the Universal SAE
    print("Training Universal SAE...")
    trainUSAE(
        data=data,
        trainer_configs=[trainer_config],
        steps=steps,
        use_wandb=False,  # Set to True if you want to log to wandb
        save_dir="./usae_output",
        log_steps=50,
        verbose=True,
    )

    print("Training complete!")

    # Example of loading and using the trained USAE
    print("\nLoading trained USAE...")
    usae = UniversalAutoEncoder.from_pretrained("./usae_output/trainer_0/ae.pt")

    # Example usage: encode from one model, decode to another
    dummy_input = t.randn(1, activation_dims["model_a"])
    features = usae.encode(dummy_input, "model_a")

    # Decode to all models
    reconstructions = usae.decode_all(features)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Features sparsity (L0): {(features != 0).sum().item()}")

    for model_name, recon in reconstructions.items():
        print(f"Reconstruction for {model_name}: {recon.shape}")

    print("\nExample complete!")


if __name__ == "__main__":
    main()
