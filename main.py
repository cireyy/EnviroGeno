import torch
import yaml
import argparse
from src.dataset import GenomicDataset
from src.model import EnviroGenoModel
from src.train import cross_validate
from src.utils import set_seed


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(42)

    # Dataset
    dataset = GenomicDataset(config["data_dir"], config["env_file"])
    num_env_features = dataset.env_features.shape[1]
    snp_per_chr = dataset.snp_per_chr

    # Cross-validation
    results = cross_validate(
        dataset=dataset,
        model_class=EnviroGenoModel,
        num_env_features=num_env_features,
        snp_per_chr=snp_per_chr,
        device=device,
        n_splits=config["n_splits"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        lr=config["lr"],
        emb_dim=config["emb_dim"],
        attn_dim=config["attn_dim"],
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        lambda_cl=config["lambda_cl"],
    )

    print("\nCross-validation results:", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
