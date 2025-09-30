import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def create_condition_labels(env_df: pd.DataFrame):
    bmi_bins = pd.cut(
        env_df["BMI"].fillna(22),
        bins=[-1, 18.5, 25, 30, np.inf],
        labels=[0, 1, 2, 3]  # Underweight, Normal, Overweight, Obese
    ).astype(int)


    alcohol_bins = pd.cut(
        env_df["Alcohol"].fillna(0),
        bins=[-1, 0, 1, 3, np.inf],
        labels=[0, 1, 2, 3]  # None, Light, Moderate, Heavy
    ).astype(int)


    activity_bins = pd.cut(
        env_df["Physical_Activity"].fillna(0),
        bins=[-1, 0, 30, 60, np.inf],
        labels=[0, 1, 2, 3]  # Inactive, Low, Moderate, High
    ).astype(int)

    sleep_bins = pd.cut(
        env_df["Sleep"].fillna(7),
        bins=[-1, 6, 7, 9, np.inf],
        labels=[0, 1, 2, 3]  # Short, Slightly short, Normal, Long
    ).astype(int)
    smoking = env_df["Smoking"].fillna(0).astype(int)
    anxious = env_df["Anxious"].fillna(0).astype(int)
    combined = list(zip(bmi_bins, alcohol_bins, activity_bins, sleep_bins, smoking, anxious))
    unique_map = {v: i for i, v in enumerate(set(combined))}
    condition_labels = [unique_map[x] for x in combined]

    return np.array(condition_labels)


class GenomicDataset(Dataset):
    def __init__(self, data_dir, env_file):
        self.data_dir = data_dir

        # Load environmental data
        env_path = os.path.join(data_dir, env_file)
        self.env_data = pd.read_csv(env_path)

        # Labels
        self.labels = self.env_data["label"].values

        # Features
        self.env_features = self.env_data.drop(columns=["ID", "label"]).values
        self.conditions = create_condition_labels(self.env_data)

        # Load genotype data
        self.genotypes = []
        max_snps = 0
        for chr_id in range(1, 23):
            file_path = os.path.join(data_dir, f"chr{chr_id}.csv")
            df = pd.read_csv(file_path).drop(columns=["ID"])
            arr = df.values
            self.genotypes.append(arr)
            max_snps = max(max_snps, arr.shape[1])
        padded_genotypes = []
        for arr in self.genotypes:
            pad_len = max_snps - arr.shape[1]
            if pad_len > 0:
                arr = np.pad(arr, ((0, 0), (0, pad_len)),
                             mode="constant", constant_values=0)
            padded_genotypes.append(arr)

        # [N, 22, max_snps]
        self.genotypes = np.stack(padded_genotypes, axis=1)

        self.num_samples = len(self.labels)
        self.snp_per_chr = self.genotypes.shape[2]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        genotype = torch.tensor(self.genotypes[idx], dtype=torch.long)   # [22, snps_per_chr]
        env = torch.tensor(self.env_features[idx], dtype=torch.float32)  # [num_env_features]
        label = torch.tensor(self.labels[idx], dtype=torch.long)         # phenotype
        condition = torch.tensor(self.conditions[idx], dtype=torch.long) # contrastive condition
        return genotype, env, label, condition
