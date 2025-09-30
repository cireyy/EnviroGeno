import torch
import torch.nn as nn
import torch.nn.functional as F


class ChromosomeWiseEmbedding(nn.Module):
    """Embedding + attention pooling per chromosome."""
    def __init__(self, num_alleles=3, embedding_dim=64, attention_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_alleles, embedding_dim=embedding_dim)
        self.query_proj = nn.Linear(embedding_dim, attention_dim)
        self.key_proj   = nn.Linear(embedding_dim, attention_dim)
        self.value_proj = nn.Linear(embedding_dim, attention_dim)
        self.attention_dim = attention_dim

    def forward(self, x):
        # x: [B, 22, M] with SNP values (0/1/2)
        batch_size, num_chr, num_snps = x.shape
        outputs = []
        for chr_idx in range(num_chr):
            chr_snps  = x[:, chr_idx, :]          # [B, M]
            embedded  = self.embedding(chr_snps)  # [B, M, D]
            Q = self.query_proj(embedded)
            K = self.key_proj(embedded)
            V = self.value_proj(embedded)
            attn_scores  = torch.matmul(Q, K.transpose(-2, -1)) / (self.attention_dim ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_output  = torch.matmul(attn_weights, V)   # [B, M, d]
            pooled = attn_output.mean(dim=1)               # [B, d]
            outputs.append(pooled)
        return torch.stack(outputs, dim=1)                 # [B, 22, d]


class EnviroGenoModel(nn.Module):
    """
    EnviroGeno with:
      - chromosome-wise embedding + Transformer encoder
      - environment MLP
      - fusion classifier (prediction)
      - decoder (reconstruct environment)
      - conditional contrastive loss (staticmethod)
    """
    def __init__(self, num_env_features, snp_per_chr,
                 emb_dim=64, attn_dim=64, hidden_dim=128,
                 num_heads=4, dropout=0.2):
        super().__init__()

        # Genomic pathway
        self.genomic_embed = ChromosomeWiseEmbedding(
            num_alleles=3, embedding_dim=emb_dim, attention_dim=attn_dim
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=attn_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        # Environmental pathway
        self.env_mlp = nn.Sequential(
            nn.Linear(num_env_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, attn_dim)
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(attn_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

        # Decoder head
        self.decoder = nn.Sequential(
            nn.Linear(attn_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_env_features)
        )

    def forward(self, genotype, env):
        # Genomic branch
        geno_emb     = self.genomic_embed(genotype)          # [B, 22, d]
        geno_encoded = self.transformer_encoder(geno_emb)    # [B, 22, d]
        geno_pooled  = geno_encoded.mean(dim=1)              # [B, d]

        # Environmental branch
        env_emb = self.env_mlp(env)                          # [B, d]

        # Fusion
        fused   = torch.cat([geno_pooled, env_emb], dim=1)   # [B, 2d]
        logits  = self.classifier(fused)                     # [B, 2]
        recon_e = self.decoder(fused)                        # [B, F]

        return logits, fused, recon_e

    @staticmethod
    def conditional_contrastive_loss(features, conditions, temperature=0.5):
        z = F.normalize(features, dim=1)                 # [B, D]
        sim = torch.mm(z, z.t()) / temperature           # [B, B]

        # Positive mask: same condition, exclude self
        pos_mask = torch.eq(conditions.unsqueeze(0), conditions.unsqueeze(1)).float()
        pos_mask = pos_mask.fill_diagonal_(0.0)

        exp_sim   = torch.exp(sim)
        pos_sim   = exp_sim * pos_mask
        numerator = pos_sim.sum(dim=1)                   # sum over positives
        denominator = exp_sim.sum(dim=1) - torch.exp(torch.diag(sim))

        loss = -torch.log((numerator + 1e-8) / (denominator + 1e-8))
        return loss.mean()
