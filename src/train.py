import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from src.utils import evaluate, print_metrics


def cross_validate(dataset, model_class, num_env_features, snp_per_chr, device,
                   n_splits=5, epochs=5, batch_size=16, lr=1e-3,
                   emb_dim=64, attn_dim=64, hidden_dim=128,
                   num_heads=4, dropout=0.2, lambda_ce=0.5,
                   lambda_cl=0.3, lambda_rec=0.2, temperature=0.5):

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n===== Fold {fold+1}/{n_splits} =====")
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

        model = model_class(num_env_features=num_env_features,
                            snp_per_chr=snp_per_chr,
                            emb_dim=emb_dim, attn_dim=attn_dim,
                            hidden_dim=hidden_dim, num_heads=num_heads,
                            dropout=dropout).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        ce_loss   = nn.CrossEntropyLoss()
        mse_loss  = nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            total_loss, total_ce, total_cl, total_rec = 0.0, 0.0, 0.0, 0.0

            for genotype, env, labels, cond in train_loader:
                genotype = genotype.to(device)
                env      = env.to(device)
                labels   = labels.to(device)
                cond     = cond.to(device)

                logits, fused, recon_env = model(genotype, env)

                # 1) Prediction loss
                loss_pred = ce_loss(logits, labels)

                # 2) Conditional contrastive loss
                loss_cl = model.conditional_contrastive_loss(fused, cond, temperature=temperature)

                # 3) Reconstruction loss
                loss_rec = mse_loss(recon_env, env)

                # Total
                loss = lambda_ce * loss_pred + lambda_cl * loss_cl + lambda_rec * loss_rec

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_ce += loss_pred.item()
                total_cl += loss_cl.item()
                total_rec += loss_rec.item()

            n_batches = max(1, len(train_loader))
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {total_loss/n_batches:.4f} | "
                  f"CE: {total_ce/n_batches:.4f} | "
                  f"CL: {total_cl/n_batches:.4f} | "
                  f"REC: {total_rec/n_batches:.4f}")

            acc, f1, auroc = evaluate(model, val_loader, device)
            print_metrics(acc, f1, auroc, prefix=f"Fold {fold+1} Validation")
            fold_results.append((acc, f1, auroc))

    return fold_results
