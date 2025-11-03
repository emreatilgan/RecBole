#!/usr/bin/env python3
import argparse
import os
import sys
import math
import csv
import random
from typing import List, Optional, Tuple, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader

torch.backends.cudnn.enabled = False  # CPU oriented

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], dropout: float = 0.1, last_activation: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for i, h in enumerate(hidden):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        if not last_activation and len(layers) > 0:
            # remove last ReLU for final layer stability if requested
            layers.pop(1)
        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.out_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CageCTR(nn.Module):
    """
    Context-Adaptive Gated Experts for CTR (prototype, CPU-friendly).
    """
    def __init__(
        self,
        id_cat_count: int,
        ctx_cat_count: int,
        id_num_count: int,
        ctx_num_count: int,
        vocab_size: int = 100000,
        embed_dim: int = 16,
        base_hidden: List[int] = [64, 64],
        ctx_hidden: List[int] = [64],
        expert_hidden: List[int] = [64],
        expert_num: int = 3,
        dropout: float = 0.1,
        temp_cap_min: float = 0.5,
        temp_cap_max: float = 2.0,
        lambda_inv: float = 1e-3,
        lambda_cal: float = 1e-4,
    ):
        super().__init__()
        self.id_cat_count = id_cat_count
        self.ctx_cat_count = ctx_cat_count
        self.id_num_count = id_num_count
        self.ctx_num_count = ctx_num_count
        self.vocab_size = int(vocab_size)
        self.embed_dim = embed_dim
        self.expert_num = expert_num
        self.temp_cap_min = temp_cap_min
        self.temp_cap_max = temp_cap_max
        self.lambda_inv = lambda_inv
        self.lambda_cal = lambda_cal
        # Embeddings
        self.id_cat_embeddings = nn.ModuleList(
            [nn.Embedding(self.vocab_size, embed_dim) for _ in range(id_cat_count)]
        )
        self.ctx_cat_embeddings = nn.ModuleList(
            [nn.Embedding(self.vocab_size, embed_dim) for _ in range(ctx_cat_count)]
        )
        # Towers
        base_in = id_cat_count * embed_dim + id_num_count
        ctx_in = ctx_cat_count * embed_dim + ctx_num_count
        self.base_tower = MLP(base_in, base_hidden, dropout=dropout, last_activation=True)
        self.ctx_tower = MLP(ctx_in, ctx_hidden, dropout=dropout, last_activation=True)
        joint_in = self.base_tower.out_dim + self.ctx_tower.out_dim
        # Gating and experts
        self.gate = nn.Linear(self.ctx_tower.out_dim, expert_num)
        self.experts = nn.ModuleList()
        for _ in range(expert_num):
            self.experts.append(
                nn.Sequential(
                    MLP(joint_in, expert_hidden, dropout=dropout, last_activation=True),
                    nn.Linear(expert_hidden[-1] if len(expert_hidden) > 0 else joint_in, 1),
                )
            )
        # Temperature head
        self.temp_head = nn.Linear(self.ctx_tower.out_dim, 1)
        # First-order optional could be added later
        # Loss
        self.bce = nn.BCEWithLogitsLoss()
        self.reset_parameters()

    def reset_parameters(self):
        for emb in list(self.id_cat_embeddings) + list(self.ctx_cat_embeddings):
            nn.init.xavier_normal_(emb.weight)
        # Linear and MLPs initialized by default Xavier via reset?
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def embed_cat(self, emb_list: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, F] longs
        return: [B, F*embed_dim] concatenated embeddings
        """
        if x is None or x.shape[1] == 0:
            return x.new_zeros((x.shape[0], 0)) if x is not None else torch.zeros(0, device=self.temp_head.weight.device)
        embs = []
        for i, emb in enumerate(emb_list):
            embs.append(emb(x[:, i]))
        return torch.cat(embs, dim=-1)

    def temperature(self, h_ctx: torch.Tensor) -> torch.Tensor:
        T = F.softplus(self.temp_head(h_ctx)) + 1e-6
        T = torch.clamp(T, min=self.temp_cap_min, max=self.temp_cap_max)
        return T

    def forward(self, id_cat: Optional[torch.Tensor], ctx_cat: Optional[torch.Tensor],
                id_num: Optional[torch.Tensor], ctx_num: Optional[torch.Tensor]) -> Tuple[torch.Tensor, dict]:
        """
        Returns logits and aux dict.
        """
        B = 0
        if id_cat is not None:
            B = id_cat.size(0)
        elif ctx_cat is not None:
            B = ctx_cat.size(0)
        elif id_num is not None:
            B = id_num.size(0)
        elif ctx_num is not None:
            B = ctx_num.size(0)
        # embeddings
        id_emb_flat = self.embed_cat(self.id_cat_embeddings, id_cat) if self.id_cat_count > 0 else torch.zeros((B,0))
        ctx_emb_flat = self.embed_cat(self.ctx_cat_embeddings, ctx_cat) if self.ctx_cat_count > 0 else torch.zeros((B,0))
        # concat with numerics
        if id_num is not None and id_num.numel() > 0:
            id_in = torch.cat([id_emb_flat, id_num], dim=-1)
        else:
            id_in = id_emb_flat
        if ctx_num is not None and ctx_num.numel() > 0:
            ctx_in = torch.cat([ctx_emb_flat, ctx_num], dim=-1)
        else:
            ctx_in = ctx_emb_flat
        # towers
        h_base = self.base_tower(id_in)
        h_ctx = self.ctx_tower(ctx_in)
        gate_logits = self.gate(h_ctx)
        gate = F.softmax(gate_logits, dim=-1)  # [B, K]
        # experts
        joint = torch.cat([h_base, h_ctx], dim=-1)
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(joint))  # [B,1]
        experts_stacked = torch.cat(expert_outputs, dim=-1)  # [B, K]
        z = torch.sum(gate * experts_stacked, dim=-1, keepdim=True)  # [B,1]
        T = self.temperature(h_ctx)  # [B,1]
        y_logit = z / T
        aux = {
            "gate": gate,
            "experts": experts_stacked,
            "T": T,
            "h_base": h_base,
            "h_ctx": h_ctx,
        }
        return y_logit.squeeze(-1), aux

    def loss_fn(self, y_logit: torch.Tensor, y: torch.Tensor, aux: dict) -> torch.Tensor:
        bce = self.bce(y_logit, y)
        # Gate entropy penalty (encourage high entropy -> uniform)
        p = aux["gate"] + 1e-12
        entropy_term = torch.sum(p * torch.log(p), dim=-1).mean()  # <= 0
        r_inv = self.lambda_inv * entropy_term
        # Temperature variance penalty
        T = aux["T"]
        var_T = torch.var(T, unbiased=False)
        r_cal = self.lambda_cal * var_T
        return bce + r_inv + r_cal

def hash_str(s: str, mod: int) -> int:
    # Stable hash
    return (hash(s) % mod + mod) % mod

class SyntheticCTR(Dataset):
    def __init__(self, n: int = 10000, id_cat_count: int = 2, ctx_cat_count: int = 4,
                 id_num_count: int = 2, ctx_num_count: int = 2, vocab_size: int = 10000):
        super().__init__()
        rng = np.random.default_rng(123)
        self.id_cat = rng.integers(0, vocab_size, size=(n, id_cat_count), dtype=np.int64)
        self.ctx_cat = rng.integers(0, vocab_size, size=(n, ctx_cat_count), dtype=np.int64)
        self.id_num = rng.standard_normal(size=(n, id_num_count)).astype(np.float32)
        self.ctx_num = rng.standard_normal(size=(n, ctx_num_count)).astype(np.float32)
        # Generate logits from a simple rule
        w_id = rng.standard_normal(size=(id_cat_count,)) if id_cat_count > 0 else np.array([])
        w_ctx = rng.standard_normal(size=(ctx_cat_count,)) if ctx_cat_count > 0 else np.array([])
        base = 0.1 * np.sum(self.id_num, axis=1) if id_num_count > 0 else 0.0
        ctx = 0.1 * np.sum(self.ctx_num, axis=1) if ctx_num_count > 0 else 0.0
        z = base + ctx + rng.normal(0, 0.5, size=(n,))
        p = 1 / (1 + np.exp(-z))
        self.y = (rng.random(n) < p).astype(np.float32)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.id_cat[idx], dtype=torch.long),
            torch.tensor(self.ctx_cat[idx], dtype=torch.long),
            torch.tensor(self.id_num[idx], dtype=torch.float32),
            torch.tensor(self.ctx_num[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )

class CriteoCSV(IterableDataset):
    """
    Streams Criteo CSV (label + I1..I13 + C1..C26).
    """
    def __init__(self, path: str, vocab_size: int = 100000, identity_cats: int = 2):
        super().__init__()
        self.path = path
        self.vocab_size = vocab_size
        self.identity_cats = identity_cats  # take first N cat fields as identity

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        with open(self.path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 40:
                    # Some versions use comma; try again
                    row = row[0].split(",")
                if len(row) < 40:
                    continue
                y = float(row[0])
                I = row[1:14]
                C = row[14:40]
                # numerics
                id_num = []
                ctx_num = []
                for i, v in enumerate(I):
                    try:
                        val = float(v)
                    except:
                        val = 0.0
                    # Assign first 2 numeric to identity
                    (id_num if i < 2 else ctx_num).append(math.log1p(abs(val)))
                # categorical
                cat_ids = [hash_str(c, self.vocab_size) for c in C]
                id_cat = cat_ids[: self.identity_cats]
                ctx_cat = cat_ids[self.identity_cats :]
                yield (
                    torch.tensor(id_cat, dtype=torch.long).unsqueeze(0),
                    torch.tensor(ctx_cat, dtype=torch.long).unsqueeze(0),
                    torch.tensor(id_num, dtype=torch.float32).unsqueeze(0),
                    torch.tensor(ctx_num, dtype=torch.float32).unsqueeze(0),
                    torch.tensor([y], dtype=torch.float32),
                )

class ML1MCSV(Dataset):
    """
    Minimal ML-1M loader. Accepts either ratings.dat (user::item::rating::timestamp) or CSV with columns:
    user_id,item_id,rating,timestamp
    Context features derived: hour (0..23), weekday (0..6).
    Labels: 1 if rating>=4; 0 if rating<=2; drop rating==3.
    """
    def __init__(self, path: str, vocab_size: int = 100000):
        super().__init__()
        self.vocab_size = vocab_size
        rows = []
        if path.endswith(".dat"):
            with open(path, "r", encoding="latin-1") as f:
                for line in f:
                    parts = line.strip().split("::")
                    if len(parts) < 4:
                        continue
                    u, i, r, ts = parts[:4]
                    r = int(float(r))
                    if r == 3:
                        continue
                    y = 1.0 if r >= 4 else 0.0
                    rows.append((u, i, y, int(ts)))
        else:
            import pandas as pd  # optional dependency
            df = pd.read_csv(path)
            assert {"user_id", "item_id", "rating", "timestamp"}.issubset(df.columns)
            df = df[df["rating"] != 3]
            df["label"] = (df["rating"] >= 4).astype(np.float32)
            rows = list(zip(df["user_id"].astype(str), df["item_id"].astype(str), df["label"].tolist(), df["timestamp"].astype(int).tolist()))
        self.n = len(rows)
        self.id_cat = np.zeros((self.n, 2), dtype=np.int64)  # user_id, item_id
        self.ctx_cat = np.zeros((self.n, 2), dtype=np.int64)  # hour, weekday
        self.id_num = np.zeros((self.n, 0), dtype=np.float32)
        self.ctx_num = np.zeros((self.n, 0), dtype=np.float32)
        self.y = np.zeros((self.n,), dtype=np.float32)
        for idx, (u, i, y, ts) in enumerate(rows):
            self.id_cat[idx, 0] = hash_str(u, self.vocab_size)
            self.id_cat[idx, 1] = hash_str(i, self.vocab_size)
            hour = int((int(ts) // 3600) % 24)
            weekday = int((int(ts) // 86400 + 4) % 7)  # epoch starts Thursday
            self.ctx_cat[idx, 0] = hour
            self.ctx_cat[idx, 1] = weekday
            self.y[idx] = y

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (
            torch.tensor(self.id_cat[idx], dtype=torch.long),
            torch.tensor(self.ctx_cat[idx], dtype=torch.long),
            torch.tensor(self.id_num[idx], dtype=torch.float32),
            torch.tensor(self.ctx_num[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )

def collate_pad(batch):
    # All fields have fixed sizes; stacking suffices
    id_cat = torch.stack([b[0] for b in batch], dim=0)
    ctx_cat = torch.stack([b[1] for b in batch], dim=0)
    id_num = torch.stack([b[2] for b in batch], dim=0) if batch[0][2].numel() > 0 else torch.zeros((len(batch),0))
    ctx_num = torch.stack([b[3] for b in batch], dim=0) if batch[0][3].numel() > 0 else torch.zeros((len(batch),0))
    y = torch.stack([b[4] for b in batch], dim=0)
    return id_cat, ctx_cat, id_num, ctx_num, y

@torch.no_grad()
def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    bins = torch.linspace(0, 1, steps=n_bins + 1, device=probs.device)
    ece = torch.tensor(0.0, device=probs.device)
    for i in range(n_bins):
        m = (probs >= bins[i]) & (probs < bins[i + 1])
        if m.any():
            avg_conf = probs[m].mean()
            avg_acc = labels[m].float().mean()
            ece += (m.float().mean()) * torch.abs(avg_conf - avg_acc)
    return float(ece.cpu().item())

def train_epoch(model: CageCTR, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device,
                log_every: int = 200) -> Tuple[float, float]:
    model.train()
    # track average loss and ece on-the-fly window
    total_loss = 0.0
    total_count = 0
    last_log = 0
    running_probs = []
    running_labels = []
    for step, (id_cat, ctx_cat, id_num, ctx_num, y) in enumerate(loader):
        id_cat = id_cat.to(device)
        ctx_cat = ctx_cat.to(device)
        id_num = id_num.to(device) if id_num.numel() > 0 else None
        ctx_num = ctx_num.to(device) if ctx_num.numel() > 0 else None
        y = y.to(device).view(-1)
        logits, aux = model(id_cat, ctx_cat, id_num, ctx_num)
        loss = model.loss_fn(logits, y, aux)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * y.size(0)
        total_count += y.size(0)
        probs = torch.sigmoid(logits.detach())
        running_probs.append(probs.cpu())
        running_labels.append(y.cpu())
        if (step + 1) % log_every == 0:
            probs_cat = torch.cat(running_probs)
            labels_cat = torch.cat(running_labels)
            ece = compute_ece(probs_cat, labels_cat)
            avg_loss = total_loss / max(1, total_count)
            print(f"[train] step={step+1} avg_loss={avg_loss:.4f} ece={ece:.4f}")
            running_probs.clear()
            running_labels.clear()
            last_log = step + 1
    # final window
    if running_probs:
        probs_cat = torch.cat(running_probs)
        labels_cat = torch.cat(running_labels)
        ece = compute_ece(probs_cat, labels_cat)
    else:
        ece = 0.0
    avg_loss = total_loss / max(1, total_count)
    return avg_loss, ece

@torch.no_grad()
def evaluate(model: CageCTR, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_probs = []
    all_labels = []
    for id_cat, ctx_cat, id_num, ctx_num, y in loader:
        id_cat = id_cat.to(device)
        ctx_cat = ctx_cat.to(device)
        id_num = id_num.to(device) if id_num.numel() > 0 else None
        ctx_num = ctx_num.to(device) if ctx_num.numel() > 0 else None
        y = y.to(device).view(-1)
        logits, aux = model(id_cat, ctx_cat, id_num, ctx_num)
        loss = model.loss_fn(logits, y, aux)
        total_loss += float(loss.item()) * y.size(0)
        total_count += y.size(0)
        all_probs.append(torch.sigmoid(logits).cpu())
        all_labels.append(y.cpu())
    if total_count == 0:
        return 0.0, 0.0
    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    ece = compute_ece(probs, labels)
    avg_loss = total_loss / max(1, total_count)
    return avg_loss, ece

def build_loaders(args, device: torch.device):
    if args.mode == "synthetic":
        train_ds = SyntheticCTR(
            n=args.synthetic_n,
            id_cat_count=args.id_cat_count,
            ctx_cat_count=args.ctx_cat_count,
            id_num_count=args.id_num_count,
            ctx_num_count=args.ctx_num_count,
            vocab_size=args.vocab_size,
        )
        val_ds = SyntheticCTR(
            n=max(1000, args.batch_size * 2),
            id_cat_count=args.id_cat_count,
            ctx_cat_count=args.ctx_cat_count,
            id_num_count=args.id_num_count,
            ctx_num_count=args.ctx_num_count,
            vocab_size=args.vocab_size,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_pad)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_pad)
        id_cat_count = args.id_cat_count
        ctx_cat_count = args.ctx_cat_count
        id_num_count = args.id_num_count
        ctx_num_count = args.ctx_num_count
    elif args.mode == "criteo":
        assert args.criteo_path and os.path.exists(args.criteo_path), "Provide --criteo_path to Criteo CSV"
        dataset = CriteoCSV(args.criteo_path, vocab_size=args.vocab_size, identity_cats=args.identity_cats)
        # For streaming dataset, we create limited epoch iteration using islice
        train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
        val_loader = None
        id_cat_count = args.identity_cats
        ctx_cat_count = 26 - args.identity_cats
        id_num_count = 2
        ctx_num_count = 11
    elif args.mode == "ml1m":
        assert args.ml1m_path and os.path.exists(args.ml1m_path), "Provide --ml1m_path to ML-1M ratings"
        dataset = ML1MCSV(args.ml1m_path, vocab_size=args.vocab_size)
        n = len(dataset)
        val_n = max(1000, int(0.1 * n))
        train_n = n - val_n
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_n, val_n], generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_pad)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_pad)
        id_cat_count = 2
        ctx_cat_count = 2
        id_num_count = 0
        ctx_num_count = 0
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    return train_loader, val_loader, id_cat_count, ctx_cat_count, id_num_count, ctx_num_count

def main():
    parser = argparse.ArgumentParser(description="CAGE-CTR Prototype (CPU)")
    parser.add_argument("--mode", type=str, default="synthetic", choices=["synthetic", "criteo", "ml1m"])
    parser.add_argument("--criteo_path", type=str, default="")
    parser.add_argument("--ml1m_path", type=str, default="")
    parser.add_argument("--identity_cats", type=int, default=2, help="For Criteo: number of first categorical fields considered identity")
    parser.add_argument("--synthetic_n", type=int, default=20000)
    parser.add_argument("--id_cat_count", type=int, default=2)
    parser.add_argument("--ctx_cat_count", type=int, default=4)
    parser.add_argument("--id_num_count", type=int, default=2)
    parser.add_argument("--ctx_num_count", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=100000)
    parser.add_argument("--embed_dim", type=int, default=16)
    parser.add_argument("--base_hidden", type=str, default="64,64")
    parser.add_argument("--ctx_hidden", type=str, default="64")
    parser.add_argument("--expert_hidden", type=str, default="64")
    parser.add_argument("--expert_num", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temp_cap_min", type=float, default=0.5)
    parser.add_argument("--temp_cap_max", type=float, default=2.0)
    parser.add_argument("--lambda_inv", type=float, default=1e-3)
    parser.add_argument("--lambda_cal", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cpu")

    train_loader, val_loader, id_cat_count, ctx_cat_count, id_num_count, ctx_num_count = build_loaders(args, device)
    base_hidden = [int(x) for x in args.base_hidden.split(",") if x.strip() != ""]
    ctx_hidden = [int(x) for x in args.ctx_hidden.split(",") if x.strip() != ""]
    expert_hidden = [int(x) for x in args.expert_hidden.split(",") if x.strip() != ""]

    model = CageCTR(
        id_cat_count=id_cat_count,
        ctx_cat_count=ctx_cat_count,
        id_num_count=id_num_count,
        ctx_num_count=ctx_num_count,
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        base_hidden=base_hidden,
        ctx_hidden=ctx_hidden,
        expert_hidden=expert_hidden,
        expert_num=args.expert_num,
        dropout=args.dropout,
        temp_cap_min=args.temp_cap_min,
        temp_cap_max=args.temp_cap_max,
        lambda_inv=args.lambda_inv,
        lambda_cal=args.lambda_cal,
    ).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ece = train_epoch(model, train_loader, optimizer, device, log_every=args.log_every)
        if val_loader is not None:
            val_loss, val_ece = evaluate(model, val_loader, device)
            print(f"Epoch {epoch}: train_loss={tr_loss:.4f} train_ece={tr_ece:.4f} | val_loss={val_loss:.4f} val_ece={val_ece:.4f}")
        else:
            print(f"Epoch {epoch}: train_loss={tr_loss:.4f} train_ece={tr_ece:.4f}")

if __name__ == "__main__":
    main()