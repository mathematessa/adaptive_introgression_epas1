import gc
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

DATA_DIR  = Path("dataset_ready")

N_FEAT = 5
SEQ_LEN = 32
EMBED_DIM = 64
HIDDEN = 128
N_LAYERS = 2
DROPOUT = 0.2
N_CLS_PER_BATCH = 10
N_PER_CLS = 4
BASE_MARGIN = 0.5
EPOCHS = 20
LR = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
NEUTRALITY_TIMES = torch.tensor([200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],dtype=torch.float32)
CLASS_NAMES = [f"t={t}" for t in NEUTRALITY_TIMES.int().tolist()]

def build_margin_matrix(times: torch.Tensor, base: float) -> torch.Tensor:
    diff = (times.unsqueeze(0) - times.unsqueeze(1)).abs()
    return base * (diff / diff.max()).clamp(min=0.1)

class SimulationDataset(Dataset):
    def __init__(self, split: str, max_snps: int | None = None):
        split_dir = DATA_DIR / split
        labels_csv = DATA_DIR / f"labels_{split}.csv"
        df = pd.read_csv(labels_csv)
        arrays, labels, lengths = [], [], []
        for _, row in df.iterrows():
            arr = np.load(split_dir / row["file"])
            arrays.append(arr)
            labels.append(int(row["target_class"]))
            lengths.append(len(arr))

        self.max_snps = max_snps or max(lengths)
        n = len(arrays)
        X_pad = np.zeros((n, self.max_snps, N_FEAT), dtype=np.float32)
        mask = np.zeros((n, self.max_snps), dtype=bool)
        for i, (arr, L) in enumerate(zip(arrays, lengths)):
            L_clip = min(L, self.max_snps)
            X_pad[i,:L_clip] = arr[:L_clip]
            mask[i,:L_clip] = True
        self.X = torch.tensor(X_pad)
        self.mask = torch.tensor(mask)
        self.y = torch.tensor(labels, dtype=torch.long)
        print(f"  {split}: {n} simulations  "
              f"SNPs/sim: min={min(lengths)} "
              f"mean={int(np.mean(lengths))} "
              f"max={max(lengths)}  "
              f"max_snps={self.max_snps}")
        print(f"  Class counts: {np.bincount(np.array(labels)).tolist()}")

    def __len__(self): 
        return len(self.y)

    def __getitem__(self, i): 
        return self.X[i], self.mask[i], self.y[i]

def _pairwise_l2(emb: torch.Tensor) -> torch.Tensor:
    dot = emb @ emb.T
    sq = dot.diagonal()
    d = (sq.unsqueeze(1) - 2.0 * dot + sq.unsqueeze(0)).clamp(min=0.0)
    mask = (d == 0.0).float()
    return torch.sqrt(d + mask * 1e-16) * (1.0 - mask)


def _masked_minimum(data, mask, dim=1):
    ax = data.max(dim=dim, keepdim=True).values
    return ((data - ax) * mask).min(dim=dim, keepdim=True).values + ax


def _masked_maximum(data, mask, dim=1):
    ax = data.min(dim=dim, keepdim=True).values
    return ((data - ax) * mask).max(dim=dim, keepdim=True).values + ax


def triplet_semihard_loss(
    labels: torch.Tensor,
    embeddings: torch.Tensor,
    margin: float,
    margin_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    device = embeddings.device
    B = labels.size(0)
    emb = embeddings.float()

    pdist = _pairwise_l2(emb)
    lc  = labels.unsqueeze(1)
    adjacency = lc == lc.T
    adjacency_not = ~adjacency
    pdist_tile = pdist.repeat(B, 1)
    sh_mask = (adjacency_not.repeat(B, 1) & (pdist_tile > pdist.T.reshape(-1, 1)))
    has_sh = (sh_mask.float().sum(1, keepdim=True) > 0).reshape(B, B).T
    neg_outside = _masked_minimum(pdist_tile, sh_mask.float()).reshape(B, B).T
    neg_inside = _masked_maximum(pdist, adjacency_not.float()).expand(-1, B)
    neg = torch.where(has_sh, neg_outside, neg_inside)
    if margin_matrix is not None:
        pair_m = margin_matrix.to(device)[lc, labels.unsqueeze(0)]
    else:
        pair_m = torch.full((B, B), margin, device=device)
    loss_mat = pair_m + pdist - neg
    pos_mask = (adjacency.float() - torch.eye(B, device=device)).clamp(min=0)
    n_pos = pos_mask.sum()
    return (loss_mat * pos_mask).clamp(min=0).sum() / (n_pos + 1e-9)

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, n_classes: int, n_per_class: int):
        super().__init__()
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.n_classes = n_classes
        self.n_per_class = n_per_class
        self.by_class = defaultdict(list)
        for i, y in enumerate(labels):
            self.by_class[int(y)].append(i)
        self.classes = sorted(self.by_class)
        self.n_batches = len(labels) // (n_classes * n_per_class)

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for c in np.random.choice(self.classes, self.n_classes, replace=False):
                pool = self.by_class[c]
                batch += np.random.choice(pool, self.n_per_class, replace=len(pool) < self.n_per_class).tolist()
            yield batch

    def __len__(self):
        return self.n_batches


class SimulationEmbedder(nn.Module):
    def __init__(
        self,
        input_dim: int = N_FEAT,
        hidden_dim: int = HIDDEN,
        n_layers: int = N_LAYERS,
        embed_dim: int = EMBED_DIM,
        seq_len: int = SEQ_LEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.seq_len = seq_len
        gru_out = hidden_dim * 2
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(gru_out, 1)
        self.proj = nn.Sequential(
            nn.Linear(gru_out, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, S, n_feat = x.shape
        T = self.seq_len
        pad = (T - S % T) % T
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
            mask = F.pad(mask, (0, pad))
        K = x.shape[1] // T 
        x_win = x.reshape(B, K, T, n_feat)
        m_win = mask.reshape(B, K, T)
        out, _  = self.gru(x_win.reshape(B * K, T, n_feat))
        w = torch.softmax(self.attn(out), dim=1)
        ctx = (w * out).sum(dim=1).reshape(B, K, -1)
        valid = m_win.any(dim=-1).float().unsqueeze(-1) 
        n_valid = valid.sum(dim=1).clamp(min=1)
        sim_repr = (ctx * valid).sum(dim=1) / n_valid 
        return F.normalize(self.proj(sim_repr), p=2, dim=1)

def get_scheduler(optimizer, n_epochs: int, warmup: int = 5):
    def lr_lambda(e):
        if e < warmup:
            return (e + 1) / warmup
        p = (e - warmup) / max(n_epochs - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def run_epoch(model, loader, mm, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total = n = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X_b, mask_b, y_b in loader:
            X_b, mask_b, y_b = X_b.to(device), mask_b.to(device), y_b.to(device)
            emb = model(X_b, mask_b)
            loss = triplet_semihard_loss(y_b, emb, BASE_MARGIN, mm)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item(); n += 1
    return total / max(n, 1)


@torch.no_grad()
def get_embeddings(model, loader, device):
    model.eval()
    embs, lbls = [], []
    for X_b, mask_b, y_b in loader:
        embs.append(model(X_b.to(device), mask_b.to(device)).cpu().numpy())
        lbls.append(y_b.numpy())
    return np.vstack(embs), np.concatenate(lbls)


if __name__ == "__main__":
    train_ds = SimulationDataset("train")
    val_ds = SimulationDataset("val",  max_snps=train_ds.max_snps)
    test_ds = SimulationDataset("test", max_snps=train_ds.max_snps)
    mm = build_margin_matrix(NEUTRALITY_TIMES, BASE_MARGIN)
    print(f"\nMargin range: {mm.min():.3f} – {mm.max():.3f}")
    train_loader = DataLoader(train_ds, batch_sampler=BalancedBatchSampler(train_ds.y, N_CLS_PER_BATCH, N_PER_CLS))
    val_loader = DataLoader(val_ds,batch_sampler=BalancedBatchSampler(val_ds.y, N_CLS_PER_BATCH, N_PER_CLS))
    seq_loaders = {
        s: DataLoader(ds, batch_size=16, shuffle=False)
        for s, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimulationEmbedder().to(device)
    print(f"\nDevice: {device}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch:  {N_CLS_PER_BATCH} × {N_PER_CLS} = "
          f"{N_CLS_PER_BATCH * N_PER_CLS} simulations\n")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer, EPOCHS, WARMUP_EPOCHS)
    print(f"{'Ep':>4}  {'train':>9}  {'val':>9}  {'lr':>9}")
    print("-" * 38)
    best_val, best_state, history = float("inf"), None, []
    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, mm, device, optimizer)
        vl = run_epoch(model, val_loader,   mm, device)
        scheduler.step()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        history.append({"epoch": epoch, "tr": tr, "vl": vl})
        print(f"{epoch:4d}  {tr:9.5f}  {vl:9.5f}, {optimizer.param_groups[0]['lr']:.2e}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print(f"\nBest val loss: {best_val:.5f}")
    torch.save(best_state, "best_sim_embedder.pt")
    model.load_state_dict(best_state)

    tr_emb, tr_lbl = get_embeddings(model, seq_loaders["train"], device)
    vl_emb, vl_lbl = get_embeddings(model, seq_loaders["val"], device)
    te_emb, te_lbl = get_embeddings(model, seq_loaders["test"], device)

    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn.fit(tr_emb, tr_lbl)
    te_pred = knn.predict(te_emb)

    print(f"\nVal  KNN-5: {knn.score(vl_emb, vl_lbl) * 100:.1f}%")
    print(f"Test KNN-5: {knn.score(te_emb, te_lbl) * 100:.1f}%")
    print(classification_report(te_lbl, te_pred, target_names=CLASS_NAMES))
