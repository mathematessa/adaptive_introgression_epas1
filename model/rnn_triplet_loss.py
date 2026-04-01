import gc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

NEUTRALITY_TIMES = torch.tensor([0, 50, 100, 200, 500, 750, 1000, 1300, 1600, 1900], dtype=torch.float32)
 
def _build_ordinal_margin_matrix(times: torch.Tensor, base_margin: float) -> torch.Tensor:
    diff = (times.unsqueeze(0) - times.unsqueeze(1)).abs()
    margin_mat = base_margin * (diff / diff.max()).clamp(min=0.1)
    return margin_mat
 
def _pairwise_distance(embeddings: torch.Tensor, squared: bool = False) -> torch.Tensor:
    dot = embeddings @ embeddings.T
    sq = torch.diag(dot)
    d = (sq.unsqueeze(1) - 2.0 * dot + sq.unsqueeze(0)).clamp(min=0.0)
    if not squared:
        mask = (d == 0.0).float()
        d = torch.sqrt(d + mask * 1e-16) * (1.0 - mask)
    return d
 
def _masked_maximum(data, mask, dim=1):
    axis_min = data.min(dim=dim, keepdim=True).values
    return ((data - axis_min) * mask).max(dim=dim, keepdim=True).values + axis_min
 
 
def _masked_minimum(data, mask, dim=1):
    axis_max = data.max(dim=dim, keepdim=True).values
    return ((data - axis_max) * mask).min(dim=dim, keepdim=True).values + axis_max
 
 
def triplet_semihard_loss(
    labels: torch.Tensor,
    embeddings: torch.Tensor,
    margin: float = 0.5,
    margin_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    device = embeddings.device
    B = labels.size(0)
    embeddings = embeddings.float()
 
    pdist = _pairwise_distance(embeddings, squared=False)
    lc = labels.unsqueeze(1)
    adjacency = lc == lc.T
    adjacency_not = ~adjacency
    pdist_tile = pdist.repeat(B, 1)
    mask = (adjacency_not.repeat(B, 1) & (pdist_tile > pdist.T.reshape(-1, 1)))
    mask_final = (mask.float().sum(dim=1, keepdim=True) > 0.0).reshape(B, B).T
    neg_outside = _masked_minimum(pdist_tile, mask.float()).reshape(B, B).T
    neg_inside = _masked_maximum(pdist, adjacency_not.float()).expand(-1, B)
    semihard_neg = torch.where(mask_final, neg_outside, neg_inside)
    if margin_matrix is not None:
        pair_margin = margin_matrix.to(device)[labels.unsqueeze(1), labels.unsqueeze(0)]
    else:
        pair_margin = torch.full((B, B), margin, device=device)
 
    loss_mat = pair_margin + pdist - semihard_neg
    mask_pos = (adjacency.float() - torch.eye(B, device=device)).clamp(min=0.0)
    n_pos = mask_pos.sum()
    return torch.clamp(loss_mat * mask_pos, min=0.0).sum() / (n_pos + 1e-9)

class BalancedBatchSampler(Sampler):
    def __init__(self, labels: np.ndarray, n_classes: int, n_samples_per_class: int):
        super().__init__()
        self.n_classes = n_classes
        self.n_samples = n_samples_per_class
        self.class_indices = defaultdict(list)
        for idx, lbl in enumerate(labels):
            self.class_indices[int(lbl)].append(idx)
        self.all_classes = sorted(self.class_indices.keys())
        self.n_batches = len(labels) // (n_classes * n_samples_per_class)
 
    def __iter__(self):
        for _ in range(self.n_batches):
            batch, chosen = [], np.random.choice(self.all_classes, size=self.n_classes, replace=False)
            for cls in chosen:
                idxs = self.class_indices[cls]
                batch.extend(
                    np.random.choice(
                        idxs, size=self.n_samples,
                        replace=len(idxs) < self.n_samples
                    ).tolist()
                )
            yield batch
 
    def __len__(self):
        return self.n_batches
 
 class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()
 
    def forward(self, x):
        return self.act(x + self.net(x))
 
 
class TLEmbedder(nn.Module):
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 128,
        n_blocks: int = 3,
        embed_dim: int = 64,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.output_proj = nn.Linear(hidden_dim, embed_dim)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        return F.normalize(self.output_proj(self.blocks(self.input_proj(x))), p=2, dim=1)
 
TLRNNEmbedder = TLEmbedder

class TLDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def get_scheduler(optimizer, n_epochs: int, warmup_epochs: int = 5):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        p = (epoch - warmup_epochs) / (n_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def get_embeddings(model, loader, device):
    model.eval()
    embs, lbls = [], []
    for X_b, y_b in loader:
        embs.append(model(X_b.to(device)).cpu().numpy())
        lbls.append(y_b.numpy())
    return np.vstack(embs), np.concatenate(lbls)


def train_epoch(model, loader, optimizer, margin, device, margin_matrix=None):
    model.train()
    total, count = 0.0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = triplet_semihard_loss(y_b, model(X_b), margin, margin_matrix)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item(); count += 1
    return total / max(count, 1)
 
 
@torch.no_grad()
def eval_epoch(model, loader, margin, device, margin_matrix=None):
    model.eval()
    total, count = 0.0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        total += triplet_semihard_loss(y_b, model(X_b), margin, margin_matrix).item()
        count += 1
    return total / max(count, 1)


DATA_DIR = Path("dataset_merged/learning_ready")

X_train = np.load(DATA_DIR / "X_train.npy")
X_val = np.load(DATA_DIR / "X_val.npy")
X_test = np.load(DATA_DIR / "X_test.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
y_val = np.load(DATA_DIR / "y_val.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
print(f"Class counts (train): {np.bincount(y_train)}")

N_CLASSES_PER_BATCH = 10
N_SAMPLES_PER_CLASS = 20
BASE_MARGIN = 0.5
EMBED_DIM = 64
EPOCHS = 20
LR = 1e-3
WARMUP_EPOCHS = 5
USE_ORDINAL_MARGIN  = True

margin_matrix = None
if USE_ORDINAL_MARGIN:
    margin_matrix = _build_ordinal_margin_matrix(NEUTRALITY_TIMES, BASE_MARGIN)
    print(f"\nOrdinal margin range: {margin_matrix.min():.3f} – {margin_matrix.max():.3f}")

train_ds = TLDataset(X_train, y_train)
val_ds = TLDataset(X_val, y_val)
test_ds  = TLDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_sampler=BalancedBatchSampler(y_train, N_CLASSES_PER_BATCH, N_SAMPLES_PER_CLASS))
val_loader_balanced = DataLoader(val_ds,batch_sampler=BalancedBatchSampler(y_val, N_CLASSES_PER_BATCH, N_SAMPLES_PER_CLASS))
val_loader_seq = DataLoader(val_ds,  batch_size=512, shuffle=False)
test_loader_seq = DataLoader(test_ds, batch_size=512, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

model = TLEmbedder(input_dim=7, hidden_dim=128, n_blocks=3, embed_dim=EMBED_DIM, dropout=0.15).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = get_scheduler(optimizer, EPOCHS, WARMUP_EPOCHS)
best_val = float("inf")

for epoch in range(1, EPOCHS + 1):
    tr = train_epoch(model, train_loader, optimizer, BASE_MARGIN, device, margin_matrix)
    val = eval_epoch(model, val_loader_balanced, BASE_MARGIN, device, margin_matrix)
    scheduler.step()
    if val < best_val:
        best_val = val
        torch.save(model.state_dict(), "best_embedder.pt")
    print(f"Epoch {epoch:3d}  train={tr:.4f}  val={val:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

print(f"\nBest val loss: {best_val:.4f}")

model.load_state_dict(torch.load("best_embedder.pt", map_location=device))
train_seq = DataLoader(train_ds, batch_size=512, shuffle=False)
tr_emb, tr_lbl = get_embeddings(model, train_seq,      device)
vl_emb, vl_lbl = get_embeddings(model, val_loader_seq, device)
te_emb, te_lbl = get_embeddings(model, test_loader_seq, device)

knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn.fit(tr_emb, tr_lbl)
te_pred = knn.predict(te_emb)

print(f"\nVal  accuracy (KNN-5): {knn.score(vl_emb, vl_lbl)*100:.1f}%")
print(f"Test accuracy (KNN-5): {knn.score(te_emb, te_lbl)*100:.1f}%")
