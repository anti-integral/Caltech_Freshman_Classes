#!/usr/bin/env python3
"""
1-hour tuning script: sweep regularization configs to reduce overfitting.
Evaluates train AND val with same transform to detect overfitting honestly.
Best config = smallest train-val gap while keeping val acc >= 85%.
"""

import os, math, time, random, json, csv
import torch, torch.nn as nn, numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from PIL import Image, ImageFilter
from collections import defaultdict
from statistics import mean

IMG_SIZE = 128
SEED = 42
DEVICE = torch.device("cuda")
TIME_BUDGET_SECONDS = 3600  # 1 hour

# ============================================================================
# Model (identical to notebook)
# ============================================================================

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0: return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.floor(torch.rand(shape, dtype=x.dtype, device=x.device) + keep)
        return x / keep * mask

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.shortcut = (nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                         nn.BatchNorm2d(out_ch)) if stride != 1 or in_ch != out_ch
                         else nn.Identity())
    def forward(self, x):
        return self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x))))) + self.shortcut(x))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dp1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim), nn.Dropout(dropout))
        self.dp2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    def forward(self, x):
        y = self.norm1(x); y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.dp1(y); x = x + self.dp2(self.mlp(self.norm2(x))); return x

class HybridViT(nn.Module):
    def __init__(self, num_classes=10, input_size=128, input_channels=3,
                 embed_dim=256, num_layers=4, num_heads=8, dropout=0.1, drop_path=0.1):
        super().__init__()
        self.tokenizer = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2), BasicBlock(64, 64), BasicBlock(64, 128, stride=2), BasicBlock(128, 256, stride=2))
        self.proj = nn.Identity() if embed_dim == 256 else nn.Conv2d(256, embed_dim, 1)
        num_patches = (input_size // 16) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)]
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, 4.0, dropout, dpr[i]) for i in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)
    def forward(self, x):
        x = self.tokenizer(x); x = self.proj(x); x = x.flatten(2).transpose(1, 2)
        b = x.shape[0]; x = torch.cat((self.cls_token.expand(b, -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.transformer_blocks: x = blk(x)
        return self.fc(self.norm(x[:, 0]))


# ============================================================================
# Data
# ============================================================================

class RandomGaussianBlur:
    def __init__(self, p=0.2, radius_range=(0.5, 2.0)):
        self.p, self.radius_range = p, radius_range
    def __call__(self, img):
        if random.random() < self.p:
            return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.radius_range)))
        return img

class AddGaussianNoise:
    def __init__(self, std=0.05):
        self.std = std
    def __call__(self, t):
        return t + torch.randn_like(t) * self.std

class WildMNIST(Dataset):
    def __init__(self, imgs, lbls, transform=None):
        self.imgs, self.lbls, self.tf = imgs, lbls, transform
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img = self.imgs[i]
        if self.tf: img = self.tf(img)
        return img, self.lbls[i]

class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset, self.tf = subset, transform
    def __len__(self): return len(self.subset)
    def __getitem__(self, i):
        img, lbl = self.subset[i]
        if self.tf: img = self.tf(img)
        return img, lbl

class MNISTRGBDataset(Dataset):
    def __init__(self, ds, transform=None):
        self.ds, self.tf = ds, transform
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        img, lbl = self.ds[i]; img = img.convert('RGB')
        if self.tf: img = self.tf(img)
        return img, lbl


def load_wild():
    images, labels = [], []
    data_dir = 'data/dataset'
    for f in sorted(os.listdir(data_dir)):
        if f.endswith('.jpg'):
            with Image.open(os.path.join(data_dir, f)) as img:
                images.append(img.convert('RGB').copy())
            labels.append(int(f.split('_')[-1].replace('.jpg', '').replace('label', '')))
    return images, labels


# ============================================================================
# Training helpers
# ============================================================================

def get_lr_warmup_cosine(epoch, warmup, base_lr, total):
    if epoch < warmup: return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total - warmup, 1)
    return max(base_lr * 0.5 * (1 + math.cos(math.pi * progress)), 1e-6)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.cuda(), lbls.cuda()
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == lbls).sum().item()
        total += lbls.size(0)
    return loss_sum / total, 100. * correct / total

@torch.no_grad()
def eval_model(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.cuda(), lbls.cuda()
        out = model(imgs)
        loss_sum += criterion(out, lbls).item() * imgs.size(0)
        correct += (out.argmax(1) == lbls).sum().item()
        total += lbls.size(0)
    return loss_sum / total, 100. * correct / total


# ============================================================================
# Config space
# ============================================================================

CONFIGS = [
    # (name, dropout, drop_path, weight_decay, label_smooth, rrc_scale_min, rotation, erase_p, perspective_p, lr, warmup, batch)
    # Baseline (notebook config)
    ("baseline",         0.1,  0.1,  0.01, 0.05, 0.6, 15, 0.2, 0.2, 5e-4, 5, 96),
    # Moderate regularization
    ("mod_reg",          0.15, 0.15, 0.02, 0.1,  0.5, 15, 0.25, 0.2, 5e-4, 5, 96),
    # Strong regularization
    ("strong_reg",       0.2,  0.2,  0.03, 0.1,  0.5, 20, 0.3, 0.3, 5e-4, 5, 96),
    # Heavy regularization
    ("heavy_reg",        0.25, 0.25, 0.05, 0.15, 0.4, 25, 0.35, 0.3, 4e-4, 8, 96),
    # Moderate + lower LR
    ("mod_lowlr",        0.15, 0.15, 0.03, 0.1,  0.5, 15, 0.25, 0.2, 3e-4, 8, 96),
    # Strong + bigger batch
    ("strong_bigbatch",  0.2,  0.2,  0.03, 0.1,  0.5, 20, 0.3, 0.3, 6e-4, 5, 128),
    # Moderate with more augmentation
    ("mod_augment",      0.15, 0.15, 0.02, 0.1,  0.4, 20, 0.3, 0.3, 5e-4, 5, 96),
    # Light reg + heavy aug
    ("light_heavyaug",   0.1,  0.15, 0.02, 0.1,  0.4, 25, 0.35, 0.35, 5e-4, 5, 96),
]


def run_trial(name, dropout, drop_path, weight_decay, label_smooth, rrc_scale_min,
              rotation_deg, erase_p, perspective_p, lr, warmup_epochs, batch_size,
              images, labels, deadline):
    """Run one full pretrain + finetune trial. Returns dict with results."""
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model = HybridViT(dropout=dropout, drop_path=drop_path).cuda()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    # --- Phase 1: MNIST pretrain (10 epochs, shortened for tuning) ---
    mnist_train_tf = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(rrc_scale_min, 1.0)),
        transforms.RandomRotation(rotation_deg),
        RandomGaussianBlur(p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
        AddGaussianNoise(std=0.05),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
    ])
    mnist_val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)])

    mnist_train = datasets.MNIST(root='./data', train=True, download=False)
    mnist_val = datasets.MNIST(root='./data', train=False, download=False)
    mnist_tl = DataLoader(MNISTRGBDataset(mnist_train, mnist_train_tf),
                          batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    mnist_vl = DataLoader(MNISTRGBDataset(mnist_val, mnist_val_tf),
                          batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    pretrain_epochs = 10
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=pretrain_epochs, eta_min=1e-6)

    for ep in range(pretrain_epochs):
        if time.time() >= deadline: break
        train_epoch(model, mnist_tl, criterion, opt)
        sched.step()
    print(f"  [{name}] MNIST pretrain done")

    # --- Phase 2: Fine-tune on wild ---
    train_tf = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(rrc_scale_min, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomRotation(rotation_deg),
        transforms.RandomPerspective(distortion_scale=perspective_p, p=perspective_p),
        RandomGaussianBlur(p=0.2),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
        transforms.RandomErasing(p=erase_p, scale=(0.02, 0.2)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)])

    dataset = WildMNIST(images, labels)
    num_train = int(0.85 * len(dataset)); num_val = len(dataset) - num_train
    train_ds, val_ds = random_split(dataset, [num_train, num_val],
                                     generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(TransformSubset(train_ds, train_tf),
                              batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(TransformSubset(val_ds, val_tf),
                            batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # For honest train eval (no augmentation)
    train_eval_loader = DataLoader(TransformSubset(train_ds, val_tf),
                                   batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    finetune_epochs = 80
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0
    best_state = None
    patience_count = 0
    patience = 15

    for ep in range(finetune_epochs):
        if time.time() >= deadline:
            print(f"  [{name}] Time budget hit at epoch {ep+1}")
            break
        cur_lr = get_lr_warmup_cosine(ep, warmup_epochs, lr, finetune_epochs)
        for pg in opt.param_groups: pg['lr'] = cur_lr

        tl, ta = train_epoch(model, train_loader, criterion, opt)
        vl, va = eval_model(model, val_loader, criterion)

        if va > best_val_acc:
            best_val_acc = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if (ep + 1) % 10 == 0:
            print(f"  [{name}] ep {ep+1} train={ta:.1f}% val={va:.1f}% lr={cur_lr:.1e}")

        if patience_count >= patience:
            print(f"  [{name}] Early stop at epoch {ep+1}")
            break

    # Load best and evaluate honestly
    if best_state:
        model.load_state_dict(best_state)
        model.cuda()
    train_loss, train_acc = eval_model(model, train_eval_loader, criterion)
    val_loss, val_acc = eval_model(model, val_loader, criterion)
    gap = train_acc - val_acc

    print(f"  [{name}] FINAL: train={train_acc:.2f}% val={val_acc:.2f}% gap={gap:.2f}%")

    # Save if this is a good result
    ckpt_path = f'vit_tune_{name}.pth'
    torch.save(best_state or model.state_dict(), ckpt_path)

    return {
        "name": name, "train_acc": train_acc, "val_acc": val_acc,
        "gap": gap, "train_loss": train_loss, "val_loss": val_loss,
        "dropout": dropout, "drop_path": drop_path, "weight_decay": weight_decay,
        "label_smooth": label_smooth, "rrc_scale_min": rrc_scale_min,
        "rotation_deg": rotation_deg, "erase_p": erase_p,
        "perspective_p": perspective_p, "lr": lr, "ckpt": ckpt_path,
    }


def main():
    torch.backends.cudnn.benchmark = True
    deadline = time.time() + TIME_BUDGET_SECONDS
    print(f"Starting 1-hour tuning sweep ({len(CONFIGS)} configs)")
    print(f"Deadline: {time.strftime('%H:%M:%S', time.localtime(deadline))}")

    images, labels = load_wild()
    print(f"Loaded {len(images)} wild images\n")

    results = []
    for i, cfg in enumerate(CONFIGS):
        if time.time() >= deadline:
            print(f"Time budget reached before config {i+1}")
            break
        name = cfg[0]
        print(f"\n{'='*60}")
        print(f"Config {i+1}/{len(CONFIGS)}: {name}")
        print(f"  dropout={cfg[1]} drop_path={cfg[2]} wd={cfg[3]} ls={cfg[4]}")
        print(f"  rrc_min={cfg[5]} rot={cfg[6]} erase={cfg[7]} persp={cfg[8]} lr={cfg[9]}")
        print(f"{'='*60}")
        t0 = time.time()
        result = run_trial(*cfg, images=images, labels=labels, deadline=deadline)
        result["time_min"] = (time.time() - t0) / 60
        results.append(result)
        print(f"  Time: {result['time_min']:.1f} min")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Name':<20} {'Train%':>7} {'Val%':>7} {'Gap%':>7} {'Score':>7}")
    print("-" * 55)

    # Score: prioritize low gap while keeping val acc high
    for r in sorted(results, key=lambda r: (-r["val_acc"] + 0.5 * r["gap"])):
        score = r["val_acc"] - 0.5 * r["gap"]
        print(f"{r['name']:<20} {r['train_acc']:>7.2f} {r['val_acc']:>7.2f} {r['gap']:>7.2f} {score:>7.2f}")

    # Pick best: highest val_acc with gap < 5%
    good = [r for r in results if r["gap"] < 5.0]
    if not good:
        good = sorted(results, key=lambda r: r["gap"])[:3]
    best = max(good, key=lambda r: r["val_acc"])
    print(f"\nBEST: {best['name']} (val={best['val_acc']:.2f}%, gap={best['gap']:.2f}%)")
    print(f"Checkpoint: {best['ckpt']}")

    # Save best as vit_best.pth
    import shutil
    shutil.copy(best['ckpt'], 'vit_best_tuned.pth')
    print(f"Saved best to vit_best_tuned.pth")

    # Save results JSON
    with open('tuning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to tuning_results.json")


if __name__ == '__main__':
    main()
