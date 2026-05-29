"""
Hypothesis check: does TC pre-training produce better bug-bite representations?

For each model (ConvNeXt, DenseNet-121, InceptionV3) compares:
  - Classification accuracy + per-class report  (TC vs baseline)
  - Silhouette score on backbone embeddings (higher = tighter, better-separated classes)
  - t-SNE plots side-by-side (TC vs baseline)

Run from the repo root:
    source ~/tc-env/bin/activate
    python miscellaneous_code/hypothesis_check.py
"""

from pathlib import Path
import numpy as np
import torch
import timm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, silhouette_score, accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

PROJECT_ROOT = Path("/mnt/c/Users/test/OneDrive - Braude College of Engineering/"
                    "Software Engineering Stuff/Capstone/"
                    "Bug-bite-classification-using-a-TC-classification-model")
MODEL_DIR = PROJECT_ROOT / "Model_Weights"
VAL_DIR   = "/home/test/bug_data/val"
OUT_DIR   = PROJECT_ROOT / "hypothesis_results"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}\n")

MODELS = [
    dict(name="ConvNeXt-Tiny", timm_id="convnext_tiny.fb_in22k_ft_in1k",
         img_size=256,
         baseline=MODEL_DIR / "multiclass_convnext_best.pt",
         tc=MODEL_DIR / "multiclass_convnext_tc_best.pt"),
    dict(name="DenseNet-121",  timm_id="densenet121.ra_in1k",
         img_size=256,
         baseline=MODEL_DIR / "multiclass_densenet121_best.pt",
         tc=MODEL_DIR / "multiclass_densenet121_tc_best.pt"),
    dict(name="InceptionV3",   timm_id="inception_v3.tv_in1k",
         img_size=299,
         baseline=MODEL_DIR / "multiclass_inceptionv3_best.pt",
         tc=MODEL_DIR / "multiclass_inceptionv3_tc_best.pt"),
]

_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CLASS_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]


def get_loader(img_size):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        _NORM,
    ])
    ds = datasets.ImageFolder(VAL_DIR, transform=tf)
    return DataLoader(ds, batch_size=32, shuffle=False,
                      num_workers=4, pin_memory=True), ds.classes


def load_full_model(timm_id, n_classes, weight_path):
    m = timm.create_model(timm_id, pretrained=False, num_classes=n_classes)
    m.load_state_dict(torch.load(weight_path, map_location="cpu"))
    return m.to(DEVICE).eval()


def get_predictions(model, loader):
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="  preds", leave=False):
            preds.append(model(imgs.to(DEVICE)).argmax(1).cpu().numpy())
            trues.append(labels.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def get_embeddings(timm_id, n_classes, weight_path, loader):
    # Load model, strip head to get raw backbone features
    m = timm.create_model(timm_id, pretrained=False, num_classes=n_classes)
    m.load_state_dict(torch.load(weight_path, map_location="cpu"))
    m.reset_classifier(0)   # timm: replaces head with Identity
    m = m.to(DEVICE).eval()
    feats, trues = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="  embeds", leave=False):
            feats.append(m(imgs.to(DEVICE)).cpu().numpy())
            trues.append(labels.numpy())
    del m
    return np.concatenate(feats), np.concatenate(trues)


summary = []

for spec in MODELS:
    name, timm_id, img_size = spec["name"], spec["timm_id"], spec["img_size"]
    print(f"\n{'='*60}\n  {name}\n{'='*60}")

    loader, classes = get_loader(img_size)
    n_cls = len(classes)
    row = {"model": name}

    for label, wpath in [("baseline", spec["baseline"]), ("tc", spec["tc"])]:
        print(f"\n  [{label}]  {wpath.name}")

        # Accuracy
        m = load_full_model(timm_id, n_cls, wpath)
        preds, trues = get_predictions(m, loader)
        del m
        torch.cuda.empty_cache()

        acc = accuracy_score(trues, preds)
        print(f"  Accuracy: {acc:.4f}")
        print(classification_report(trues, preds, target_names=classes, digits=3))

        # Embeddings + silhouette
        feats, _ = get_embeddings(timm_id, n_cls, wpath, loader)
        torch.cuda.empty_cache()

        sil = silhouette_score(feats, trues,
                               sample_size=min(2000, len(trues)), random_state=42)
        print(f"  Silhouette score: {sil:.4f}")

        # t-SNE
        print("  t-SNE...")
        perp = min(30, len(trues) // 4)
        emb2d = TSNE(n_components=2, perplexity=perp,
                     random_state=42, n_iter=1000).fit_transform(feats)

        row[label] = dict(acc=acc, sil=sil, emb=emb2d, trues=trues, preds=preds)

    summary.append(row)

    # t-SNE plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{name} — backbone embeddings t-SNE (bug-bite val)", fontsize=13)
    for ax, label, title in [
        (axes[0], "baseline", "Baseline (ImageNet only)"),
        (axes[1], "tc",       "TC pre-trained"),
    ]:
        d = row[label]
        for ci, cls in enumerate(classes):
            mask = d["trues"] == ci
            ax.scatter(d["emb"][mask, 0], d["emb"][mask, 1],
                       s=12, alpha=0.7, color=CLASS_COLORS[ci], label=cls)
        ax.set_title(f"{title}\nAcc={d['acc']:.3f}   Sil={d['sil']:.3f}", fontsize=10)
        ax.legend(fontsize=7, markerscale=2)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    out = OUT_DIR / f"tsne_{name.replace(' ', '_').replace('-', '')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

# Summary table
print(f"\n{'='*65}")
print(f"{'Model':<18} {'Base Acc':>9} {'TC Acc':>9} {'ΔAcc':>7}  "
      f"{'Base Sil':>9} {'TC Sil':>9} {'ΔSil':>7}")
print("-" * 65)
for r in summary:
    b, tc = r["baseline"], r["tc"]
    print(f"{r['model']:<18} {b['acc']:>9.4f} {tc['acc']:>9.4f} {tc['acc']-b['acc']:>+7.4f}  "
          f"{b['sil']:>9.4f} {tc['sil']:>9.4f} {tc['sil']-b['sil']:>+7.4f}")
print("=" * 65)
print(f"\nPlots saved to: {OUT_DIR}")
