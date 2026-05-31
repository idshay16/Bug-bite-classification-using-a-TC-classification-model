"""
Section 4 — Performance evaluation and XAI (LIME, SHAP, DiCE, t-SNE, ensemble).

Usage (called by run_experiments.py):
    python scripts/04_xai.py \\
        --run-id s0_ls0.10_lr5.0e-06 \\
        --ctrl-convnext <path> --ctrl-densenet <path> --ctrl-inception <path> \\
        --tc-convnext   <path> --tc-densenet   <path> --tc-inception   <path>
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torchvision.transforms as _T

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'miscellaneous_code'))

from shared import (
    BUG_VAL, MODEL_SPECS, N_CLASSES, RESULTS_DIR,
    _MEAN_NP, _STD_NP, full_metrics, get_pytorch_loaders,
    get_pt_probs_preds, load_model, out_dir, savefig,
)

BUG_TRAIN     = '/home/test/bug_data/train'
SHAP_BG_N     = 50
LIME_SAMPLES  = 1000
LIME_PER_CLS  = 3
DICE_PCA_DIMS = 50
DICE_N_CF     = 3


def _denorm_img(t):
    img = t.cpu().permute(1, 2, 0).numpy()
    return np.clip(img * _STD_NP + _MEAN_NP, 0, 1)


def _collect_bg_queries(loader, n_bg, n_classes):
    bg_list, queries = [], {}
    for imgs, labels in loader:
        for img, lbl in zip(imgs, labels):
            l = lbl.item()
            if l not in queries:
                queries[l] = img.unsqueeze(0)
        bg_list.append(imgs)
        if sum(x.shape[0] for x in bg_list) >= n_bg and len(queries) == n_classes:
            break
    return torch.cat(bg_list)[:n_bg], queries


def run_tsne(model, loader, classes, device, out_path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_score

    model.eval()
    feats_list, labels_list = [], []
    cap = {}

    def _hook(mod, inp, out):
        cap['f'] = out.detach().cpu()

    # Attach hook to the last non-classifier layer
    last_layer = None
    for name, mod in model.named_modules():
        if 'classifier' not in name and 'head' not in name and 'fc' not in name:
            last_layer = mod
    h = last_layer.register_forward_hook(_hook) if last_layer else None

    with torch.no_grad():
        for imgs, lbls in loader:
            model(imgs.to(device))
            if 'f' in cap:
                f = cap['f']
                if f.dim() > 2:
                    f = f.mean(dim=list(range(2, f.dim())))
                feats_list.append(f.numpy())
                labels_list.append(lbls.numpy())
    if h:
        h.remove()

    if not feats_list:
        return
    feats  = np.concatenate(feats_list)
    labels = np.concatenate(labels_list)

    perp = min(30, len(labels) // 4)
    emb2d = __import__('sklearn').manifold.TSNE(
        n_components=2, perplexity=perp, random_state=42, max_iter=1000
    ).fit_transform(feats)

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    fig, ax = plt.subplots(figsize=(7, 6))
    for ci, cls in enumerate(classes):
        mask = labels == ci
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1],
                   s=10, alpha=0.7, color=colors[ci % len(colors)], label=cls)
    sil = silhouette_score(feats, labels, sample_size=min(2000, len(labels)), random_state=42)
    ax.set_title(f't-SNE  Sil={sil:.4f}')
    ax.legend(fontsize=7, markerscale=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    savefig(out_path)
    return sil


def run_shap(model, loader, classes, device, model_tag, out_path):
    try:
        import shap
    except ImportError:
        print('shap not installed — skipping')
        return
    import matplotlib.pyplot as plt

    bg, queries = _collect_bg_queries(loader, SHAP_BG_N, len(classes))
    explainer = shap.GradientExplainer(model, bg.to(device))

    n_rows = len(queries)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f'SHAP — {model_tag}', fontsize=12, fontweight='bold')

    for row_i, (cls_idx, test_img) in enumerate(sorted(queries.items())):
        test_dev  = test_img.to(device)
        shap_vals = explainer.shap_values(test_dev)
        with torch.no_grad():
            pred_cls = model(test_dev).argmax(1).item()

        if isinstance(shap_vals, list):
            sv_raw = np.array(shap_vals[pred_cls])[0]
        else:
            sv_arr = np.array(shap_vals)
            if sv_arr.ndim == 5:
                sv_raw = sv_arr[0, :, :, :, pred_cls]
            else:
                sv_raw = sv_arr[0]
        sv_map = sv_raw.mean(0)
        vmax    = np.abs(sv_map).max() + 1e-8
        img_d   = _denorm_img(test_img[0])

        axes[row_i, 0].imshow(img_d)
        axes[row_i, 0].set_title(f'True: {classes[cls_idx]}')
        axes[row_i, 0].axis('off')
        axes[row_i, 1].imshow(np.maximum(sv_map, 0), cmap='Reds')
        axes[row_i, 1].set_title(f'SHAP+ (pred: {classes[pred_cls]})')
        axes[row_i, 1].axis('off')
        axes[row_i, 2].imshow(np.maximum(-sv_map, 0), cmap='Blues')
        axes[row_i, 2].set_title('SHAP−')
        axes[row_i, 2].axis('off')
        bwr_rgb = plt.cm.bwr((sv_map / vmax + 1) / 2)[:, :, :3]
        axes[row_i, 3].imshow(np.clip(img_d * 0.5 + bwr_rgb * 0.5, 0, 1))
        axes[row_i, 3].set_title('Overlay (BWR)')
        axes[row_i, 3].axis('off')

    plt.tight_layout()
    savefig(out_path)


def run_lime(model, classes, img_size, device, model_tag, out_dir_path):
    try:
        from lime.lime_image import LimeImageExplainer
        import skimage.segmentation
    except ImportError:
        print('lime/scikit-image not installed — skipping')
        return
    import matplotlib.pyplot as plt

    _norm = _T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    explainer = LimeImageExplainer()

    def _predict(images):
        model.eval()
        tensors = [_norm(torch.from_numpy(img).permute(2, 0, 1).float().div(255))
                   for img in images]
        with torch.no_grad():
            return torch.softmax(
                model(torch.stack(tensors).to(device)), dim=1).cpu().numpy()

    for cls in classes:
        cls_dir = Path(BUG_VAL) / cls
        imgs = sorted([f for f in cls_dir.iterdir()
                       if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])[:LIME_PER_CLS]
        for img_path in imgs:
            img_rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            img_res = cv2.resize(img_rgb, (img_size, img_size))
            exp = explainer.explain_instance(img_res, _predict,
                                             top_labels=1, num_samples=LIME_SAMPLES)
            top = exp.top_labels[0]
            img_vis, mask = exp.get_image_and_mask(
                top, positive_only=True, num_features=10, hide_rest=False)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle(f'LIME — {model_tag} | {cls} — {img_path.name}',
                         fontsize=10, fontweight='bold')
            axes[0].imshow(img_res)
            axes[0].set_title('Original')
            axes[0].axis('off')
            axes[1].imshow(skimage.segmentation.mark_boundaries(
                img_vis.astype(np.float32) / 255.0, mask))
            axes[1].set_title('LIME')
            axes[1].axis('off')
            plt.tight_layout()
            stem = img_path.stem[:20]
            savefig(out_dir_path / f'lime_{model_tag}_{cls}_{stem}.png')


def run_dice(model, loader, classes, device, model_tag, out_path):
    try:
        import dice_ml
    except ImportError:
        print('dice_ml not installed — skipping')
        return
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    feats_list, labels_list = [], []
    model.eval()
    # extract embeddings via reset_classifier
    import copy
    embed_model = copy.deepcopy(model)
    embed_model.reset_classifier(0)
    embed_model.eval()
    safe_loader = torch.utils.data.DataLoader(
        loader.dataset, batch_size=loader.batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for imgs, lbls in safe_loader:
            feats_list.append(embed_model(imgs.to(device)).cpu().numpy())
            labels_list.append(lbls.numpy())
    del embed_model
    torch.cuda.empty_cache()

    embs   = np.concatenate(feats_list)
    labels = np.concatenate(labels_list)

    pca      = PCA(n_components=DICE_PCA_DIMS, random_state=42)
    embs_pca = pca.fit_transform(embs).astype(np.float64)
    feat_names = [f'pc{i}' for i in range(DICE_PCA_DIMS)]
    df_all = pd.DataFrame(embs_pca, columns=feat_names)
    df_all['label'] = labels.astype(int)

    clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    clf.fit(df_all[feat_names], labels)

    try:
        d_obj   = dice_ml.Data(dataframe=df_all,
                               continuous_features=feat_names, outcome_name='label')
        m_obj   = dice_ml.Model(model=clf, backend='sklearn')
        exp_obj = dice_ml.Dice(d_obj, m_obj, method='random')
    except Exception as e:
        print(f'DiCE init failed: {e}')
        return

    fig, axes = plt.subplots(len(classes), 1,
                             figsize=(14, 3 * len(classes)), squeeze=False)
    fig.suptitle(f'DiCE Counterfactual Shifts — {model_tag}',
                 fontsize=12, fontweight='bold')

    for cls_idx, cls_name in enumerate(classes):
        mask = labels == cls_idx
        ax   = axes[cls_idx, 0]
        if not mask.any():
            ax.axis('off')
            continue
        query_row = df_all.iloc[[np.where(mask)[0][0]]][feat_names]
        try:
            cfs   = exp_obj.generate_counterfactuals(
                query_row, total_CFs=DICE_N_CF,
                desired_class=(cls_idx + 1) % len(classes), random_seed=42)
            cf_df = cfs.cf_examples_list[0].final_cfs_df
            if cf_df is None or cf_df.empty:
                ax.text(0.5, 0.5, 'No CFs found', ha='center')
                ax.axis('off')
                continue
            shifts = cf_df[feat_names].values - query_row.values
            x = np.arange(DICE_PCA_DIMS)
            for cf_i, (shift_row, cf_lbl) in enumerate(zip(shifts, cf_df['label'].values)):
                ax.bar(x + cf_i * 0.25, shift_row, width=0.25,
                       label=f'CF{cf_i+1}→{classes[int(cf_lbl)]}', alpha=0.7)
            ax.axhline(0, color='k', linewidth=0.5)
            ax.set_title(f'True: {cls_name}', fontsize=9)
            ax.set_xlabel('PCA Component')
            ax.set_ylabel('Δ embedding')
            ax.legend(fontsize=7)
        except Exception as e:
            ax.text(0.5, 0.5, f'DiCE error: {e}', ha='center', fontsize=8)
            ax.axis('off')

    plt.tight_layout()
    savefig(out_path)


def run_ensemble_metrics(ctrl_models, tc_models, sizes, classes, device, out_path, run_id):
    import matplotlib.pyplot as plt
    import os

    _norm = _T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    true_all, pred_ctrl, prob_ctrl, pred_tc, prob_tc = [], [], [], [], []

    for folder in sorted(os.listdir(BUG_VAL)):
        fp = os.path.join(BUG_VAL, folder)
        if not os.path.isdir(fp) or folder not in classes:
            continue
        label_idx = classes.index(folder)
        for fname in [f for f in os.listdir(fp)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]:
            img_raw = cv2.imread(os.path.join(fp, fname))
            if img_raw is None:
                continue
            img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            true_all.append(label_idx)
            ca, ta = np.zeros(N_CLASSES), np.zeros(N_CLASSES)
            for cm_, tm_, sz in zip(ctrl_models, tc_models, sizes):
                t = _norm(torch.from_numpy(cv2.resize(img_rgb, (sz, sz)))
                          .permute(2, 0, 1).float().div(255)).unsqueeze(0).to(device)
                with torch.no_grad():
                    ca += torch.softmax(cm_(t), 1).cpu().numpy().flatten()
                    ta += torch.softmax(tm_(t), 1).cpu().numpy().flatten()
            ca /= 3; ta /= 3
            pred_ctrl.append(int(np.argmax(ca))); prob_ctrl.append(ca.tolist())
            pred_tc.append(int(np.argmax(ta)));   prob_tc.append(ta.tolist())

    metrics_d = RESULTS_DIR / 'metrics'
    metrics_d.mkdir(parents=True, exist_ok=True)

    m_ctrl = full_metrics(true_all, pred_ctrl, prob_ctrl, classes,
                          title=f'Control Ensemble — {run_id}',
                          save_path=out_path.parent / 'metrics_ensemble_control.png')
    m_tc   = full_metrics(true_all, pred_tc, prob_tc, classes,
                          title=f'TC Ensemble — {run_id}',
                          save_path=out_path.parent / 'metrics_ensemble_tc.png')
    with open(out_path, 'w') as f:
        json.dump({'control_ensemble': m_ctrl, 'tc_ensemble': m_tc}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',          required=True)
    parser.add_argument('--ctrl-convnext',   required=True)
    parser.add_argument('--ctrl-densenet',   required=True)
    parser.add_argument('--ctrl-inception',  required=True)
    parser.add_argument('--tc-convnext',     required=True)
    parser.add_argument('--tc-densenet',     required=True)
    parser.add_argument('--tc-inception',    required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xai_d  = out_dir('xai', args.run_id)

    specs = [
        ('convnext',  MODEL_SPECS[0][1], MODEL_SPECS[0][2],
         args.ctrl_convnext, args.tc_convnext),
        ('densenet',  MODEL_SPECS[1][1], MODEL_SPECS[1][2],
         args.ctrl_densenet, args.tc_densenet),
        ('inception', MODEL_SPECS[2][1], MODEL_SPECS[2][2],
         args.ctrl_inception, args.tc_inception),
    ]

    ctrl_models, tc_models, sizes = [], [], []
    sil_results = {}

    for model_key, timm_id, img_size, ctrl_path, tc_path in specs:
        _, _, _, _, bug_batch = next(s for s in MODEL_SPECS if s[0] == model_key)
        _, val_loader, classes = get_pytorch_loaders(
            BUG_TRAIN, BUG_VAL, img_size=img_size, batch_size=bug_batch, augment=False)

        ctrl_m = load_model(timm_id, ctrl_path, device)
        tc_m   = load_model(timm_id, tc_path,   device)
        ctrl_models.append(ctrl_m)
        tc_models.append(tc_m)
        sizes.append(img_size)

        # t-SNE
        sil_ctrl = run_tsne(ctrl_m, val_loader, classes, device,
                            xai_d / f'tsne_control_{model_key}.png')
        sil_tc   = run_tsne(tc_m,   val_loader, classes, device,
                            xai_d / f'tsne_tc_{model_key}.png')
        sil_results[model_key] = {'ctrl': sil_ctrl, 'tc': sil_tc}

        # SHAP
        run_shap(ctrl_m, val_loader, classes, device,
                 f'Control {model_key}',
                 xai_d / f'shap_control_{model_key}.png')
        run_shap(tc_m,   val_loader, classes, device,
                 f'TC {model_key}',
                 xai_d / f'shap_tc_{model_key}.png')

        # LIME
        run_lime(ctrl_m, classes, img_size, device,
                 f'ctrl_{model_key}', xai_d)
        run_lime(tc_m,   classes, img_size, device,
                 f'tc_{model_key}',   xai_d)

        # DiCE
        run_dice(ctrl_m, val_loader, classes, device,
                 f'Control {model_key}',
                 xai_d / f'dice_control_{model_key}.png')
        run_dice(tc_m,   val_loader, classes, device,
                 f'TC {model_key}',
                 xai_d / f'dice_tc_{model_key}.png')

        torch.cuda.empty_cache()

    print('[04_xai] XAI_DONE', flush=True)
    # Ensemble metrics
    run_ensemble_metrics(ctrl_models, tc_models, sizes, classes, device,
                         xai_d / 'ensemble_metrics.json', args.run_id)

    # Save silhouette summary
    with open(xai_d / 'silhouette.json', 'w') as f:
        json.dump(sil_results, f, indent=2)

    for m in ctrl_models + tc_models:
        del m
    torch.cuda.empty_cache()

    print(f'[04_xai] saved → {xai_d}')


if __name__ == '__main__':
    main()
