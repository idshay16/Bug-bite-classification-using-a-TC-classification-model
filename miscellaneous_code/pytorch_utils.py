import itertools
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm


def get_pytorch_loaders(train_dir, val_dir, img_size=310, batch_size=4):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    val_ds   = datasets.ImageFolder(val_dir,   transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, train_ds.classes


def train_pytorch_model(model, train_loader, val_loader, device,
                        phase1_epochs=5, phase2_epochs=20, patience=5, save_path=None,
                        phase1_batch_size=16):
    torch.cuda.empty_cache()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}

    p1_train_loader = DataLoader(train_loader.dataset, batch_size=phase1_batch_size,
                                 shuffle=True, num_workers=0)
    p1_val_loader   = DataLoader(val_loader.dataset,   batch_size=phase1_batch_size,
                                 shuffle=False, num_workers=0)

    def run_epoch(loader, optimizer=None, desc=''):
        training = optimizer is not None
        model.train(training)
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        bar = tqdm(loader, desc=desc, leave=False,
                   bar_format='{l_bar}{bar:30}{r_bar}')
        with torch.set_grad_enabled(training):
            for imgs, labels in bar:
                imgs, labels = imgs.to(device), labels.to(device)
                if training:
                    optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, labels)
                if training:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * imgs.size(0)
                preds = out.argmax(1)
                correct += (preds == labels).sum().item()
                total += imgs.size(0)
                if not training:
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                bar.set_postfix(loss=f'{total_loss/total:.4f}',
                                acc=f'{correct/total:.4f}')
        prec = precision_score(all_labels, all_preds, average='macro', zero_division=0) if not training else None
        rec  = recall_score(all_labels, all_preds, average='macro', zero_division=0)    if not training else None
        return total_loss / total, correct / total, prec, rec

    best_val_loss, patience_ctr, best_state = float('inf'), 0, None

    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-2)
    def run_phase(n_epochs, optimizer, phase_name, t_loader, v_loader):
        nonlocal best_val_loss, patience_ctr, best_state
        print(f'{phase_name}')
        epoch_bar = tqdm(range(n_epochs), desc='epochs',
                         bar_format='{l_bar}{bar:20}{r_bar}')
        for epoch in epoch_bar:
            t0 = time.time()
            tr_loss, tr_acc, _, _            = run_epoch(t_loader, optimizer,
                                                         desc=f'train {epoch+1}/{n_epochs}')
            va_loss, va_acc, va_prec, va_rec = run_epoch(v_loader,
                                                         desc=f'val   {epoch+1}/{n_epochs}')
            elapsed = time.time() - t0
            history['loss'].append(tr_loss); history['acc'].append(tr_acc)
            history['val_loss'].append(va_loss); history['val_acc'].append(va_acc)
            if va_loss < best_val_loss:
                best_val_loss, patience_ctr = va_loss, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if save_path:
                    torch.save(best_state, save_path)
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    epoch_bar.set_description(f'early stop @ {epoch+1}')
                    break
            epoch_bar.set_postfix(
                loss=f'{tr_loss:.4f}', acc=f'{tr_acc:.4f}',
                val_loss=f'{va_loss:.4f}', val_acc=f'{va_acc:.4f}',
                prec=f'{va_prec:.4f}', rec=f'{va_rec:.4f}',
                pat=f'{patience_ctr}/{patience}', t=f'{elapsed:.0f}s'
            )

    run_phase(phase1_epochs, optimizer, 'Phase 1: training classifier head',
              p1_train_loader, p1_val_loader)

    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-2)
    patience_ctr = 0
    run_phase(phase2_epochs, optimizer, 'Phase 2: fine-tuning full backbone',
              train_loader, val_loader)

    if best_state:
        model.load_state_dict(best_state)
    return history


def plot_pytorch_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(history['acc'], label='Train Acc')
    ax1.plot(history['val_acc'], label='Val Acc')
    ax1.set_title('Accuracy'); ax1.legend()
    ax2.plot(history['loss'], label='Train Loss')
    ax2.plot(history['val_loss'], label='Val Loss')
    ax2.set_title('Loss'); ax2.legend()
    plt.tight_layout(); plt.show()


def evaluate_pytorch_model(model, val_loader, class_names, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    print(classification_report(all_labels, all_preds, target_names=class_names))
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.viridis)
    plt.title('Confusion Matrix'); plt.colorbar()
    ticks = range(len(class_names))
    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)
    thresh = cm.max() * 0.8
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], ha='center',
                 color='black' if cm[i, j] > thresh else 'white')
    plt.tight_layout(); plt.show()
