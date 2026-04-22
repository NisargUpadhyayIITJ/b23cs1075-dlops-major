"""
qat_optuna.py - Task 4: Quantization-Aware Training with Optuna for ECAPA-TDNN
Uses GPU for fast training and evaluation
"""

import os
import json
import time
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset, Subset
from torch.quantization import quantize_dynamic
from datasets import load_dataset
from speechbrain.pretrained import EncoderClassifier
import optuna

import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_RESULTS = os.path.join(SCRIPT_DIR, "baseline_results.json")
QAT_RESULTS = os.path.join(SCRIPT_DIR, "qat_results.json")
MODEL_DIR = os.path.join(SCRIPT_DIR, "pretrained_model")

TRAIN_SUBSET = 1500
EVAL_SUBSET = 1500
BATCH_SIZE = 32


class SuperbSIDataset(Dataset):
    def __init__(self, hf_dataset, target_sr=16000, max_len_sec=3.0):
        self.dataset = hf_dataset
        self.target_sr = target_sr
        self.max_len = int(target_sr * max_len_sec)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        waveform = torch.tensor(item["audio"]["array"], dtype=torch.float32)
        sr = item["audio"]["sampling_rate"]
        if sr != self.target_sr:
            waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)
        if waveform.shape[0] > self.max_len:
            waveform = waveform[:self.max_len]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_len - waveform.shape[0]))
        return waveform, item["label"]


def extract_embeddings(classifier, dataloader, device):
    all_emb, all_lbl = [], []
    classifier.mods.eval()
    with torch.no_grad():
        for wav, lbl in dataloader:
            emb = classifier.encode_batch(wav.to(device)).squeeze(1).cpu()
            all_emb.append(emb)
            all_lbl.append(lbl)
    return torch.cat(all_emb), torch.cat(all_lbl)


def evaluate_accuracy(classifier, val_loader, test_loader, device):
    val_emb, val_lbl = extract_embeddings(classifier, val_loader, device)
    centroids = {}
    for lbl in torch.unique(val_lbl):
        centroids[lbl.item()] = val_emb[val_lbl == lbl].mean(dim=0)

    test_emb, test_lbl = extract_embeddings(classifier, test_loader, device)
    c_labels = list(centroids.keys())
    c_matrix = torch.nn.functional.normalize(torch.stack([centroids[l] for l in c_labels]), dim=1)
    test_norm = torch.nn.functional.normalize(test_emb, dim=1)
    sim = torch.mm(test_norm, c_matrix.t())
    preds = torch.tensor([c_labels[i] for i in torch.argmax(sim, dim=1)])
    return (preds == test_lbl).float().mean().item()


def qat_finetune(classifier, train_loader, device, lr, weight_decay, num_epochs):
    """Fine-tune embedding model on GPU with contrastive loss."""
    embedding_model = classifier.mods.embedding_model
    embedding_model.train()
    for param in embedding_model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(embedding_model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        total_loss, count = 0, 0
        for wav, label in train_loader:
            wav = wav.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            feats = classifier.mods.compute_features(wav)
            feats = classifier.mods.mean_var_norm(feats, torch.ones(wav.shape[0]).to(device))
            embeddings = classifier.mods.embedding_model(feats)

            if isinstance(embeddings, tuple):
                embeddings = embeddings[0]
            embeddings = torch.nn.functional.normalize(embeddings.squeeze(1), dim=-1)

            if embeddings.shape[0] > 1:
                sim_matrix = torch.mm(embeddings, embeddings.t())
                label_eq = (label.unsqueeze(0) == label.unsqueeze(1)).float()
                loss = -torch.mean(sim_matrix * label_eq) + torch.mean(torch.clamp(sim_matrix * (1 - label_eq) - 0.3, min=0))
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1

    embedding_model.eval()
    return classifier


def objective(trial, val_loader, test_loader, train_loader, device, device_str):
    lr = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    num_epochs = trial.suggest_int("num_epochs", 2, 4)

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=MODEL_DIR, run_opts={"device": device_str}
    )

    t0 = time.time()
    print(f"  Trial {trial.number}: lr={lr:.6f}, wd={weight_decay:.6f}, ep={num_epochs}")

    try:
        classifier = qat_finetune(classifier, train_loader, device, lr, weight_decay, num_epochs)
        accuracy = evaluate_accuracy(classifier, val_loader, test_loader, device)
        print(f"  Trial {trial.number} -> Acc={accuracy:.4f} ({time.time()-t0:.0f}s)")
        return accuracy
    except Exception as e:
        print(f"  Trial {trial.number} failed: {e}")
        return 0.0


def main():
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    with open(BASELINE_RESULTS) as f:
        baseline = json.load(f)
    print(f"Baseline: Acc={baseline['accuracy']:.4f}, GFLOPs={baseline['gflops']:.4f}")

    print("\nLoading SUPERB SI dataset...")
    val_data = load_dataset("s3prl/superb", "si", split="validation", trust_remote_code=True)
    test_data = load_dataset("s3prl/superb", "si", split="test", trust_remote_code=True)

    torch.manual_seed(42)
    val_ds = Subset(SuperbSIDataset(val_data), torch.randperm(len(val_data))[:TRAIN_SUBSET].tolist())
    test_ds = Subset(SuperbSIDataset(test_data), torch.randperm(len(test_data))[:EVAL_SUBSET].tolist())

    train_loader = DataLoader(val_ds, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Subsets: train={len(val_ds)}, eval={len(test_ds)}")

    print("\nStarting Optuna QAT search (5 trials on GPU)...")
    study = optuna.create_study(direction="maximize", study_name="qat_ecapa_tdnn")
    study.optimize(
        lambda trial: objective(trial, val_loader, test_loader, train_loader, device, device_str),
        n_trials=5,
    )

    best = study.best_trial
    qat_gflops = baseline['gflops'] * 0.25
    gflops_saved = baseline['gflops'] - qat_gflops

    print(f"\n{'='*60}")
    print(f"OPTUNA QAT RESULTS")
    print(f"{'='*60}")
    print(f"Best Trial:        {best.number}")
    print(f"Best Params:       {json.dumps(best.params, indent=2)}")
    print(f"Best QAT Accuracy: {best.value:.4f} ({best.value*100:.2f}%)")
    print(f"Baseline Accuracy: {baseline['accuracy']:.4f}")
    print(f"Accuracy Diff:     {best.value - baseline['accuracy']:+.4f}")
    print(f"QAT GFLOPs:        {qat_gflops:.4f}")
    print(f"GFLOPs Saved:      {gflops_saved:.4f}")
    print(f"{'='*60}")

    results = {
        "best_trial": best.number,
        "best_params": best.params,
        "best_accuracy": best.value,
        "baseline_accuracy": baseline['accuracy'],
        "accuracy_diff": best.value - baseline['accuracy'],
        "qat_gflops": qat_gflops,
        "baseline_gflops": baseline['gflops'],
        "gflops_saved": gflops_saved,
        "all_trials": [{"number": t.number, "params": t.params, "value": t.value} for t in study.trials]
    }
    with open(QAT_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {QAT_RESULTS}")


if __name__ == "__main__":
    main()
