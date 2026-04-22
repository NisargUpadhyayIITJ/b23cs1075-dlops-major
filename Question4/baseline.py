"""
baseline.py - Task 1: Baseline Inference and Basic Profiling for ECAPA-TDNN
- Load pre-trained ECAPA-TDNN from SpeechBrain
- Evaluate on SUPERB SI dataset using embedding-based classification
- Compute Top-1 Accuracy and GFLOPs
"""

import os
import json
import torch
import torchaudio
import numpy as np
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from speechbrain.pretrained import EncoderClassifier

import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(SCRIPT_DIR, "baseline_results.json")
MODEL_DIR = os.path.join(SCRIPT_DIR, "pretrained_model")


class SuperbSIDataset(Dataset):
    """Dataset wrapper for SUPERB Speaker Identification."""
    def __init__(self, hf_dataset, target_sr=16000, max_len_sec=5.0):
        self.dataset = hf_dataset
        self.target_sr = target_sr
        self.max_len = int(target_sr * max_len_sec)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio = item["audio"]
        waveform = torch.tensor(audio["array"], dtype=torch.float32)
        sr = audio["sampling_rate"]

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)

        if waveform.shape[0] > self.max_len:
            waveform = waveform[:self.max_len]
        elif waveform.shape[0] < self.max_len:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_len - waveform.shape[0]))

        label = item["label"]
        return waveform, label


def extract_embeddings(classifier, dataloader, device):
    """Extract speaker embeddings for all samples."""
    all_embeddings = []
    all_labels = []
    classifier.mods.eval()
    with torch.no_grad():
        for i, (waveform, label) in enumerate(dataloader):
            waveform = waveform.to(device)
            # Extract embedding using SpeechBrain's encode_batch
            embedding = classifier.encode_batch(waveform)
            # embedding shape: (batch, 1, embed_dim) -> squeeze
            embedding = embedding.squeeze(1).cpu()
            all_embeddings.append(embedding)
            all_labels.append(label)

            if (i + 1) % 500 == 0:
                print(f"  Extracted {i+1}/{len(dataloader)} embeddings")

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_embeddings, all_labels


def compute_centroids(embeddings, labels):
    """Compute per-class centroid embeddings."""
    unique_labels = torch.unique(labels)
    centroids = {}
    for lbl in unique_labels:
        mask = labels == lbl
        class_embeds = embeddings[mask]
        centroids[lbl.item()] = class_embeds.mean(dim=0)
    return centroids


def classify_nearest_centroid(embeddings, centroids):
    """Classify using cosine similarity to nearest centroid."""
    # Stack centroids
    centroid_labels = list(centroids.keys())
    centroid_matrix = torch.stack([centroids[l] for l in centroid_labels])  # (num_classes, embed_dim)

    # Normalize
    embeddings_norm = torch.nn.functional.normalize(embeddings, dim=1)
    centroid_norm = torch.nn.functional.normalize(centroid_matrix, dim=1)

    # Cosine similarity
    similarity = torch.mm(embeddings_norm, centroid_norm.t())  # (num_samples, num_classes)
    pred_indices = torch.argmax(similarity, dim=1)
    predictions = torch.tensor([centroid_labels[idx] for idx in pred_indices])
    return predictions


def main():
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Load pre-trained ECAPA-TDNN
    print("Loading ECAPA-TDNN model from SpeechBrain...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=MODEL_DIR,
        run_opts={"device": device_str}
    )

    # Count parameters
    total_params, trainable_params = 0, 0
    seen = set()
    for key, mod in classifier.mods.items():
        if isinstance(mod, torch.nn.Module):
            for name, param in mod.named_parameters():
                if id(param) not in seen:
                    seen.add(id(param))
                    total_params += param.numel()
                    if param.requires_grad:
                        trainable_params += param.numel()

    print(f"Total parameters: {total_params:,}")

    # Compute GFLOPs
    print("\nComputing GFLOPs...")
    sample_waveform = torch.randn(1, 80000).to(device)
    try:
        from thop import profile
        model_to_profile = classifier.mods.embedding_model
        model_to_profile.eval()
        with torch.no_grad():
            feats = classifier.mods.compute_features(sample_waveform)
            feats = classifier.mods.mean_var_norm(feats, torch.ones(1).to(device))
        flops, _ = profile(model_to_profile, inputs=(feats,), verbose=False)
        gflops = flops / 1e9
    except Exception as e:
        print(f"GFLOPs computation error: {e}")
        gflops = -1.0

    print(f"Computational cost: {gflops:.4f} GFLOPs")

    # Load datasets
    print("\nLoading SUPERB SI dataset...")
    val_dataset = load_dataset("s3prl/superb", "si", split="validation", trust_remote_code=True)
    test_dataset = load_dataset("s3prl/superb", "si", split="test", trust_remote_code=True)

    num_speakers = len(set(test_dataset["label"]))
    print(f"Val set size: {len(val_dataset)}, Test set size: {len(test_dataset)}")
    print(f"Number of speakers: {num_speakers}")

    val_ds = SuperbSIDataset(val_dataset, target_sr=16000)
    test_ds = SuperbSIDataset(test_dataset, target_sr=16000)

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    # Extract embeddings for val set (to build centroids)
    print("\nExtracting val set embeddings (for centroids)...")
    val_embeddings, val_labels = extract_embeddings(classifier, val_loader, device)
    print(f"Val embeddings shape: {val_embeddings.shape}")

    # Compute per-class centroids from val set
    print("Computing class centroids...")
    centroids = compute_centroids(val_embeddings, val_labels)
    print(f"Number of centroid classes: {len(centroids)}")

    # Extract test embeddings
    print("\nExtracting test set embeddings...")
    test_embeddings, test_labels = extract_embeddings(classifier, test_loader, device)
    print(f"Test embeddings shape: {test_embeddings.shape}")

    # Classify test set using nearest centroid
    print("Classifying test set...")
    predictions = classify_nearest_centroid(test_embeddings, centroids)
    correct = (predictions == test_labels).sum().item()
    total = len(test_labels)
    accuracy = correct / total

    print(f"\n{'='*60}")
    print(f"BASELINE RESULTS")
    print(f"{'='*60}")
    print(f"Top-1 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"GFLOPs:         {gflops:.4f}")
    print(f"Total Params:   {total_params:,}")
    print(f"{'='*60}")

    # Save results
    results = {
        "accuracy": accuracy,
        "gflops": gflops,
        "total_params": total_params,
        "num_speakers": num_speakers,
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
