"""
baseline.py - Task 1: Baseline Inference and Basic Profiling for ECAPA-TDNN
- Load pre-trained ECAPA-TDNN from SpeechBrain
- Evaluate on SUPERB SI eval split
- Compute Top-1 Accuracy and GFLOPs
"""

import os
import sys
import json
import torch
import torchaudio
import numpy as np
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from speechbrain.pretrained import EncoderClassifier

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(SCRIPT_DIR, "baseline_results.json")
MODEL_DIR = os.path.join(SCRIPT_DIR, "pretrained_model")


def count_parameters(model):
    """Count total parameters in the model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_gflops(model, input_tensor):
    """Compute GFLOPs using thop."""
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        gflops = flops / 1e9
        return gflops
    except Exception as e:
        print(f"thop profiling failed: {e}")
        # Fallback: manual estimation
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, input_tensor)
        return flops.total() / 1e9


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

        # Resample if needed
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)

        # Truncate or pad
        if waveform.shape[0] > self.max_len:
            waveform = waveform[:self.max_len]
        elif waveform.shape[0] < self.max_len:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_len - waveform.shape[0]))

        label = item["label"]
        return waveform, label


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained ECAPA-TDNN
    print("Loading ECAPA-TDNN model from SpeechBrain...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=MODEL_DIR,
        run_opts={"device": str(device)}
    )

    # Count parameters
    # The main model is the embedding model (encoder)
    total_params, trainable_params = 0, 0
    for module_name in ['mods', 'modules']:
        if hasattr(classifier, module_name):
            mod = getattr(classifier, module_name)
            for name, param in mod.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load SUPERB SI dataset
    print("\nLoading SUPERB SI dataset (eval split)...")
    dataset = load_dataset("s3prl/superb", "si", split="test", trust_remote_code=True)

    print(f"Eval set size: {len(dataset)}")

    # Get number of unique speakers
    labels = dataset["label"]
    num_speakers = len(set(labels))
    print(f"Number of speakers: {num_speakers}")

    # Create dataset and dataloader
    eval_dataset = SuperbSIDataset(dataset, target_sr=16000)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Compute GFLOPs with a sample input
    print("\nComputing GFLOPs...")
    sample_waveform = torch.randn(1, 80000).to(device)  # 5 seconds at 16kHz
    try:
        from thop import profile, clever_format
        # We need to profile the actual embedding model
        model_to_profile = classifier.mods.embedding_model if hasattr(classifier.mods, 'embedding_model') else None

        if model_to_profile is not None:
            model_to_profile.eval()
            # ECAPA-TDNN takes features, not raw audio. Need to get features first
            with torch.no_grad():
                feats = classifier.mods.compute_features(sample_waveform)
                feats = classifier.mods.mean_var_norm(feats, torch.ones(1).to(device))
            flops, params = profile(model_to_profile, inputs=(feats,), verbose=False)
            gflops = flops / 1e9
        else:
            gflops = -1.0
            print("Could not find embedding model for profiling")
    except Exception as e:
        print(f"GFLOPs computation error: {e}")
        gflops = -1.0

    print(f"Computational cost: {gflops:.4f} GFLOPs")

    # Evaluate accuracy
    print("\nEvaluating on test set...")
    correct = 0
    total = 0

    classifier.mods.eval()
    with torch.no_grad():
        for i, (waveform, label) in enumerate(eval_loader):
            waveform = waveform.to(device)

            # Get prediction
            output = classifier.classify_batch(waveform)
            # output is (posterior, score, index, text_lab)
            pred_idx = output[3]  # predicted label index

            # Compare
            # The classifier returns string labels, we need to map
            # Actually classify_batch returns: (out_prob, score, index, text_lab)
            pred_index = output[2].squeeze().item()

            if pred_index == label.item():
                correct += 1
            total += 1

            if (i + 1) % 500 == 0:
                print(f"  Processed {i+1}/{len(eval_loader)} | Running Acc: {correct/total:.4f}")

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
        "trainable_params": trainable_params,
        "num_speakers": num_speakers,
        "eval_size": len(dataset),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
