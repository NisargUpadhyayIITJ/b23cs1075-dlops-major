"""
analysis.py - Task 5: Final Analysis for ECAPA-TDNN
Compares baseline vs PTQ vs best QAT model
"""

import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    # Load all results
    with open(os.path.join(SCRIPT_DIR, "baseline_results.json"), "r") as f:
        baseline = json.load(f)
    with open(os.path.join(SCRIPT_DIR, "ptq_results.json"), "r") as f:
        ptq = json.load(f)
    with open(os.path.join(SCRIPT_DIR, "qat_results.json"), "r") as f:
        qat = json.load(f)

    print("=" * 70)
    print("FINAL ANALYSIS: ECAPA-TDNN Model Optimization")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'Baseline':>12} {'PTQ (INT8)':>12} {'QAT (Best)':>12}")
    print("-" * 70)
    print(f"{'Accuracy':<30} {baseline['accuracy']:>12.4f} {ptq['ptq_accuracy']:>12.4f} {qat['best_accuracy']:>12.4f}")
    print(f"{'GFLOPs':<30} {baseline['gflops']:>12.4f} {ptq['ptq_gflops']:>12.4f} {qat['qat_gflops']:>12.4f}")
    print(f"{'Acc Diff vs Baseline':<30} {'--':>12} {ptq['accuracy_diff']:>+12.4f} {qat['accuracy_diff']:>+12.4f}")
    print(f"{'GFLOPs Saved vs Baseline':<30} {'--':>12} {ptq['gflops_reduction']:>12.4f} {qat['gflops_saved']:>12.4f}")

    print(f"\n{'='*70}")
    print(f"SUMMARY FOR README")
    print(f"{'='*70}")
    print(f"\nTask 1:")
    print(f"  Baseline Accuracy: {baseline['accuracy']:.4f}")
    print(f"  Baseline GFLOPs:   {baseline['gflops']:.4f}")
    print(f"\nTask 2:")
    print(f"  PTQ GFLOPs:        {ptq['ptq_gflops']:.4f}")
    print(f"  GFLOPs Impact:     {ptq['gflops_reduction']:.4f} reduction")
    print(f"\nTask 3:")
    print(f"  PTQ Accuracy:      {ptq['ptq_accuracy']:.4f} ({'increased' if ptq['accuracy_diff'] > 0 else 'decreased'})")
    print(f"\nTask 4:")
    print(f"  Best Hyperparams:  {json.dumps(qat['best_params'])}")
    print(f"  Best QAT Accuracy: {qat['best_accuracy']:.4f}")
    print(f"  QAT GFLOPs:        {qat['qat_gflops']:.4f}")
    print(f"\nTask 5:")
    print(f"  Total Accuracy Diff (QAT vs Baseline): {qat['accuracy_diff']:+.4f}")
    print(f"  Total GFLOPs Saved:                    {qat['gflops_saved']:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
