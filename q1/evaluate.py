"""
evaluate.py - Evaluate translation quality using sacrebleu BLEU score
Compares generated output.txt against reference.txt
"""

import os
import sacrebleu

def load_lines(filepath):
    """Load non-empty, non-comment lines from a file."""
    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "output.txt")
    reference_file = os.path.join(script_dir, "reference.txt")

    # Load generated translations and references
    hypotheses = load_lines(output_file)
    references = load_lines(reference_file)

    print(f"Loaded {len(hypotheses)} hypotheses and {len(references)} references")

    # Ensure same number of lines
    min_len = min(len(hypotheses), len(references))
    hypotheses = hypotheses[:min_len]
    references = references[:min_len]

    # Compute BLEU score using sacrebleu
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])

    print(f"\n{'='*60}")
    print(f"BLEU Score Evaluation Results")
    print(f"{'='*60}")
    print(f"BLEU Score: {bleu.score:.2f}")
    print(f"Details:    {bleu}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
