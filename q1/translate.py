"""
translate.py - Bengali to English Translation using Helsinki-NLP/opus-mt-bn-en
Uses HuggingFace Transformers to load the pretrained model and translate
Bengali sentences from input.txt, saving results to output.txt
"""

import os
from transformers import MarianMTModel, MarianTokenizer

def load_sentences(filepath):
    """Load sentences from file, skipping comments and empty lines."""
    sentences = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comment lines
            if line and not line.startswith("#"):
                sentences.append(line)
    return sentences

def translate_sentences(sentences, model, tokenizer, batch_size=4):
    """Translate a list of sentences from Bengali to English."""
    translations = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        # Tokenize
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # Generate translations
        translated = model.generate(**inputs, max_length=512, num_beams=4)
        # Decode
        for t in translated:
            decoded = tokenizer.decode(t, skip_special_tokens=True)
            translations.append(decoded)
        print(f"Translated {min(i + batch_size, len(sentences))}/{len(sentences)} sentences")
    return translations

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "input.txt")
    output_file = os.path.join(script_dir, "output.txt")

    # Load model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-bn-en"
    print(f"Loading model: {model_name}")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    print("Model loaded successfully!")

    # Load Bengali sentences
    sentences = load_sentences(input_file)
    print(f"Loaded {len(sentences)} sentences from input.txt")

    # Translate
    print("Translating...")
    translations = translate_sentences(sentences, model, tokenizer)

    # Save output
    with open(output_file, "w", encoding="utf-8") as f:
        for t in translations:
            f.write(t + "\n")

    print(f"\nTranslations saved to {output_file}")
    print(f"\nFirst translated statement:")
    print(f"  {translations[0]}")

if __name__ == "__main__":
    main()
