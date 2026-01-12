#!/usr/bin/env python
"""
Test embedding distances for paraphrased memories with different embedding models.
This helps determine appropriate deduplication thresholds for each model.
"""

import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine distance (1 - cosine_similarity)."""
    similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return 1 - similarity


# Test memories - paraphrases of the same fact
PARAPHRASED_MEMORIES = [
    "User likes coffee, flat white usually",
    "They are a coffee enthusiast, favorite coffee is flatwhite",
    "User loves coffee, especially flat white",
    "The user prefers flat white as their go-to coffee drink",
]

# Distinct memories - should NOT be merged
DISTINCT_MEMORIES = [
    "User likes coffee, flat white usually",
    "User prefers tea over coffee",
    "User has a dog named Max",
]

# Models to test
MODELS = [
    "all-MiniLM-L6-v2",  # 384 dims, fast
    "all-mpnet-base-v2",  # 768 dims, better quality
    "paraphrase-MiniLM-L6-v2",  # 384 dims, optimized for paraphrase detection
]


def test_model(model_name: str):
    """Test a single model and report distances."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print("="*60)
    
    model = SentenceTransformer(model_name)
    
    # Test paraphrased memories
    print("\n--- Paraphrased Memories (SHOULD be merged) ---")
    embeddings = model.encode(PARAPHRASED_MEMORIES)
    
    distances = []
    for i in range(len(PARAPHRASED_MEMORIES)):
        for j in range(i + 1, len(PARAPHRASED_MEMORIES)):
            dist = cosine_distance(embeddings[i], embeddings[j])
            distances.append(dist)
            print(f"  [{i+1}] vs [{j+1}]: {dist:.4f}")
    
    avg_paraphrase = np.mean(distances)
    max_paraphrase = np.max(distances)
    print(f"\n  Average distance: {avg_paraphrase:.4f}")
    print(f"  Max distance: {max_paraphrase:.4f}")
    
    # Test distinct memories
    print("\n--- Distinct Memories (should NOT be merged) ---")
    embeddings = model.encode(DISTINCT_MEMORIES)
    
    distances = []
    for i in range(len(DISTINCT_MEMORIES)):
        for j in range(i + 1, len(DISTINCT_MEMORIES)):
            dist = cosine_distance(embeddings[i], embeddings[j])
            distances.append(dist)
            print(f"  [{i+1}] vs [{j+1}]: {dist:.4f}")
    
    min_distinct = np.min(distances)
    print(f"\n  Min distance: {min_distinct:.4f}")
    
    # Recommend threshold
    print("\n--- Recommendation ---")
    if max_paraphrase < min_distinct:
        recommended = (max_paraphrase + min_distinct) / 2
        print(f"  Recommended threshold: {recommended:.2f}")
        print(f"  (between max paraphrase {max_paraphrase:.4f} and min distinct {min_distinct:.4f})")
    else:
        print(f"  WARNING: Overlap detected!")
        print(f"  Max paraphrase ({max_paraphrase:.4f}) >= Min distinct ({min_distinct:.4f})")
        print(f"  Consider using a model better suited for paraphrase detection")


def main():
    print("Testing embedding distances for deduplication threshold calibration")
    print("="*60)
    
    for model_name in MODELS:
        try:
            test_model(model_name)
        except Exception as e:
            print(f"\nError testing {model_name}: {e}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
Key findings:
- Different embedding models have different distance distributions
- The deduplication threshold should be calibrated per model
- paraphrase-MiniLM-L6-v2 is specifically trained for paraphrase detection
- A threshold that works for OpenAI embeddings may not work for others
""")


if __name__ == "__main__":
    main()

