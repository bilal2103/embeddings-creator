from pathlib import Path
import json
import numpy as np

def test_embedding_quality(embedding_file, embedding_name):
    """Test how many problematic files we get with a given embedding"""
    folder = Path("Diarizations")
    allFiles = list(folder.glob("*.json"))
    problematic, fine = 0, 0
    max_scores = []
    
    # Load embeddings
    with open("fatima_embedding.json", "r") as f:
        fatimaEmbedding = np.array(json.load(f))
    with open(embedding_file, "r") as f:
        abdullahEmbedding = np.array(json.load(f))
    
    for file in allFiles:
        with open(file, "r") as f:
            data = json.load(f)
            max_f, max_a = 0, 0
            
            for speaker, speakerEmbedding in data.items():
                speakerEmbedding = np.array(speakerEmbedding)
                cosineDistance_fatima = np.dot(speakerEmbedding, fatimaEmbedding) / (np.linalg.norm(speakerEmbedding) * np.linalg.norm(fatimaEmbedding))
                cosineDistance_abdullah = np.dot(speakerEmbedding, abdullahEmbedding) / (np.linalg.norm(speakerEmbedding) * np.linalg.norm(abdullahEmbedding))
                max_f = max(max_f, cosineDistance_fatima)
                max_a = max(max_a, cosineDistance_abdullah)
                
            max_scores.append(max_a)
            if max_f < 0.80 and max_a < 0.80:
                problematic += 1
            else:
                fine += 1
    
    avg_max_score = np.mean(max_scores)
    return problematic, fine, avg_max_score

# Test all embedding versions
embedding_versions = [
    ("abdullah_embedding.json", "Original Clean Abdullah"),
    ("abdullah_weighted_80_20.json", "80% Clean, 20% Noisy"),
    ("abdullah_weighted_70_30.json", "70% Clean, 30% Noisy"), 
    ("abdullah_weighted_60_40.json", "60% Clean, 40% Noisy"),
    ("abdullah_averaged_with_speaker_00.json", "50% Clean, 50% Noisy")
]

print("=== Testing Different Abdullah Embedding Versions ===")
print("Threshold: 0.80 for both Fatima and Abdullah")
print("=" * 60)

results = []
for embedding_file, name in embedding_versions:
    try:
        problematic, fine, avg_score = test_embedding_quality(embedding_file, name)
        results.append((name, problematic, fine, avg_score))
        print(f"{name:25} | Problematic: {problematic:2d} | Fine: {fine:2d} | Avg Max Score: {avg_score:.4f}")
    except FileNotFoundError:
        print(f"{name:25} | File not found - run tempFile.py first")

print("=" * 60)
print("Analysis:")
print("- Lower 'Problematic' count is better")  
print("- Higher 'Avg Max Score' indicates better similarity detection")
print("- Original had 4 problematic, 50-50 average had 18 problematic")
print("- Weighted averages should reduce problematic count while maintaining good scores")
