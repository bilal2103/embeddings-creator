from pathlib import Path
import json
import numpy as np

folder = Path("Diarizations")
allFiles = list(folder.glob("*.json"))
problematic, fine = 0, 0
for file in allFiles:
    with open("fatima_embedding.json", "r") as f:
        fatimaEmbedding = np.array(json.load(f))
    with open("abdullah_weighted_70_30.json", "r") as f:
        abdullahEmbedding = np.array(json.load(f))
    
    with open(file, "r") as f:
        data = json.load(f)
        max_f, max_a = 0, 0
        distances_f, distances_a = {}, {}
        for speaker, speakerEmbedding in data.items():
            speakerEmbedding = np.array(speakerEmbedding)
            cosineDistance_fatima = np.dot(speakerEmbedding, fatimaEmbedding) / (np.linalg.norm(speakerEmbedding) * np.linalg.norm(fatimaEmbedding))
            cosineDistance_abdullah = np.dot(speakerEmbedding, abdullahEmbedding) / (np.linalg.norm(speakerEmbedding) * np.linalg.norm(abdullahEmbedding))
            max_f = max(max_f, cosineDistance_fatima)
            max_a = max(max_a, cosineDistance_abdullah)
            distances_f[speaker] = cosineDistance_fatima.item()
            distances_a[speaker] = cosineDistance_abdullah.item()
            
        if max_f < 0.80 and max_a < 0.80:
            print("Gotta look into this file: ", file.name)
            print("Fatima: ", distances_f)
            print("Abdullah: ", distances_a)
            print("--------------------------------")
            problematic += 1
        else:
            fine += 1

print("Problematic: ", problematic)
print("Fine: ", fine)
