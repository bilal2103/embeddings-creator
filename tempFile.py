from DiarizationService import Diarization
import json
import os
import numpy as np

with open("Diarizations/4_a.wav.json", "r") as f:
    data = np.array(json.load(f)["SPEAKER_00"])

abdullah_vector = np.array(json.load(open("abdullah_embedding.json", "r")))

cosine_distance = np.dot(data, abdullah_vector) / (np.linalg.norm(data) * np.linalg.norm(abdullah_vector))

print("Cosine distance before: ", cosine_distance)

# Try different weighted averages - giving more weight to the clean Abdullah vector
# Weight ratios: [clean_weight, noisy_weight] should sum to 1.0

# Option 1: 80% clean, 20% noisy (conservative approach)
weight_clean_80 = 0.8
weight_noisy_20 = 0.2
updated_abdullah_vector_80_20 = (weight_clean_80 * abdullah_vector + weight_noisy_20 * data)

# Option 2: 70% clean, 30% noisy (moderate approach)  
weight_clean_70 = 0.7
weight_noisy_30 = 0.3
updated_abdullah_vector_70_30 = (weight_clean_70 * abdullah_vector + weight_noisy_30 * data)

# Option 3: 60% clean, 40% noisy (more balanced)
weight_clean_60 = 0.6
weight_noisy_40 = 0.4
updated_abdullah_vector_60_40 = (weight_clean_60 * abdullah_vector + weight_noisy_40 * data)

# For backward compatibility, keep the 50-50 average
updated_abdullah_vector = (abdullah_vector + data) / 2

# Calculate cosine distances for all weighted versions
cosine_distance_80_20 = np.dot(data, updated_abdullah_vector_80_20) / (np.linalg.norm(data) * np.linalg.norm(updated_abdullah_vector_80_20))
cosine_distance_70_30 = np.dot(data, updated_abdullah_vector_70_30) / (np.linalg.norm(data) * np.linalg.norm(updated_abdullah_vector_70_30))
cosine_distance_60_40 = np.dot(data, updated_abdullah_vector_60_40) / (np.linalg.norm(data) * np.linalg.norm(updated_abdullah_vector_60_40))
cosine_distance_50_50 = np.dot(data, updated_abdullah_vector) / (np.linalg.norm(data) * np.linalg.norm(updated_abdullah_vector))

print("=== Cosine Distance Results ===")
print(f"Original (clean Abdullah only): {cosine_distance:.4f}")
print(f"80% clean, 20% noisy: {cosine_distance_80_20:.4f}")
print(f"70% clean, 30% noisy: {cosine_distance_70_30:.4f}")
print(f"60% clean, 40% noisy: {cosine_distance_60_40:.4f}")
print(f"50% clean, 50% noisy: {cosine_distance_50_50:.4f}")

print("\n=== Vector Shapes ===")
print("Original Abdullah vector shape: ", abdullah_vector.shape)
print("SPEAKER_00 data shape: ", data.shape)

# Save all weighted versions for testing
with open("abdullah_weighted_80_20.json", "w") as f:
    json.dump(updated_abdullah_vector_80_20.tolist(), f)

with open("abdullah_weighted_70_30.json", "w") as f:
    json.dump(updated_abdullah_vector_70_30.tolist(), f)

with open("abdullah_weighted_60_40.json", "w") as f:
    json.dump(updated_abdullah_vector_60_40.tolist(), f)

# Keep the 50-50 version for backward compatibility
with open("abdullah_averaged_with_speaker_00.json", "w") as f:
    json.dump(updated_abdullah_vector.tolist(), f)

print("\n=== Files Saved ===")
print("abdullah_weighted_80_20.json - Conservative (80% clean)")
print("abdullah_weighted_70_30.json - Moderate (70% clean)")  
print("abdullah_weighted_60_40.json - Balanced (60% clean)")
print("abdullah_averaged_with_speaker_00.json - Equal weights (50% clean)")

print("\nRecommendation: Try the 80-20 or 70-30 weighted versions in your checkDistances.py")








