import json
import numpy as np
with open("concatenatedEmbeddings.json", "r") as f:
    existingEmbeddings = json.load(f)

with open("toBeConcatenated.json", "r") as f:
    toConcatenate = json.load(f)

for key, value in toConcatenate.items():
    if key in existingEmbeddings:
        # Both are already averaged embeddings, so average them
        vec1 = np.array(existingEmbeddings[key])
        vec2 = np.array(value)
        average = np.mean([vec1, vec2], axis=0)
        existingEmbeddings[key] = average.tolist()
        print(f"Updated average for speaker {key}")
    else:
        # New speaker, just add their embedding
        existingEmbeddings[key] = value
        print(f"Added new speaker {key}")

with open("concatenatedEmbeddings.json", "w") as f:
    json.dump(existingEmbeddings, f)

