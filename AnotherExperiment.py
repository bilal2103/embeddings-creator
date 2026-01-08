from pyannote.audio import Inference
from DiarizationService import Diarization
from pyannote.audio import Model
from pyannote.core import Segment
import json
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

def _Segment(segment):
    turn, _, speaker = segment
    return {
        "start": turn.start,
        "stop": turn.end,
        "speaker": speaker
    }

def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            else:
                return obj
model = Model.from_pretrained("pyannote/embedding", 
                              use_auth_token=os.getenv("HF_EMBEDDING_TOKEN"))
inference = Inference(model, window="whole")

diarization = Diarization()

audio = "Urdu/20250703-091106-0563956028-IN-seder.wav"

diarization_result = diarization.diarize(audio)
diarizationSegments = [_Segment(segment) for segment in list(diarization_result.itertracks(yield_label=True))]

for segment in diarizationSegments:
    # Skip segments that are too short (less than 0.5 seconds)
    segment_duration = segment["stop"] - segment["start"]
    if segment_duration < 0.5:
        print(f"Skipping short segment: {segment_duration:.3f}s for speaker {segment['speaker']}")
        segment["embedding"] = None
        continue
    
    try:
        excerpt = Segment(segment["start"], segment["stop"])
        embedding = inference.crop(audio, excerpt)
        segment["embedding"] = convert_to_serializable(embedding)
    except RuntimeError as e:
        print(f"Error processing segment {segment['start']:.2f}-{segment['stop']:.2f}s: {e}")
        segment["embedding"] = None

with open("anotherAttemptResult.json", "w") as f:
    json.dump(diarizationSegments, f)

speaker_embeddings = {}
with open("anotherAttemptResult.json", "r") as f:
    diarizationSegments = json.load(f)

for segment in diarizationSegments:
    if segment["embedding"] is not None and segment["stop"] - segment["start"] > 2.0:
        if segment["speaker"] not in speaker_embeddings:
            speaker_embeddings[segment["speaker"]] = []
        speaker_embeddings[segment["speaker"]].append(segment["embedding"])

for key, value in speaker_embeddings.items():
    # Calculate mean embedding for each speaker
    embeddings_array = np.array(value)
    mean_embedding = np.mean(embeddings_array, axis=0)
    speaker_embeddings[key] = mean_embedding
    print(f"Shape of {key} embedding: {mean_embedding.shape}")
    
with open("toBeConcatenated.json", "w") as f:
    for key, value in speaker_embeddings.items():
        speaker_embeddings[key] = value.tolist()
    json.dump(speaker_embeddings, f)

with open("concatenatedEmbeddings.json", "r") as f:
    concatenatedEmbeddings = json.load(f)

abdullah_embedding = np.array(concatenatedEmbeddings["ABDULLAH"])

for key, value in speaker_embeddings.items():
    cosine_similarity = np.dot(abdullah_embedding, value) / (np.linalg.norm(abdullah_embedding) * np.linalg.norm(value))
    print(f"Cosine similarity between ABDULLAH and {key}: {cosine_similarity}")










