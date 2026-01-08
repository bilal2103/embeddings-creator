from pydub import AudioSegment
from DiarizationService import Diarization
from pyannote.core import Segment
from pyannote.audio import Inference, Model
import numpy as np
import json
from typing import Dict
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
diarization = Diarization()
model = Model.from_pretrained("pyannote/embedding", 
                              use_auth_token=os.getenv("HF_EMBEDDING_TOKEN"))
inference = Inference(model, window="whole")
def GetEmbedding(audioFiles: Dict[str, str], speakerName: str):
    os.makedirs("cleanedFiles", exist_ok=True)
    
    
    embeddings = []
    for audioFile, speakerToExtract in audioFiles.items():
        cleanedAudioPath = f"cleanedFiles/{audioFile.split('/')[-1]}"
        audio = AudioSegment.from_wav(audioFile)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(cleanedAudioPath, format="wav")
        diarizationResult = diarization.diarize(cleanedAudioPath)
        diarizationSegments = [_Segment(segment) for segment in list(diarizationResult.itertracks(yield_label=True))]
        for segment in diarizationSegments:
            if segment["stop"] - segment["start"] > 2.0 and segment["speaker"] == speakerToExtract:
                embedding = inference.crop(cleanedAudioPath, Segment(segment["start"], segment["stop"]))
                embeddings.append(embedding)
    embeddingVector = np.mean(embeddings, axis=0)
    with open(f"{speakerName}_cleaned_embedding.json", "w") as f:
        json.dump(embeddingVector.tolist(), f)

def ExtractSpeakerAudios(speakerName):
    if speakerName == "fatima":
        audioFiles = ["Urdu/4.wav", "Urdu/20250602-105000-0503150168-IN-seder.wav", "Urdu/20250611-090346-0572434213-IN-seder.wav", "Urdu/20250612-094020-0571591930-IN-seder.wav", "Urdu/20250612-114342-0568150871-IN-seder.wav", "English/1_f.wav", "English/2_f.wav", "English/3_f.wav", "English/4_f.wav", "English/5_f.wav"]
    elif speakerName == "abdullah":
        audioFiles = ["Urdu/1.wav", "Urdu/3.wav", "Urdu/5.wav", "Urdu/6.wav", "Urdu/20250601-154622-0594733981-IN-seder.wav", "English/1_a.wav", "English/2_a.wav", "English/3_a.wav", "English/4_a.wav", "English/5_a.wav"]
    
    embeddings = []
    for audioFile in audioFiles:
        diarization_result = diarization.diarize(audioFile)
        diarizationSegments = [_Segment(segment) for segment in list(diarization_result.itertracks(yield_label=True))]
        for segment in diarizationSegments:
            if segment["stop"] - segment["start"] > 2.0:
                excerpt = Segment(segment["start"], segment["stop"])
                embedding = inference.crop(audioFile, excerpt)
                embeddings.append(embedding)
    embeddingVector = np.mean(embeddings, axis=0)
    with open(f"{speakerName}_embedding.json", "w") as f:
        json.dump(embeddingVector.tolist(), f)
    return embeddings

def ComputeSimilarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

if __name__ == "__main__":
    # audioFiles = {
    #     "Urdu/1.wav": "SPEAKER_01",
    #     "Urdu/3.wav": "SPEAKER_01",
    #     "Urdu/20250601-154622-0594733981-IN-seder.wav": "SPEAKER_00"
    # }
    # GetEmbedding(audioFiles, "abdullah")
    embedding1 = json.load(open("abdullah_cleaned_embedding.json"))
    embedding2 = json.load(open("abdullah_embedding.json"))
    print(ComputeSimilarity(np.array(embedding1), np.array(embedding2)))
    