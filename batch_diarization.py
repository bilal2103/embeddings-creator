import os
import json
import glob
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from DiarizationService import Diarization
from pyannote.core import Segment
from pyannote.audio import Inference, Model
import numpy as np
import json
from pydub import AudioSegment

def _Segment(segment):
    turn, _, speaker = segment
    return {
        "start": turn.start,
        "stop": turn.end,
        "speaker": speaker
    }
class BatchDiarizer:
    def __init__(self):
        print("🎵 Initializing Batch Diarizer...")
        self.diarizer = Diarization()
        print("✅ Batch Diarizer ready!")
        self.model = Model.from_pretrained("pyannote/embedding", 
                              use_auth_token=os.getenv("HF_EMBEDDING_TOKEN"))
        self.inference = Inference(self.model, window="whole")
        self.abdullahEmbedding = json.load(open("abdullah_embedding.json"))
        self.fatimaEmbedding = json.load(open("fatima_embedding.json"))
    
    def get_audio_files(self, folder_path: str) -> List[str]:
        folder = Path(folder_path)
        if not folder.exists():
            print(f"⚠️ Folder {folder_path} does not exist!")
            return []
        
        wav_files = list(folder.glob("*.wav"))
        print(f"📁 Found {len(wav_files)} audio files in {folder_path}")
        
        return [str(file) for file in wav_files]
    
    def process_single_file(self, audio_path: str):
        try:
            os.makedirs("Diarizations", exist_ok=True)
            print(f"\n🎯 Processing: {Path(audio_path).name}")
            cleanedAudioPath = f"cleanedFiles/{Path(audio_path).name}"
            audio = AudioSegment.from_wav(audio_path)
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            audio.export(cleanedAudioPath, format="wav")
            diarization_result = self.diarizer.diarize(cleanedAudioPath)
            diarizationSegments = [_Segment(segment) for segment in list(diarization_result.itertracks(yield_label=True))]
            speakerEmbeddings = self.GetSpeakerEmbeddings(diarizationSegments, cleanedAudioPath)
            toSave = {}
            for key, value in speakerEmbeddings.items():
                toSave[key] = value.tolist()
            with open(f"Diarizations/{Path(audio_path).name}.json", "w") as f:
                json.dump(toSave, f, indent=4)
            return speakerEmbeddings
            
        except Exception as e:
            print(f"❌ Error processing {Path(audio_path).name}: {str(e)}")
            raise e
    def GetSpeakerEmbeddings(self, diarizationSegments, audio) -> np.ndarray:
        embeddings = {}
        try:
            for segment in diarizationSegments:
                segment_duration = segment["stop"] - segment["start"]
                if segment_duration < 2.0:
                    print(f"Skipping short segment: {segment_duration:.3f}s for speaker {segment['speaker']}")
                    segment["embedding"] = None
                    continue
                excerpt = Segment(segment["start"], segment["stop"])
                embedding = self.inference.crop(audio, excerpt)
                if segment["speaker"] not in embeddings:
                    embeddings[segment["speaker"]] = []
                embeddings[segment["speaker"]].append(embedding)
            for key, value in embeddings.items():
                embeddings[key] = np.mean(value, axis=0)
            return embeddings
        except RuntimeError as e:
            print(f"Error processing segment {segment['start']:.2f}-{segment['stop']:.2f}s: {e}")
            segment["embedding"] = None
            raise
    def process_folder(self, folder_path: str, folder_name: str):
        print(f"\n📂 Processing {folder_name} folder: {folder_path}")
        
        audio_files = self.get_audio_files(folder_path)
        if not audio_files:
            print(f"⚠️ No audio files found in {folder_path}")
            return []
        
        for i, audio_file in enumerate(audio_files, 1):
            try:
                print(f"\n--- File {i}/{len(audio_files)} ---")
                speakerEmbeddings = self.process_single_file(audio_file)
                for key, value in speakerEmbeddings.items():
                    print(f"Similarity between speaker {key} and Abdullah: {np.dot(self.abdullahEmbedding, value) / (np.linalg.norm(self.abdullahEmbedding) * np.linalg.norm(value))}")
                    print(f"Similarity between speaker {key} and Fatima: {np.dot(self.fatimaEmbedding, value) / (np.linalg.norm(self.fatimaEmbedding) * np.linalg.norm(value))}")
            except Exception as e:
                print(f"❌ Error processing {Path(audio_file).name}: {str(e)}")
                continue
    def diarize_all_files(self):
        print("🚀 Starting batch diarization for all audio files...")
        print("=" * 60)
        
        self.process_folder("Urdu", "Urdu")
        self.process_folder("English", "English")
    
def main():
    try:
        batch_diarizer = BatchDiarizer()
        
        batch_diarizer.diarize_all_files()
        
        print("\n🎉 Batch diarization completed successfully!")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
