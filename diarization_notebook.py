import torch
from pyannote.audio import Audio
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import Segment
import json
from diarization_post_processor import DiarizationPostProcessor
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

class DiarizationNotebook:
    def __init__(self):
        self.diarization = SpeakerDiarization(
            clustering="AgglomerativeClustering",
            segmentation_batch_size=32,
            embedding_batch_size=32,
            embedding_exclude_overlap=True,
            use_auth_token=os.getenv("HF_EMBEDDING_TOKEN")
        )
        self.diarization_post = DiarizationPostProcessor()
        self.diarization.instantiate({
            "clustering": {
                "method": "centroid",
                "min_cluster_size": 15,
                "threshold": 0.7153814381597874,
            },
            "segmentation": {
                "min_duration_off": 0.5817029604921046,
                "threshold": 0.4442333667381752,
            },
        })
    def run_diarization(self, audio_path):
        closure = {'embeddings': None}

        def hook(name, *args, **kwargs):
            if name == "embeddings" and len(args) > 0 and args[0] is not None:
                closure['embeddings'] = args[0]

        print("Running speaker diarization...")
        diarization = self.diarization(audio_path, hook=hook)
        
        # Convert embeddings to JSON-serializable format
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            else:
                return obj
        
        embeddings = {
            'data': convert_to_serializable(closure['embeddings']),
            'embedding_info': {
                'chunk_duration': 2.0,  # Default pyannote chunk duration
                'chunk_step': 1.0,      # Default pyannote chunk step
            }
        }
        with open("embedding_from_hook.json", "w") as f:
            json.dump(embeddings, f)
        skipped_more_than_one_speaker = 0
        skipped_nan = 0
        processed = 0
        for i, chunk in enumerate(embeddings['data']):
            # chunk shape: (local_num_speakers, dimension)
            speakers = []
            for speaker_embedding in chunk:
                if not np.all(np.isnan(speaker_embedding)):
                    speakers.append(speaker_embedding)
                else:
                    skipped_nan += 1
                    continue
            if len(speakers) != 1:
                print(f"Found more than one speaker in chunk {i}")
                skipped_more_than_one_speaker += 1
                continue
            processed += 1
        print(f"Skipped {skipped_more_than_one_speaker} chunks with more than one speaker")
        print(f"Skipped {skipped_nan} chunks with nan")
        print(f"Processed {processed} chunks")
        #return self.diarization_post.process(diarization, embeddings)

if __name__ == "__main__":
    diarization_notebook = DiarizationNotebook()
    print(diarization_notebook.run_diarization("Urdu/20250602-105000-0503150168-IN-seder.wav"))