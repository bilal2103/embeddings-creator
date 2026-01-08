import os
import torch
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Callable
from dotenv import load_dotenv
from pyannote.audio import Pipeline, Model, Inference
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import Segment, Annotation
from pyannote.audio.core.io import AudioFile

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*")
warnings.filterwarnings("ignore", message=".*custom_fwd.*")
warnings.filterwarnings("ignore", message=".*ReproducibilityWarning.*")

class EmbeddingExtractor:
    def __init__(self, use_auth_token: Optional[str] = None):
        load_dotenv()
        
        if use_auth_token is None:
            use_auth_token = os.getenv("HF_TOKEN")
            
        self.use_auth_token = use_auth_token
        
        # Enable TF32 for better performance if supported
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except:
                pass
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize models
        self._init_models()
    
    def _init_models(self):
        print("Loading pyannote models...")
        
        # 1. Load the embedding model directly
        print("  - Loading embedding model...")
        self.embedding_model = Model.from_pretrained(
            "pyannote/embedding", 
            use_auth_token=self.use_auth_token
        ).to(self.device)
        
        # 2. Load the diarization pipeline
        print("  - Loading diarization pipeline...")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization", 
            use_auth_token=self.use_auth_token
        ).to(self.device)
        
        # 3. Create inference objects for different approaches
        print("  - Setting up inference objects...")
        self.embedding_inference_whole = Inference(self.embedding_model, window="whole")
        self.embedding_inference_sliding = Inference(
            self.embedding_model, 
            window="sliding", 
            duration=3.0, 
            step=1.0
        )
        
        print("✓ All models loaded successfully!")
    
    def extract_embeddings_from_diarization(
        self, 
        audio_path: str, 
        return_diarization: bool = True
    ) -> Dict:
        print(f"Extracting embeddings from diarization for: {audio_path}")
        
        # Storage for captured embeddings
        captured_data = {
            'embeddings': None,
            'segmentations': None,
            'binary_segmentations': None
        }
        
        def embedding_hook(name: str, *args, **kwargs):
            if name == "embeddings" and len(args) > 0 and args[0] is not None:
                captured_data['embeddings'] = args[0].copy()
            elif name == "segmentation" and len(args) > 0 and args[0] is not None:
                captured_data['segmentations'] = args[0]
            elif name == "binary_segmentations" and len(args) > 0 and args[0] is not None:
                captured_data['binary_segmentations'] = args[0]
        
        # Run diarization with hook
        diarization = self.diarization_pipeline(audio_path, hook=embedding_hook)
        
        result = {
            'embeddings': captured_data['embeddings'],
            'embedding_shape': captured_data['embeddings'].shape if captured_data['embeddings'] is not None else None,
            'embedding_info': {
                'description': 'Embeddings extracted during diarization process',
                'shape_meaning': '(num_chunks, num_speakers, embedding_dimension)',
                'chunk_duration': 2.0,  # Default pyannote chunk duration
                'chunk_step': 1.0,      # Default pyannote chunk step
            }
        }
        
        if return_diarization:
            result['diarization'] = diarization
            result['segments'] = [
                {
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker,
                    'duration': turn.duration
                }
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]
        
        return result
    
    def extract_speaker_embeddings(
        self, 
        audio_path: str, 
        min_speaker_time: float = 2.0,
        min_segment_duration: float = 1.0
    ) -> Dict:

        print(f"Extracting speaker-specific embeddings for: {audio_path}")
        
        # First get diarization results with embeddings
        diarization_result = self.extract_embeddings_from_diarization(audio_path)
        diarization = diarization_result['diarization']
        chunk_embeddings = diarization_result['embeddings']
        
        # Calculate speaker statistics
        speaker_stats = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {'total_time': 0.0, 'segments': []}
            speaker_stats[speaker]['total_time'] += turn.duration
            speaker_stats[speaker]['segments'].append((turn.start, turn.end))
        
        # Filter speakers by minimum speaking time
        valid_speakers = {
            speaker: stats for speaker, stats in speaker_stats.items()
            if stats['total_time'] >= min_speaker_time
        }
        
        print(f"Found {len(valid_speakers)} speakers with sufficient speaking time:")
        for speaker, stats in valid_speakers.items():
            print(f"  Speaker {speaker}: {stats['total_time']:.1f}s")
        
        # Extract embeddings for each valid speaker
        speaker_embeddings = {}
        for speaker, stats in valid_speakers.items():
            # Get embeddings from the speaker's segments
            speaker_segment_embeddings = []
            valid_segments = []
            
            for start, end in stats['segments']:
                segment_duration = end - start
                
                # Skip segments that are too short for the model
                if segment_duration < min_segment_duration:
                    print(f"  Skipping short segment for {speaker}: {segment_duration:.2f}s < {min_segment_duration}s")
                    continue
                
                try:
                    segment_embedding = self.embedding_inference_whole.crop(
                        audio_path, Segment(start, end)
                    )
                    speaker_segment_embeddings.append(segment_embedding)
                    valid_segments.append((start, end))
                except Exception as e:
                    print(f"  Error processing segment for {speaker} ({start:.2f}-{end:.2f}s): {str(e)}")
                    continue
            
            # Calculate representative embedding (mean) if we have valid segments
            if speaker_segment_embeddings:
                representative_embedding = np.mean(speaker_segment_embeddings, axis=0)
                speaker_embeddings[speaker] = {
                    'representative_embedding': representative_embedding,
                    'all_segment_embeddings': speaker_segment_embeddings,
                    'num_segments': len(speaker_segment_embeddings),
                    'num_valid_segments': len(valid_segments),
                    'total_speaking_time': stats['total_time'],
                    'valid_segments': valid_segments
                }
                print(f"  Successfully extracted {len(speaker_segment_embeddings)} embeddings for {speaker}")
            else:
                print(f"  No valid segments found for {speaker} (all segments too short or failed)")
        
        return {
            'speaker_embeddings': speaker_embeddings,
            'speaker_count': len(speaker_embeddings),
            'all_speakers': list(speaker_embeddings.keys()),
            'embedding_info': {
                'description': 'Representative embeddings for each detected speaker',
                'method': 'mean_of_segments',
                'min_speaker_time': min_speaker_time,
                'min_segment_duration': min_segment_duration
            }
        }
    
    def save_embeddings(
        self, 
        embeddings_data: Dict, 
        output_path: str, 
        format: str = "numpy"
    ):

        if format == "numpy":
            np.save(output_path, embeddings_data)
            print(f"Embeddings saved to {output_path}")
        elif format == "json":
            # Convert numpy arrays to lists for JSON serialization
            import json
            json_data = self._convert_numpy_to_lists(embeddings_data)
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"Embeddings saved to {output_path}")
        else:
            raise ValueError("Format must be 'numpy' or 'json'")
    
    def _convert_numpy_to_lists(self, data):
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {key: self._convert_numpy_to_lists(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_numpy_to_lists(item) for item in data]
        else:
            return data


if __name__ == "__main__":
    extractor = EmbeddingExtractor()
    
    audio_file = "Urdu/4.wav"
    
    if os.path.exists(audio_file):
        print("=" * 60)
        print("DEMONSTRATION: Extracting embeddings from diarization pipeline")
        print("=" * 60)
        
        print("\n1. Extracting embeddings from diarization pipeline")
        diarization_embeddings = extractor.extract_embeddings_from_diarization(audio_file)
        print(f"   Embedding shape: {diarization_embeddings['embedding_shape']}")
        
        print("\n2. Extracting speaker-specific embeddings:")
        speaker_embeddings = extractor.extract_speaker_embeddings(audio_file)
        print(f"   Number of speakers: {speaker_embeddings['speaker_count']}")
        #print(speaker_embeddings)

        print("\n" + "=" * 60)
        print("All embedding extraction methods demonstrated!")
        print("=" * 60)
    else:
        print(f"Audio file not found: {audio_file}")
        print("Please provide a valid audio file path to run the demonstration.")
