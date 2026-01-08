import os
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Callable, List, Tuple
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pathlib import Path

# Try to import visualization libraries
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠ sklearn not available. Install with: pip install scikit-learn")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠ UMAP not available. Install with: pip install umap-learn")

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*")
warnings.filterwarnings("ignore", message=".*custom_fwd.*")
warnings.filterwarnings("ignore", message=".*ReproducibilityWarning.*")

class Diarization:
    def __init__(self):
        load_dotenv()
        os.environ['SPEECHBRAIN_CACHE_STRATEGY'] = 'copy'
        
        # Enable TF32 for better performance if supported (suppresses the warning)
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except:
                pass
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        print("Loading diarization model... (this may take a moment)")
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=os.getenv("HF_TOKEN"))
        self.pipeline = self.pipeline.to(device)
        print("✓ Diarization model loaded successfully!")

    def diarize(self, audio_path: str) -> str:
        print(f"Starting diarization for: {audio_path}")
        
        diarization = self.pipeline(audio_path)
        
        print("✓ Diarization completed!")
        return diarization