import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
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

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠ Plotly not available. Install with: pip install plotly")

from DiarizationService import Diarization


class EmbeddingVisualizer:
    """
    Visualize pyannote embedding chunks in 2D space using various dimensionality reduction techniques
    """
    
    def __init__(self):
        """Initialize the visualizer"""
        self.diarizer = Diarization()
        self.embeddings_data = None
        self.segments_data = None
        self.reduced_embeddings = {}
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
        
    def extract_and_prepare_embeddings(self, audio_path: str) -> Dict:
        """
        Extract embeddings and prepare them for visualization
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with processed embedding data
        """
        print(f"📊 Extracting embeddings from: {audio_path}")
        
        # Get embeddings from diarization
        results = self.diarizer.diarize_with_embeddings(audio_path)
        
        if 'embeddings' not in results:
            raise ValueError("No embeddings captured from diarization")
        
        embeddings_raw = results['embeddings']['data']  # Shape: (num_chunks, num_speakers, embedding_dim)
        segments = results['segments']
        
        # Process embeddings: flatten and filter valid ones
        chunk_embeddings = []
        chunk_metadata = []
        chunk_speakers = []
        chunk_times = []
        
        chunk_duration = results['embeddings']['chunk_duration']
        chunk_step = results['embeddings']['chunk_step']
        
        print(f"📦 Processing {embeddings_raw.shape[0]} chunks...")
        
        for chunk_idx, chunk in enumerate(embeddings_raw):
            chunk_start_time = chunk_idx * chunk_step
            chunk_end_time = chunk_start_time + chunk_duration
            
            for speaker_idx, embedding in enumerate(chunk):
                # Filter out invalid embeddings
                if not np.all(np.isnan(embedding)) and np.any(embedding != 0):
                    chunk_embeddings.append(embedding)
                    chunk_metadata.append({
                        'chunk_idx': chunk_idx,
                        'speaker_idx': speaker_idx,
                        'start_time': chunk_start_time,
                        'end_time': chunk_end_time,
                        'chunk_center': chunk_start_time + chunk_duration / 2
                    })
                    
                    # Try to match with actual speaker labels from segments
                    matched_speaker = self._match_chunk_to_speaker(
                        chunk_start_time, chunk_end_time, segments
                    )
                    chunk_speakers.append(matched_speaker or f"Unknown_{speaker_idx}")
                    chunk_times.append(chunk_start_time + chunk_duration / 2)
        
        chunk_embeddings = np.array(chunk_embeddings)
        
        print(f"✅ Found {len(chunk_embeddings)} valid embedding chunks")
        print(f"📏 Embedding dimension: {chunk_embeddings.shape[1]}")
        
        self.embeddings_data = {
            'embeddings': chunk_embeddings,
            'metadata': chunk_metadata,
            'speakers': chunk_speakers,
            'times': chunk_times,
            'segments': segments,
            'audio_path': audio_path,
            'original_shape': embeddings_raw.shape
        }
        
        return self.embeddings_data
    
    def _match_chunk_to_speaker(self, chunk_start: float, chunk_end: float, segments: List[Dict]) -> Optional[str]:
        """Match a chunk to the most overlapping speaker segment"""
        max_overlap = 0
        best_speaker = None
        
        for seg in segments:
            # Calculate overlap
            overlap_start = max(chunk_start, seg['start'])
            overlap_end = min(chunk_end, seg['end'])
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = seg['speaker']
        
        return best_speaker
    
    def reduce_dimensions(self, method: str = 'tsne', **kwargs) -> np.ndarray:
        """
        Reduce embedding dimensions to 2D using various methods
        
        Args:
            method: 'tsne', 'pca', 'umap', or 'all'
            **kwargs: Parameters for the reduction method
            
        Returns:
            2D coordinates of embeddings
        """
        if self.embeddings_data is None:
            raise ValueError("No embeddings data available. Run extract_and_prepare_embeddings first.")
        
        embeddings = self.embeddings_data['embeddings']
        
        print(f"🔄 Reducing dimensions using {method.upper()}...")
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        if method.lower() == 'pca':
            if not SKLEARN_AVAILABLE:
                raise ImportError("sklearn required for PCA")
            
            reducer = PCA(n_components=2, **kwargs)
            reduced = reducer.fit_transform(embeddings_scaled)
            
            print(f"✅ PCA explained variance: {reducer.explained_variance_ratio_.sum():.3f}")
            
        elif method.lower() == 'tsne':
            if not SKLEARN_AVAILABLE:
                raise ImportError("sklearn required for t-SNE")
            
            # Default t-SNE parameters optimized for embeddings
            tsne_params = {
                'n_components': 2,
                'perplexity': min(30, len(embeddings) // 4),
                'random_state': 42,
                'n_iter': 1000,
                'learning_rate': 'auto'
            }
            tsne_params.update(kwargs)
            
            reducer = TSNE(**tsne_params)
            reduced = reducer.fit_transform(embeddings_scaled)
            
            print(f"✅ t-SNE completed with perplexity={tsne_params['perplexity']}")
            
        elif method.lower() == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError("umap-learn required for UMAP")
            
            # Default UMAP parameters
            umap_params = {
                'n_components': 2,
                'n_neighbors': min(15, len(embeddings) // 3),
                'min_dist': 0.1,
                'random_state': 42
            }
            umap_params.update(kwargs)
            
            reducer = umap.UMAP(**umap_params)
            reduced = reducer.fit_transform(embeddings_scaled)
            
            print(f"✅ UMAP completed with n_neighbors={umap_params['n_neighbors']}")
            
        elif method.lower() == 'all':
            # Compute all available methods
            results = {}
            for m in ['pca', 'tsne', 'umap']:
                try:
                    results[m] = self.reduce_dimensions(m, **kwargs)
                except ImportError as e:
                    print(f"⚠ Skipping {m.upper()}: {e}")
            return results
            
        else:
            raise ValueError("Method must be 'pca', 'tsne', 'umap', or 'all'")
        
        self.reduced_embeddings[method] = reduced
        return reduced
    
    def plot_2d_scatter(self, method: str = 'tsne', figsize: Tuple[int, int] = (12, 8), 
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a 2D scatter plot of reduced embeddings
        
        Args:
            method: Dimensionality reduction method to plot
            figsize: Figure size
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        if method not in self.reduced_embeddings:
            print(f"Computing {method.upper()} reduction...")
            self.reduce_dimensions(method)
        
        reduced = self.reduced_embeddings[method]
        speakers = self.embeddings_data['speakers']
        times = self.embeddings_data['times']
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Color by speaker
        unique_speakers = list(set(speakers))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_speakers)))
        
        for i, speaker in enumerate(unique_speakers):
            mask = np.array(speakers) == speaker
            ax1.scatter(reduced[mask, 0], reduced[mask, 1], 
                       c=[colors[i]], label=f'Speaker {speaker}', 
                       alpha=0.7, s=50)
        
        ax1.set_title(f'Embedding Clusters ({method.upper()}) - Colored by Speaker')
        ax1.set_xlabel(f'{method.upper()} Component 1')
        ax1.set_ylabel(f'{method.upper()} Component 2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Color by time
        scatter = ax2.scatter(reduced[:, 0], reduced[:, 1], 
                            c=times, cmap='viridis', alpha=0.7, s=50)
        ax2.set_title(f'Embedding Clusters ({method.upper()}) - Colored by Time')
        ax2.set_xlabel(f'{method.upper()} Component 1')
        ax2.set_ylabel(f'{method.upper()} Component 2')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar for time
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Time (seconds)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved to: {save_path}")
        
        return fig
    
    def plot_timeline_with_embeddings(self, method: str = 'tsne', 
                                    figsize: Tuple[int, int] = (15, 10),
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a timeline visualization showing speaker segments and embedding positions
        """
        if method not in self.reduced_embeddings:
            self.reduce_dimensions(method)
        
        reduced = self.reduced_embeddings[method]
        segments = self.embeddings_data['segments']
        speakers = self.embeddings_data['speakers']
        times = self.embeddings_data['times']
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3)
        
        # Main 2D embedding plot
        ax_main = fig.add_subplot(gs[0, :])
        unique_speakers = list(set(speakers))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_speakers)))
        color_map = {speaker: colors[i] for i, speaker in enumerate(unique_speakers)}
        
        for i, speaker in enumerate(unique_speakers):
            mask = np.array(speakers) == speaker
            ax_main.scatter(reduced[mask, 0], reduced[mask, 1], 
                           c=[colors[i]], label=f'Speaker {speaker}', 
                           alpha=0.7, s=60)
        
        ax_main.set_title(f'Speaker Embedding Clusters ({method.upper()})')
        ax_main.set_xlabel(f'{method.upper()} Component 1')
        ax_main.set_ylabel(f'{method.upper()} Component 2')
        ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_main.grid(True, alpha=0.3)
        
        # Timeline plot
        ax_timeline = fig.add_subplot(gs[1, :])
        
        for seg in segments:
            speaker = seg['speaker']
            color = color_map.get(speaker, 'gray')
            ax_timeline.barh(speaker, seg['duration'], left=seg['start'], 
                           color=color, alpha=0.7, height=0.6)
        
        ax_timeline.set_xlabel('Time (seconds)')
        ax_timeline.set_ylabel('Speaker')
        ax_timeline.set_title('Speaker Timeline')
        ax_timeline.grid(True, alpha=0.3)
        
        # Embedding timeline (showing how embeddings change over time)
        ax_emb_time = fig.add_subplot(gs[2, :])
        
        # Plot first component over time
        for speaker in unique_speakers:
            mask = np.array(speakers) == speaker
            speaker_times = np.array(times)[mask]
            speaker_coords = reduced[mask, 0]  # First component
            
            # Sort by time for proper line plotting
            sort_idx = np.argsort(speaker_times)
            ax_emb_time.plot(speaker_times[sort_idx], speaker_coords[sort_idx], 
                           'o-', color=color_map[speaker], alpha=0.7, 
                           label=f'Speaker {speaker}', markersize=4)
        
        ax_emb_time.set_xlabel('Time (seconds)')
        ax_emb_time.set_ylabel(f'{method.upper()} Component 1')
        ax_emb_time.set_title('Embedding Evolution Over Time')
        ax_emb_time.grid(True, alpha=0.3)
        ax_emb_time.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Timeline plot saved to: {save_path}")
        
        return fig
    
    def create_interactive_plot(self, method: str = 'tsne', 
                               save_path: Optional[str] = None) -> Optional[go.Figure]:
        """
        Create an interactive plot using Plotly
        """
        if not PLOTLY_AVAILABLE:
            print("⚠ Plotly not available for interactive plots")
            return None
        
        if method not in self.reduced_embeddings:
            self.reduce_dimensions(method)
        
        reduced = self.reduced_embeddings[method]
        speakers = self.embeddings_data['speakers']
        times = self.embeddings_data['times']
        metadata = self.embeddings_data['metadata']
        
        # Create hover text
        hover_text = []
        for i, meta in enumerate(metadata):
            text = (f"Speaker: {speakers[i]}<br>"
                   f"Time: {times[i]:.1f}s<br>"
                   f"Chunk: {meta['chunk_idx']}<br>"
                   f"Duration: {meta['start_time']:.1f}-{meta['end_time']:.1f}s")
            hover_text.append(text)
        
        # Create the plot
        fig = go.Figure()
        
        unique_speakers = list(set(speakers))
        colors = px.colors.qualitative.Set3[:len(unique_speakers)]
        
        for i, speaker in enumerate(unique_speakers):
            mask = np.array(speakers) == speaker
            
            fig.add_trace(go.Scatter(
                x=reduced[mask, 0],
                y=reduced[mask, 1],
                mode='markers',
                name=f'Speaker {speaker}',
                text=[hover_text[j] for j in np.where(mask)[0]],
                hovertemplate='%{text}<extra></extra>',
                marker=dict(
                    size=8,
                    color=colors[i % len(colors)],
                    opacity=0.7
                )
            ))
        
        fig.update_layout(
            title=f'Interactive Embedding Visualization ({method.upper()})',
            xaxis_title=f'{method.upper()} Component 1',
            yaxis_title=f'{method.upper()} Component 2',
            hovermode='closest',
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"💾 Interactive plot saved to: {save_path}")
        
        return fig
    
    def compare_methods(self, methods: List[str] = ['pca', 'tsne', 'umap'],
                       figsize: Tuple[int, int] = (18, 6),
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare different dimensionality reduction methods side by side
        """
        available_methods = []
        for method in methods:
            try:
                if method not in self.reduced_embeddings:
                    self.reduce_dimensions(method)
                available_methods.append(method)
            except ImportError as e:
                print(f"⚠ Skipping {method.upper()}: {e}")
        
        if not available_methods:
            raise ValueError("No dimensionality reduction methods available")
        
        fig, axes = plt.subplots(1, len(available_methods), figsize=figsize)
        if len(available_methods) == 1:
            axes = [axes]
        
        speakers = self.embeddings_data['speakers']
        unique_speakers = list(set(speakers))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_speakers)))
        
        for i, method in enumerate(available_methods):
            reduced = self.reduced_embeddings[method]
            
            for j, speaker in enumerate(unique_speakers):
                mask = np.array(speakers) == speaker
                axes[i].scatter(reduced[mask, 0], reduced[mask, 1], 
                               c=[colors[j]], label=f'Speaker {speaker}' if i == 0 else "", 
                               alpha=0.7, s=50)
            
            axes[i].set_title(f'{method.upper()}')
            axes[i].set_xlabel(f'{method.upper()} Component 1')
            axes[i].set_ylabel(f'{method.upper()} Component 2')
            axes[i].grid(True, alpha=0.3)
        
        # Add legend to the first subplot
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.suptitle('Comparison of Dimensionality Reduction Methods')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Comparison plot saved to: {save_path}")
        
        return fig
    
    def analyze_clusters(self, method: str = 'tsne', n_clusters: Optional[int] = None) -> Dict:
        """
        Perform cluster analysis on the reduced embeddings
        """
        if not SKLEARN_AVAILABLE:
            print("⚠ sklearn required for cluster analysis")
            return {}
        
        if method not in self.reduced_embeddings:
            self.reduce_dimensions(method)
        
        reduced = self.reduced_embeddings[method]
        speakers = self.embeddings_data['speakers']
        
        # Determine number of clusters
        if n_clusters is None:
            n_clusters = len(set(speakers))
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(reduced)
        
        # Analyze cluster purity (how well clusters match actual speakers)
        from collections import Counter
        
        cluster_analysis = {}
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            cluster_speakers = [speakers[i] for i in np.where(mask)[0]]
            speaker_counts = Counter(cluster_speakers)
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'size': int(np.sum(mask)),
                'dominant_speaker': speaker_counts.most_common(1)[0][0],
                'purity': speaker_counts.most_common(1)[0][1] / len(cluster_speakers),
                'speaker_distribution': dict(speaker_counts)
            }
        
        return {
            'method': method,
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels,
            'cluster_centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'analysis': cluster_analysis
        }
    
    def save_visualization_report(self, output_dir: str = "embedding_visualizations"):
        """
        Generate and save a comprehensive visualization report
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"📊 Generating visualization report in {output_path}/")
        
        if self.embeddings_data is None:
            print("⚠ No embedding data available")
            return
        
        # Generate all available visualizations
        methods = []
        for method in ['pca', 'tsne', 'umap']:
            try:
                self.reduce_dimensions(method)
                methods.append(method)
            except ImportError:
                continue
        
        if not methods:
            print("⚠ No dimensionality reduction methods available")
            return
        
        # 1. Comparison plot
        if len(methods) > 1:
            fig_comparison = self.compare_methods(methods)
            fig_comparison.savefig(output_path / "comparison.png", dpi=300, bbox_inches='tight')
            plt.close(fig_comparison)
        
        # 2. Individual scatter plots for each method
        for method in methods:
            fig_scatter = self.plot_2d_scatter(method)
            fig_scatter.savefig(output_path / f"scatter_{method}.png", dpi=300, bbox_inches='tight')
            plt.close(fig_scatter)
        
        # 3. Timeline visualization
        fig_timeline = self.plot_timeline_with_embeddings(methods[0])
        fig_timeline.savefig(output_path / "timeline.png", dpi=300, bbox_inches='tight')
        plt.close(fig_timeline)
        
        # 4. Interactive plot (if Plotly available)
        if PLOTLY_AVAILABLE:
            self.create_interactive_plot(methods[0], str(output_path / "interactive.html"))
        
        # 5. Cluster analysis
        if SKLEARN_AVAILABLE:
            cluster_results = self.analyze_clusters(methods[0])
            
            # Save cluster analysis as text report
            with open(output_path / "cluster_analysis.txt", 'w') as f:
                f.write("EMBEDDING CLUSTER ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Method: {cluster_results['method'].upper()}\n")
                f.write(f"Number of clusters: {cluster_results['n_clusters']}\n")
                f.write(f"K-means inertia: {cluster_results['inertia']:.2f}\n\n")
                
                for cluster_id, analysis in cluster_results['analysis'].items():
                    f.write(f"{cluster_id.upper()}:\n")
                    f.write(f"  Size: {analysis['size']} chunks\n")
                    f.write(f"  Dominant speaker: {analysis['dominant_speaker']}\n")
                    f.write(f"  Purity: {analysis['purity']:.3f}\n")
                    f.write(f"  Speaker distribution: {analysis['speaker_distribution']}\n\n")
        
        print(f"✅ Visualization report saved to {output_path}/")
        return output_path


# Example usage and demonstration
if __name__ == "__main__":
    import sys
    
    # Find an audio file
    possible_files = [
        "Urdu/1.wav", "Urdu/3.wav", "Urdu/4.wav", "Urdu/5.wav", "Urdu/6.wav", "better5.wav"
    ]
    
    audio_file = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            audio_file = file_path
            break
    
    if not audio_file:
        print("❌ No audio files found!")
        print("Please ensure you have audio files in the Urdu/ directory")
        sys.exit(1)
    
    print("🎵 EMBEDDING VISUALIZATION DEMO")
    print("=" * 50)
    print(f"📁 Using audio file: {audio_file}")
    
    # Initialize visualizer
    visualizer = EmbeddingVisualizer()
    
    try:
        # Extract embeddings
        embeddings_data = visualizer.extract_and_prepare_embeddings(audio_file)
        
        # Generate visualization report
        output_dir = visualizer.save_visualization_report()
        
        print("\n🎉 SUCCESS!")
        print(f"📊 Visualizations saved to: {output_dir}/")
        print("\nGenerated files:")
        for file_path in output_dir.glob("*"):
            print(f"  📄 {file_path.name}")
        
        print("\n💡 What to look for in the visualizations:")
        print("  1. Distinct clusters = good speaker separation")
        print("  2. Color consistency = accurate speaker tracking")
        print("  3. Temporal patterns = speaker turn dynamics")
        print("  4. Cluster purity = embedding quality")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔧 Make sure you have your HF_TOKEN set up and audio files available")
