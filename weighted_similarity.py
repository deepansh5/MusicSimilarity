import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

class WeightedMusicSimilarity:
    def __init__(self):
        print("Loading dataset...")
        self.features = pd.read_csv('fma_metadata/features.csv', index_col=0, header=[0, 1, 2])
        self.tracks = pd.read_csv('fma_metadata/tracks.csv', index_col=0, header=[0, 1])

        #Filter to small subset
        self.small_tracks = self.tracks[self.tracks['set', 'subset'] <= 'small']
        self.features = self.features.loc[self.small_tracks.index]

        print(f"Loaded {len(self.features)} tracks")

        self.available_features = self.features.columns.get_level_values(0).unique().tolist()
        self.feature_cache = {} # Cache prepared features

    def prepare_weighted_features(self, feature_weights, name="weighted"):
        """
        Prepare features with custom weights

        Args:
            feature_weights: Dict mapping feature names to weights
                Example: {'mfcc': 2.0, 'chroma_cqt': 0.5, 'spectral_centroid': 1.0}
            name: Name for this wieghted feature set
        
        Returns: 
            Name of the prepared feature set
        """
        print(f"\nPreparing weighted feature set: '{name}'")
        print(f"Weights: {feature_weights}")

        # Extract and weight each feature category
        weighted_features_list = []
        feature_names = []
        scalers = []

        for feature_name, weight in feature_weights.items():
            if feature_name not in self.available_features:
                print(f"Warning: '{feature_name}' not found in dataset.")
                continue

            # Get this feature category
            feature_data = self.features[feature_name].dropna()

            if len(feature_data) == 0:
                print(f"Warning: '{feature_name}' has no data after dropping NaN")
                continue


            print(f"    {feature_name:20s}: weight={weight:.2f}, shape={feature_data.shape}")

            # Normalize first (mean = 0, std = 1)
            scaler = StandardScaler()
            normalized = scaler.fit_transform(feature_data)

            # Apply weight using sqrt (because of dot product in cosine similarity)
            # If weight = 4, we want 4x importance, so multiply by sqrt(4) = 2
            # Because in dot product: (2x)^2 = 4x
            weighted = normalized * np.sqrt(weight)

            weighted_features_list.append(weighted)
            feature_names.append(feature_name)
            scalers.append(scaler)

        # Concatenate all weighted features
        if not weighted_features_list:
            print("Error: No valid features found")
            return None

        # Find common indices (tracks that have all features)
        indices = self.features[feature_names[0]].dropna().index
        for fname in feature_names[1:]:
            indices = indices.intersection(self.features[fname].dropna().index)
        
        print(f"Found {len(indices)} tracks will all features")

        if len(indices) == 0:
            print("Error: No tracks have all the requested features")
            return None
        

        # Align all feature matrices to common indices
        aligned_features = []
        for i, fname in enumerate(feature_names):
            # Get the feature Data
            feat_data = self.features[fname].dropna()

            # Gt positions of common indices in this feature's index
            indexer = feat_data.index.get_indexer(indices)

            # Extract the weighted features at those positions
            aligned_feat = weighted_features_list[i][indexer]
            aligned_features.append(aligned_feat)


        # # Concatenate features for common indices
        # final_matrix = np.hstack([
        #     feat[self.features[name].dropna().index.get_indexer(indices)]
        #     for feat, fname in zip(weighted_features_list, feature_names)
        # ])
        final_matrix = np.hstack(aligned_features)

        print(f"\n Final weighted matrix shape: {final_matrix.shape}")
        print(f"  ({len(indices)} tracks x {final_matrix.shape[1]} weighted features)")

        # Store
        self.feature_cache[name] = {
            'matrix': final_matrix,
            'indices': indices,
            'weights': feature_weights.copy(),
            'featur_names': feature_names
        }

        return name
    
    def load_weights_from_file(self, filepath='recommended_feature_weights.json', config_name='balanced'):
        """
        Load feature weights from the analyzer output file

        Args:
            filepath: Path to the JSON file from feature_importance_analyzer
            config_name: Which configuration to use ('minimal', 'balanced', 'comprehensive')
        
        Returns:
            Name of the prepared feature set
        """
        import json

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            if config_name not in data['configurations']:
                print(f"Configuration '{config_name}' not found in file")
                print(f"Available: {list(data['confifurations'].keys())}")
                return None
            
            weights = data['configurations'][config_name]['features']
            description = data['configurations'][config_name]['description']

            print(f"\nLoading '{config_name}' configuration:")
            print(f"   {description}")
            print(f"    Timestamp:{data.get('timestamp', 'unknown')}")

            #prepare the weighted features
            return self.prepare_weighted_features(weights, name=config_name)
        
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            print("Run feature_importance_analyzer.py first and save the results")
            return None
        except Exception as e:
            import traceback
            tb = traceback.extract_tb(e.__traceback__)[-1]
            print(f"Error loading weights: {e} on line {tb.lineno}")
            traceback.print_exc()
            return None
        
    def find_similar(self, track_id, feature_set_name='weighted', n_similar=10):
        """
        Find similar tracks using specified weighted feature set
        """

        if feature_set_name not in self.feature_cache:
            print(f"Error: features set '{feature_set_name}' not prepared")
            print(f"Available: {list(self.feature_cache.keys())}")
            return None
        
        fs = self.feature_cache[feature_set_name]

        if track_id not in fs['indices']:
            print(f"Track '{track_id}' not in this feature set")
            return None
        
        # get track's feature vector
        idx = fs['indices'].get_loc(track_id)
        query_vector = fs['matrix'][idx].reshape(1, -1)

        # Compute similarities
        similarities = cosine_similarity(query_vector, fs['matrix'])[0]

        # Get top N (Excluding the query track itself)
        top_indices = np.argsort(similarities)[::-1][1:n_similar+1]

        similar_ids = [fs['indices'][i] for i in top_indices]
        similar_scores = [similarities[i] for i in top_indices]

        return similar_ids, similar_scores
    
    def get_track_info(self, track_id):
        """Get track metadata"""

        if track_id not  in self.tracks.index:
            return None
        
        track = self.tracks.loc[track_id]
        return{
            'id': track_id,
            'title': track['track', 'title'],
            'artist': track['artist', 'name'],
            'genre': track['track', 'genre_top']
        }
    
    def display_results(self, track_id, feature_set_name='weighted', n_similar=10):
        """Display search results with track info"""

        print(f"\n{'='*80}")
        print(f"SIMILARITY SEARCH: Track{track_id}")

        #show weights being used
        if feature_set_name in self.feature_cache:
            weights = self.feature_cache[feature_set_name]['weights']
            print(f"Feature weights: {weights}")

        print(f"{'='*80}\n")

        # Query track info
        query_info = self.get_track_info(track_id)
        if query_info:
            print("Query Track:")
            print(f"     Title:   {query_info['title']}")
            print(f"     Artist:  {query_info['artist']}")
            print(f"     Genre:   {query_info['genre']}")
            print()

        # Find Similar
        results = self.find_similar(track_id, feature_set_name, n_similar)

        if not results:
            return
        
        similar_ids, scores = results

        print(f"Top {n_similar} Similar Tracks:")
        print("-"*80)

        for i, (tid, score) in enumerate(zip(similar_ids, scores), 1):
            info = self.get_track_info(tid)
            if info:
                print(f"{i:2d}. [Similarity: {score:.3f}] Track {tid}")
                print(f"     {info['title']}")
                print(f"     {info['artist']}  |  Genre: {info['genre']}")
                print()
    
    def compare_weight_schemes(self, track_id, weight_schemes, n_similar = 5):
        """
        Comapre results from different weighting schemes side-by-side

        Args: 
            track_id: Track to search for
            weight_schemes: Dict of {name: weights} pairs
            n_similar: Number of results per scheme
        """

        print(f"\n{'='*100}")
        print(f"COMPARING DIFFERENT FEATURE WEIGHTS FOR TRACK {track_id}")
        print(f"{'='*100}\n")

        # Show query track
        query_info = self.get_track_info(track_id)
        if query_info:
            print("Query Track:")
            print(f"    {query_info['title']} - {query_info['artist']} [{query_info['genre']}]")
            print()

        # Prepare all weight schemes
        scheme_names = []
        for scheme_name, weights in weight_schemes.items():
            name = self.prepare_weighted_features(weights, name=scheme_name)
            if name:
                scheme_names.append(name)

        # Compare Results
        print(f"\n{'-'*100}")

        for scheme_name in scheme_names:
            print(f"\nWEIGHT SCHEME: {scheme_name}")
            print(f"Weights: {self.feature_cache[scheme_name]['weights']}")
            print("-"*100)

            results = self.find_similar(track_id, scheme_name, n_similar)

            if results:
                similar_ids, scores = results
                
                for i, (tid, score) in enumerate(zip(similar_ids, scores), 1):
                    info = self.get_track_info(tid)
                    if info:
                        print(f"   {i}. [{score:.3f}]  {info['title'][:40]:40s}  |  {info['artist'][:25]:25s}  |  {info['genre']}")

            print()
        
        print(f"{'='*100}\n")


# Example usage and experimentation
if __name__ == "__main__":
    searcher = WeightedMusicSimilarity()


    print("\n" + "=" * 50)
    print("WEIGHTED SIMILARITY SEARCH")
    print("="*50)


    # Option 1: Load from analyzer results (if available)
    import os
    if os.path.exists('recommended_feature_weights.json'):
        print("\nFound recommended weights from analyzer!")
        print("\nWhich configuration would you like to use?")
        print("    1. minimal - Fastest (1-2 features)")
        print("    2. balanced - Recommended (3-4 features)")
        print("    3. comprehensive - Highest quality (5.6 features)")
        print("    4. manual - Define your own weights")

        choice = input("\nEnter choice (1-4) [default: 2]: ").strip() or "2"

        config_map = {
            '1': 'minimal',
            '2': 'balanced',
            '3': 'comprehensive'
        }

        if choice in config_map:
            config_name = config_map[choice]
            searcher.load_weights_from_file(config_name=config_name)

            # Test it
            test_track = int(input("\nEnter track ID to test: "))
            searcher.display_results(test_track, feature_set_name=config_name, n_similar=10)
        else:
            print("\nUsing manual configuration...")

    # Option 2: manual weight deifnition (fallback)
    if not os.path.exists('recommended_feature_weights.json') or choice == '4':
        print("\nDefining custom weights....")

        weight_schemes = {
            'equal': {
                'mfcc': 1.0,
                'chroma_cqt': 1.0,
                'spectral_centroid': 1.0,
                'spectral_bandwidth': 1.0,
                'spectral_rollof': 1.0
            },
            'timbre_focused': {
                'mfcc': 3.0,
                'chroma_cqt': 0.3,
                'spectral_centroid': 1.0,
                'spectral_bandwidth': 1.0,
                'spectral_rollof': 1.0
            },
            'harmony_focused':{
                'mfcc': 0.5,
                'chroma_cqt': 3.0,
                'spectral_centroid': 0.5,
                'spectral_bandwidth': 0.5,
                'spectral_rollof': 0.5
            },
            'balanced':{
                'mfcc': 2.0,
                'chroma_cqt': 1.0,
                'spectral_centroid': 1.5,
                'spectral_bandwidth': 1.5,
                'spectral_rollof': 1.5
            }
        }

        test_track_id = 2

        searcher.compare_weight_schemes(
            track_id=test_track_id,
            weight_schemes=weight_schemes,
            n_similar=5
        )

        print("\n" + "=" * 50)
        print("ANALYSIS QUESTIONS TO CONSIDER:")
        print("\n" + "=" * 50)
        print("""
              1. Which weighting scheme gives you results that SOUND most similar?
              2. Do 'timbre_focused' results have similar instrumentation?
              3. Do 'harmony_focused' results share the same key/chords?
              4. Is 'balanced' actually better, or does one extreme work better?
              5. Are certain genres more sensitive to specific features?

              TRY THIS:
              - Pick a song you know well
              - Run with different weight schemes
              - Listen to the top 3 results from each
              - Note which scheme matches your intuition the best

              """)
