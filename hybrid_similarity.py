import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from genre_compatibility import GenreCompatibility

class HybridMusicSimilarity:
    def __init__(self):
        """Intialize hybrid similarity system (audio + metadata)"""
        print("Loading Dataset...")
        self.features = pd.read_csv('fma_metadata/features.csv', index_col=0, header=[0, 1, 2])
        self.tracks = pd.read_csv('fma_metadata/tracks.csv', index_col=0, header=[0, 1])

        # Filter to small subset
        self.small_tracks = self.tracks[self.tracks['set', 'subset'] <= 'small']
        self.features = self.features.loc[self.small_tracks.index]

        print(f"Loaded {len(self.features)} tracks")

        self.available_features = self.features.columns.get_level_values(0).unique().tolist()
        self.feature_cache = {}

        # Initialize genre compatibility system
        print("\nInitializing genre compatibility system")
        self.genre_compat = GenreCompatibility()

        print("\nHybrid System ready")

    def prepare_weighted_features(self, feature_weights, name="weighted"):
        """Prepare features with custom weights"""
        print(f"\nPreparing weighted feature set: '{name}'")
        print(f"Weights: {feature_weights}")

        weighted_features_list = []
        feature_names = []
        scalers = []

        for feature_name, weight in feature_weights.items():
            if feature_name not in self.available_features:
                print(f"Warning: '{feature_name}' not found in dataset")
                continue

            feature_data = self.features[feature_name].dropna()

            if len(feature_data) == 0:
                print(f"Warning: '{feature_name}' has no data after dropping NaN")
                continue

            print(f"    {feature_name:20s}: weight: {weight:.2f}, shape: {feature_data.shape}")

            scaler = StandardScaler()
            normalized = scaler.fit_transform(feature_data)
            weighted = normalized * np.sqrt(weight)

            weighted_features_list.append(weighted)
            feature_names.append(feature_name)
            scalers.append(scaler)
        
        if not weighted_features_list:
            print("Error: No valid features found")
            return None
        
        print(f"\nFinding common indices across {len(feature_names)} features....")

        indices = self.features[feature_names[0]].dropna().index
        for fname in feature_names[1:]:
            indices = indices.intersection(self.features[fname].dropna().index)

        print(f"    Found {len(indices)} tracks with all features")

        if len(indices) == 0:
            print("ErrorL No tracks have all the requested features")
            return None
        
        aligned_features = []
        for i, fname in enumerate(feature_names):
            feat_data = self.features[fname].dropna()
            indexer = feat_data.index.get_indexer(indices)
            aligned_feat = weighted_features_list[i][indexer]
            aligned_features.append(aligned_feat)

        final_matrix = np.hstack(aligned_features)

        print(f"\n Final weighted matrix shape: {final_matrix.shape}")
        print(f" ({len(indices)} tracks * {final_matrix.shape[1]} weighted features)")

        self.feature_cache[name] = {
            'matrix': final_matrix,
            'indices': indices,
            'weights': feature_weights.copy(),
            'feature_names': feature_names,
            'scalers': scalers
        }

        return name
    
    def load_weights_from_file(self, filepath='recommended_feature_weights.json', config_name='balanced'):
        """Load feature weights from file"""

        import json

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if config_name not in data['configurations']:
                print(f"Configuration '{config_name}' not found")
                return None
            
            weights = data['configurations'][config_name]['features']
            description = data['configurations'][config_name]['description']

            print(f"\nLoading '{config_name}' configuration:")
            print(f"    {description}")

            return self.prepare_weighted_features(weights, name=config_name)
        
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            return None
        except Exception as e:
            print(f"Error loading weights: {e}")
            return None
        
    def calculate_audio_similarity(self, track_id, feature_set_name):
        """Calculate audio-only similarity (existing method)"""
    
        if feature_set_name not in self.feature_cache:
          print(f"Error: Feature set '{feature_set_name}' not prepared")
          return None
    
        fs = self.feature_cache[feature_set_name]
    
        if track_id not in fs['indices']:
            print(f"Track {track_id} not in this feature set")
            print(f"Feature set has {len(fs['indices'])} tracks")
            print(f"Track {track_id} might not have all required features: {fs['feature_names']}")
            return None
    
        idx = fs['indices'].get_loc(track_id)
        query_vector = fs['matrix'][idx].reshape(1, -1)
    
        # This returns similarities for ALL tracks in the feature set
        similarities = cosine_similarity(query_vector, fs['matrix'])[0]
    
        print(f"DEBUG calculate_audio_similarity:")
        print(f"  Feature set '{feature_set_name}' has {len(fs['indices'])} tracks")
        print(f"  Calculated {len(similarities)} similarity scores")
    
        return similarities
    
    def calculate_genre_similarity(self, query_track_id, candidate_track_ids):
        """
        Calculate genre similarity between query and candidates

        Args:
            query_track_id: ID of the query track
            candidate_track_ids: List of candidate track IDs

        Returns:
            dict: {track_id: genre_similarity_score}
        """

        # Get query track genre
        if query_track_id not in self.tracks.index:
            return {}
        
        query_genre = self.tracks.loc[query_track_id, ('track', 'genre_top')]

        genre_scores = {}


        for cand_id in candidate_track_ids:
            if cand_id not in self.tracks.index:
                genre_scores[cand_id] = 0.5 # Unknown
                continue

            cand_genre = self.tracks.loc[cand_id, ('track', 'genre_top')]

            # Calculate genre compatibility
            score = self.genre_compat.get_compatibility(query_genre, cand_genre)
            genre_scores[cand_id] = score

        return genre_scores
    
    # def find_similar_hybrid(self, track_id, feature_set_name='balanced', n_similar=10, audio_weight=0.7, genre_weight=0.3):
    #     """
    #     Find Similar tracks using hybrid scoring (audio + genre)
        
    #     Args:
    #         track_id: Query track ID
    #         feature_set_name: Which feature set to use
    #         n_similar: Number of results
    #         audio_weight: Weight for audio similarity (0-1)
    #         genre_weight: Weight for genre similarity (0-1)

    #     Returns:
    #         Similar tracks with breakdown of scores
    #     """

    #     print(f"\n{'='*80}")
    #     print(f"HYBRID SIMILARITY SEARCH")
    #     print(f"  Audio weight: {audio_weight:.2f}")
    #     print(f"  Genre weight: {genre_weight:.2f}")
    #     print(f"{'='*80}\n")

    #     # Calculate audio similarity
    #     audio_similarities = self.calculate_audio_similarity(track_id, feature_set_name)

    #     if audio_similarities is None:
    #         return None
        
    #     # Get all track IDs
    #     fs = self.feature_cache[feature_set_name]
    #     all_track_ids = fs['indices'].tolist()

    #     # Calculate genre similarity
    #     genre_similarities = self.calculate_genre_similarity(track_id, all_track_ids)

    #     hybrid_scores = {}

    #     combined_scores = {}

    #     print(f"\nDEBUG: Processing {len(all_track_ids)} candidate tracks...")
    #     print(f"Tracks in feature set: {len(hybrid.feature_cache['balanced']['indices'])}")

    #     skipped_query = 0
    #     processed = 0

    #     for i, tid in enumerate(all_track_ids):
    #         if tid == track_id:
    #             skipped_query += 1
    #             continue # Skip query track itself

    #         audio_sim = audio_similarities[i]
    #         genre_sim = genre_similarities.get(tid, 0.5) # Default 0.5 if unknown

    #         # Weighted combination
    #         hybrid_score = (audio_weight * audio_sim) + (genre_weight * genre_sim)

    #         hybrid_scores[tid] = {
    #             'hybrid': hybrid_score,
    #             'audio': audio_sim,
    #             'genre': genre_sim
    #         }
    #         combined_scores[tid] = {
    #             'hybrid': hybrid_score,
    #             'audio': audio_sim,
    #             'genre': genre_sim
    #         }
    #         processed += 1
    #         print(f"  Skipped query track: {skipped_query}")
    #         print(f"  Processed candidates: {processed}")
    #         print(f"  Scores calculated: {len(combined_scores)}")
    
    #         if len(combined_scores) == 0:
    #             print("ERROR: No candidates after scoring!")
    #             return None



    #         # Sort by hybrid score
    #         sorted_tracks = sorted(hybrid_scores.items(), key=lambda x: x[1]['hybrid'], reverse=True)

    #         print(f"  Top score: {sorted_tracks[0][1]['hybrid']:.3f}")
    #         print(f"  Lowest score: {sorted_tracks[-1][1]['hybrid']:.3f}")

    #         # Get top N
    #         top_tracks = sorted_tracks[:n_similar]

    #         return top_tracks

    def find_similar_hybrid(self, track_id, feature_set_name="balanced", n_similar=10,
                       audio_weight=0.7, genre_weight=0.3):
        """
        Find similar tracks using hybrid scoring (audio + genre)
        """
    
        print(f"\nHybrid Search: audio_weight={audio_weight:.2f}, genre_weight={genre_weight:.2f}")
    
        # Calculate audio similarity
        audio_similarities = self.calculate_audio_similarity(track_id, feature_set_name)
    
        if audio_similarities is None:
            return None
    
        # CRITICAL FIX: Get track IDs from the FEATURE SET, not full dataset
        fs = self.feature_cache[feature_set_name]
        all_track_ids = fs['indices'].tolist()  # These are the tracks with features
    
        print(f"DEBUG find_similar_hybrid:")
        print(f"  Track IDs from feature set: {len(all_track_ids)}")
        print(f"  Audio similarities calculated: {len(audio_similarities)}")
    
        # Verify they match
        if len(all_track_ids) != len(audio_similarities):
            print(f"ERROR: Mismatch! {len(all_track_ids)} tracks but {len(audio_similarities)} similarities")
            return None
    
        # Calculate genre similarity for tracks in feature set
        genre_similarities = self.calculate_genre_similarity(track_id, all_track_ids)
    
        print(f"  Genre similarities calculated: {len(genre_similarities)}")
    
        # Combine scores
        combined_scores = {}
    
        skipped_query = 0
        processed = 0
    
        for i, tid in enumerate(all_track_ids):
            if tid == track_id:
                skipped_query += 1
                continue  # Skip query track itself
        
            audio_sim = audio_similarities[i]
            genre_sim = genre_similarities.get(tid, 0.5)  # Default 0.5 if unknown
        
            # Weighted combination
            hybrid_score = (audio_weight * audio_sim) + (genre_weight * genre_sim)
        
            # Store all scores
            combined_scores[tid] = {
               'hybrid': hybrid_score,
                'audio': audio_sim,
                'genre': genre_sim
            }
        
            processed += 1
    
        print(f"  Skipped query track: {skipped_query}")
        print(f"  Processed candidates: {processed}")
        print(f"  Scores calculated: {len(combined_scores)}")
    
        if len(combined_scores) == 0:
            print("ERROR: No candidates after scoring!")
            return None
    
        # Sort by hybrid score
        sorted_tracks = sorted(combined_scores.items(), 
                          key=lambda x: x[1]['hybrid'], 
                          reverse=True)
    
        print(f"  Top 5 scores: {[s[1]['hybrid'] for s in sorted_tracks[:5]]}")
    
        # Get top N
        top_tracks = sorted_tracks[:n_similar]
    
        return top_tracks

            
    def get_track_info(self, track_id):
        """Get track metadata"""
        if track_id not in self.tracks.index:
            return None
        
        track = self.tracks.loc[track_id]
    
        return {
            'id': track_id,
            'title': track['track', 'title'],
            'artist': track['artist', 'name'],
            'genre': track['track', 'genre_top']
        }
        
    def display_hybrid_results(self, track_id, feature_set_name='balanced', n_similar=10, audio_weight=0.7, genre_weight=0.3):
        """Display hybrid similarity results with score breakdown"""

        print(f"\n{'='*80}")
        print(f"HYBRID SIMILARITY SEARCH: Track {track_id}")
        print(f"{'='*80}\n")


        #Query track info
        query_info = self.get_track_info(track_id)
        if query_info:
            print("Query Track:")
            print(f"  Title:  {query_info['title']}")
            print(f"  Artist: {query_info['artist']}")
            print(f"  Genre:  {query_info['genre']}")
            print()

        results = self.find_similar_hybrid(track_id, feature_set_name, n_similar, audio_weight, genre_weight)

        if not results:
            return
        
        print(f"Top {n_similar} Similar Tracks:")
        print("-" * 80)
        print(f"{'#':<4} {'Score':<8} {'Audio':<8} {'Genre':<8} {'Track'}")
        print("-" * 80)

        for i, (tid, scores) in enumerate(results, 1):
            info = self.get_track_info(tid)
            if info:
                hybrid = scores['hybrid']
                audio = scores['audio']
                genre = scores['genre']

                print(f"{i:<4} {hybrid:.3f}   {audio:.3f}   {genre:.3f}   "
                      f"{info['title'][:30]:30s}")
                print(f"{'':4} {'':8} {'':8} {'':8} "
                      f"{info['artist'][:25]:25s} | {info['genre']}")
                print()

    def compare_audio_vs_hybrid(self, track_id, feature_set_name='balanced', n_similar=5):
        """Compare audio-only vs hybrid results side by side"""

        print(f"\n{'='*100}")
        print(f"COMPARISON: Audio-Only vs Hybrid Similarity")
        print(f"Query Track: {track_id}")
        print(f"{'='*100}\n")

        query_info = self.get_track_info(track_id)
        if query_info:
            print(f"Query: {query_info['title']} - {query_info['artist']} [{query_info['genre']}]")
            print()

        # Get audio-only results 
        audio_similarities = self.calculate_audio_similarity(track_id, feature_set_name)
        fs = self.feature_cache[feature_set_name]

        # Sort by audio results
        audio_results = []
        for i,tid in enumerate(fs['indices']):
            if tid != track_id:
                audio_results.append((tid, audio_similarities[i]))
        audio_results.sort(key=lambda x: x[1], reverse=True)
        audio_top = audio_results[:n_similar]

        # Get hybrid results
        hybrid_results = self.find_similar_hybrid(track_id, feature_set_name, n_similar, audio_weight=0.7, genre_weight=0.3)

        # Display side by side
        print(f"{'AUDIO-ONLY':<80} | {'HYBRID (70% audio + 30% genre)': <80}")
        print("-" * 100)

        for i in range(n_similar):
            if i < len(audio_top):
                audio_tid, audio_score = audio_top[i]
                audio_info = self.get_track_info(audio_tid)
                if audio_info:
                     # Safely get strings with defaults
                     title = str(audio_info.get('title', 'Unknown'))[:20]
                     artist = str(audio_info.get('artist', 'Unknown'))[:20]
                     genre = str(audio_info.get('genre', 'Unknown'))[:15]
                     audio_text = f"{i+1}. [{audio_score:.3f}] {title:20s} | {artist:20s} | {genre:15s}"
                else:
                    audio_text = f"{i+1}. Track {audio_tid} - Info not available"
            else:
                    audio_text = ""
            
            # Hybrid
            if i < len(hybrid_results):
                hybrid_tid, hybrid_scores = hybrid_results[i]
                hybrid_info = self.get_track_info(hybrid_tid)
                if hybrid_info:
                    # Safely get strings with defaults
                    title = str(hybrid_info.get('title', 'Unknown'))[:20]
                    artist = str(hybrid_info.get('artist', 'Unknown'))[:20]
                    genre = str(hybrid_info.get('genre', 'Unknown'))[:15]
                    hybrid_text = f"{i+1}. [{hybrid_scores['hybrid']:.3f}] {title:20s} | {artist:20s} | {genre:15s}"
                else:
                    hybrid_text = f"{i+1}. Track {hybrid_tid} - Info not available"
            else:
                hybrid_text = ""
        
            print(f"{audio_text:<80}  |  {hybrid_text:<80}")
        
        print("\n" + "="*100)
        print("ANALYSIS:")
        print("- Left side: Pure audio similarity (may cross genres)")
        print("- Right side: Hybrid (balances audio + genre compatibility)")
        print("- Notice which results feel more coherent")
        print("="*100)
    
    def track_file_exists(self, track_id):
        """Check if audio file exists for a track"""
        # FMA file structure: fma_small/xxx/track_id.mp3
        # Where xxx is first 3 digits of track_id padded to 6 digits
    
        track_id_str = str(track_id).zfill(6)  # Pad to 6 digits: 2 -> 000002
        folder = track_id_str[:3]  # First 3 digits: 000
        filepath = f'fma_small/{folder}/{track_id_str}.mp3'
    
        return os.path.exists(filepath)

    def find_similar_hybrid_with_files(self, track_id, feature_set_name="balanced", 
                                   n_similar=10, audio_weight=0.7, genre_weight=0.3):
        """
        Find similar tracks, but only return ones with actual audio files
        """
    
        # Get more results than needed (in case some are missing)
        results = self.find_similar_hybrid(track_id, feature_set_name, 
                                      n_similar=n_similar*3,  # Get 3x more
                                      audio_weight=audio_weight, 
                                      genre_weight=genre_weight)
    
        if not results:
            return None
    
        # Filter to only tracks with files
        filtered_results = []
        for tid, scores in results:
            if self.track_file_exists(tid):
                filtered_results.append((tid, scores))
                if len(filtered_results) >= n_similar:
                    break
    
        print(f"\nFiltered to {len(filtered_results)} tracks with audio files (from {len(results)} candidates)")

        return filtered_results


if __name__ == "__main__":
    print("\n" + "="*50)
    print("HYYRID MUSIC SIMILARITY SYSTEM")
    print("="*50)

    # Initialize
    hybrid = HybridMusicSimilarity()

    # Load recommended weights
    import os
    if os.path.exists('recommended_feature_weights.json'):
        hybrid.load_weights_from_file(config_name='comprehensive')
    else:
        weights = {
            'mfcc': 2.5,
            'chroma_cqt': 1.5,
            'spectral_centroid': 1.0
        }

        hybrid.prepare_weighted_features(weights, name='comprehensive')
    
    # Test track
    print("\n" + "="*80)
    test_track = int(input("Enter track ID to test (e.g. 2): "))

    # Compare audio only vs hybrid
    hybrid.compare_audio_vs_hybrid(test_track, n_similar=20)

    print("\n" + "=" * 80)
    print("Want to see full hybrid results with score breakdown? (y/n)")
    if input().lower() == 'y':
        hybrid.display_hybrid_results(test_track, n_similar=20, audio_weight=0.7, genre_weight=0.3)
    

    print("\n" + "="*80)
    print("Try different weights? (y/n)")
    if input().lower() == 'y':
        audio_w = float(input("Audio weight (0-1, defualt 0.7): ") or 0.7)
        genre_w = 1.0 - audio_w

        hybrid.display_hybrid_results(test_track, n_similar=20, audio_weight=audio_w, genre_weight=genre_w)

