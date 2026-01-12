import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import os

class MusicSimilaritySearch:
    def __init__(self):
        print("Loading dataset...")
        # loading features and tracks metadata
        self.features = pd.read_csv('fma_metadata/features.csv', header=[0, 1, 2], index_col=0)
        self.tracks = pd.read_csv('fma_metadata/tracks.csv', header=[0, 1], index_col=0)

        # filer to apply small subset
        self.small_tracks = self.tracks[self.tracks['set', 'subset'] <= 'small']

        # get features for small subset only
        self.features = self.features.loc[self.small_tracks.index]

        print(f"Loaded {len(self.features)} tracks with")
        
        self._prepare_features()

    def _prepare_features(self):
        """Extract and normalize relevant features"""

        # use MFCC features (good for timbre)
        mfcc_features = self.features['mfcc']

        #remove any rows with missing values
        mfcc_features = mfcc_features.dropna()

        #normalize features 
        scaler = StandardScaler()
        self.features_matrix = scaler.fit_transform(mfcc_features)
        self.features_indices = mfcc_features.index

        print(f"Feature matrix shape: {self.features_matrix.shape}")

    def find_similar(self, track_id, n_similar=5):
        """Find n most similar tracks to the given track_id"""

        if track_id not in self.features_indices:
            print(f"Track {track_id} not found in the dataset.")
            return None
        
        #get index of the track
        idx = self.features_indices.get_loc(track_id)

        #get features vector for this track
        track_features = self.features_matrix[idx].reshape(1, -1)

        #compute cosine similarity with all the tracks
        similarities = cosine_similarity(track_features, self.features_matrix)[0]

        # get indices of the most similar tracks (excluding the track itself)
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]

        #get track ids
        similar_track_ids = [self.features_indices[i] for i in similar_indices]
        similar_scores = [similarities[i] for i in similar_indices]

        return similar_track_ids, similar_scores
    
    def get_track_info(self, track_id):
        """Get information about a track"""
        if track_id not in self.small_tracks.index:
            print(f"Track {track_id} not found in the dataset.")
            return None
        
        track = self.tracks.loc[track_id]
        info = {
            'title': track['track', 'title'],
            'artist': track['artist', 'name'],
            'genre': track['track', 'genre_top'],
        }
        return info
    
    def search_and_display(self, track_id, n_similar=5):
        """Search for similar tracks and display their result"""
        print(f"\n{'='*60}")
        print(f"Finding songs similar to Track ID: {track_id}...")
        print(f"\n{'='*60}")

        #get infor about query tracks
        query_info = self.get_track_info(track_id)
        if query_info:
            print(f"Query track:")
            print(f"    Title: {query_info['title']}")
            print(f"    Artist: {query_info['artist']}")
            print(f"    Genre: {query_info['genre']}")

        #find similar tracks
        results = self.find_similar(track_id, n_similar)

        if results is None:
            return
        
        similar_ids, scores = results
        
        print(f"\n{'='*60}")
        print(f"Top {n_similar} similar tracks:")
        print(f"\n{'='*60}")

        for i, (tid, score) in enumerate(zip(similar_ids, scores), 1):
            info = self.get_track_info(tid)
            if info:
                print(f"{i}. [Similarity: {score:.3f}] Track ID: {tid}")
                print(f"    Title: {info['title']}")
                print(f"    Artist: {info['artist']}")
                print(f"    Genre: {info['genre']}")
                print()

if __name__ == "__main__":

    #initialize the search system

    searcher = MusicSimilaritySearch()

    #Try searching for similar tracks
    #Use track ID 2 as an example

    searcher.search_and_display(track_id=2, n_similar=5)

    searcher.search_and_display(track_id=5, n_similar=5)

