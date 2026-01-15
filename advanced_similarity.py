import pandas as pd
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt

class AdvancedMusicSimilarity:
    def __init__(self): 
        print("Loading dataset...")
        # loading features and tracks metadata
        self.features = pd.read_csv('fma_metadata/features.csv', header=[0, 1, 2], index_col=0)
        self.tracks = pd.read_csv('fma_metadata/tracks.csv', header=[0, 1], index_col=0)

        # filer to apply small subset
        self.small_tracks = self.tracks[self.tracks['set', 'subset'] <= 'small']

        # get features for small subset only
        self.features = self.features.loc[self.small_tracks.index]

        print(f"Loaded {len(self.features)} tracks")

        self.feature_sets = {}
        self.prepared_matrices = {}
        
        #Available feature categories
        self.available_features = self.features.columns.get_level_values(0).unique().tolist()
        print(f"\nAvailable feature categories: {self.available_features}")

    def prepare_feature_set(self, feature_list, name="custom"):
        """Prepare a feature set from selected feature categories
           Args:
                feature_list: List of feature category names (e.g. ['mfcc', 'chroma_cqt'])
                name: Name for this feature set
        """
        print(f"\nPreparing feature set '{name}' with:{feature_list}")

        #Extract selected features
        selected_features = pd.DataFrame()
        for feat in feature_list:
            if feat in self.available_features:
                selected_features = pd.concat([selected_features, self.features[feat]], axis=1)
            else:
                print(f"Warning: Feature '{feat}' not found in dataset.")

        #Remove rows with missing values
        selected_features = selected_features.dropna()

        print(f"Feature matrix shape before scaling: {selected_features.shape}")
        
        #Normalize features
        scaler = StandardScaler()
        features_matrix = scaler.fit_transform(selected_features)

        #Store
        self.feature_sets[name] = {
            'feature_list': feature_list,
            'raw_features': selected_features,
            'matrix': features_matrix,
            'indices': selected_features.index,
            'scaler': scaler
        }

        print(f"Feature set '{name}' prepared with shape: {features_matrix.shape}")
        return name
    
    def analyze_feature_importance(self, feature_list=None):
        """
        Data-driven analysis to understand feature importance
        """

        if feature_list is None:
            feature_list = self.available_features

        print("\n" + "="*60)
        print("DATA-DRIVEN FEATURE ANALYSIS")
        print("="*60)

        #prepare features
        all_features = pd.DataFrame()
        for feat in feature_list:
            if feat in feature_list:
                all_features = pd.concat([all_features, self.features[feat]], axis=1)

        all_features = all_features.dropna()

        #1. Variance Analysis
        print("\n1. VARIANCE ANALYSIS (Removing low frequency features)")
        print("-"*60)

        selector = VarianceThreshold(threshold=0.01)
        selector.fit(all_features)

        variance = selector.variances_
        high_var_features = all_features.columns[selector.get_support()]

        print(f"Features with variance >0.01: {len(high_var_features)}/{all_features.columns}")
        print(f"Removed {len(all_features.columns) - len(high_var_features)} low variance features.")

        # 2. Correlation Analysis
        print("\n2. CORRELATION ANALYSIS (FINDING REDUNDANT FEATURES)")
        print("-"*60)

        corr_matrix = all_features.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        #Find highly correlated features (>0.95)
        highly_correlated = [column for column in upper_triangle.columns
                             if any(upper_triangle[column] > 0.95)]

        print(f"Highly correlated feature pairs (>0.95): {len(highly_correlated)}")
        print("These features are redundant and can be considered for removal.")

        # 3. PCA Analysis
        print("\n3. PCA Analysis (dimensionality reduction)")
        print("-"*60)

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(all_features)

        pca = PCA()
        pca.fit(scaled_features)

        #Cumulative explained variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)

        #Find number of components for 95% variance
        n_components_95 = np.argmax(cumsum >= 0.95) + 1
        n_components_99 = np.argmax(cumsum >= 0.99) + 1
        

        print(f"Components for 95% variance: {n_components_95}/{len(cumsum)}")
        print(f"Components for 99% variance: {n_components_99}/{len(cumsum)}")
        print(f"This suggests we could reduce from {all_features.shape[1]} to ~{n_components_95} features.")

        # 4. Feature category importance
        print("\n4. FEATURE CATEGORY VARIANCE IMPORTANCE")
        print("-"*60)

        category_variance = {}
        for category in feature_list:
            if category in self.available_features:
                cat_features = self.features[category].dropna()
                if len(cat_features) > 0:
                    cat_scaled = StandardScaler().fit_transform(cat_features)
                    category_variance[category] = np.mean(np.var(cat_scaled, axis=0))

        #Sort by variance
        sorted_categories = sorted(category_variance.items(), key=lambda x: x[1], reverse=True)

        print("\nFeature cateogories ranked by variance (higher = more informative):")
        for i, (cat, var) in enumerate(sorted_categories, 1):
            print(f"{i}. {cat:20s}: {var:.4f}")

        #Recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("-"*60)

        top_categories = [cat for cat, _ in sorted_categories[:5]]
        print(f"\nTop 5 feature categories to use: {top_categories}")

        print(f"\nSuggested feature sets:")
        print(f"  Minimal (fast):    {sorted_categories[0][0]}")
        print(f"  Balanced (recommended):         {[cat for cat, _ in sorted_categories[:3]]}")
        print(f"  Comprehensive:     {[cat for cat, _ in sorted_categories[:5]]}")

        return {
            'variance_analysis': high_var_features,
            'highly_correlated': highly_correlated,
            'pca_components_95': n_components_95,
            'pca_components_99': n_components_99,
            'category_ranking': sorted_categories,
            'recommended_minimal': [sorted_categories[0][0]],
            'recommended_balanced': [cat for cat, _ in sorted_categories[:3]],  
            'recommended_comprehensive': [cat for cat, _ in sorted_categories[:5]],
        }
    
    def find_similar(self, track_id, feature_set_name="custom", n_similar=5):
        """Find similar tracks using specified feature set"""

        if feature_set_name not in self.feature_sets:
            print(f"Error: Feature set '{feature_set_name}' not found.")
            print(f"Available sets: {list(self.feature_sets.keys())}")
            return None
        
        fs = self.feature_sets[feature_set_name]

        if track_id not in fs['indices']:
            print(f"Track {track_id} not found in the dataset for feature set '{feature_set_name}'.")
            return None
        
        #Get index
        idx = fs['indices'].get_loc(track_id)

        #Get features vector
        track_features = fs['matrix'][idx].reshape(1, -1)

        #Compute similarities
        similarities = cosine_similarity(track_features, fs['matrix'])[0]

        #Get top similar
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
        similar_track_ids = [fs['indices'][i] for i in similar_indices]
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
    
    def compare_feature_sets(self, track_id, feature_sets_to_compare, n_similar=5):
        """
        Compare resultes from different feature sets side-by-side
        
        """

        print(f"\n{'='*80}")
        print(f"COMPARING FEATURE SETS FOR TRACK {track_id}")
        print(f"{'='*80}\n")

        #show query results
        query_info = self.get_track_info(track_id)
        if query_info:
            print(f"Query Track:")
            print(f"    Title: {query_info['title']}")
            print(f"    Artist: {query_info['artist']}")
            print(f"    Genre: {query_info['genre']}")
            print()

        #compare each feature set
        for fs_name in feature_sets_to_compare:
            if fs_name not in self.feature_sets:
                print(f"Feature set '{fs_name}' not found. Skipping...")
                continue

            print(f"\n{'-'*80}")
            print(f"FEATURE SET: {fs_name}")
            print(f"Features used: {self.feature_sets[fs_name]['feature_list']}")
            print(f"{'-'*80}\n")

            results = self.find_similar(track_id, fs_name, n_similar)

            if results:
                similar_ids, scores = results
                
                for i, (tid, score) in enumerate(zip(similar_ids, scores), 1):
                    info = self.get_track_info(tid)
                    if info:
                        print(f"{i}. [Similarity: {score:.3f} Track {tid}]")
                        print(f" {info['title']} - {info['artist']}")
                        print(f" Genre: {info['genre']}\n")
        
        print(f"{'='*80}\n")
    

if __name__ == "__main__":

        #Initialize 
        searcher = AdvancedMusicSimilarity()

        #STEP 1: Data-driven analysis to find the best features
        analysis_results = searcher.analyze_feature_importance()

        #STEP 2: Prepare different feature sets based on recommendations

        #Minimal (Fastest, single feature)
        searcher.prepare_feature_set(
            analysis_results['recommended_minimal'],
            name = 'minimal'
        )

        #Balanced (Recommended)
        searcher.prepare_feature_set(
            analysis_results['recommended_balanced'],
            name = 'balanced'
        )

        #Comprehensive (Most features)
        searcher.prepare_feature_set(
            analysis_results['recommended_comprehensive'],
            name = 'comprehensive'
        )

        #Custom: Music-Theory based (Manual selection)
        searcher.prepare_feature_set(
            ['mfcc', 'chroma_cqt', 'tonnetz'],
            name = 'music_theory'
        )

        # All features
        searcher.prepare_feature_set(
            searcher.available_features,
            name = 'all_features'
        )

        #STEP 3 Compare different feature sets for a given track
        print("\n\n" + "Comparing different feature sets...")
        searcher.compare_feature_sets(
            track_id=117971,
            feature_sets_to_compare=['minimal', 'balanced', 'comprehensive', 'music_theory', 'all_features'],
            n_similar=5
        )