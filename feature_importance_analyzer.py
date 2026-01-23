import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

class FeatureImportanceAnalyzer:
    def __init__(self):
        print("Loading Dataset... ")
        self.features = pd.read_csv('fma_metadata/features.csv', index_col=0, header=[0, 1, 2])
        self.tracks = pd.read_csv('fma_metadata/tracks.csv', index_col=0, header=[0, 1])

        self.small_tracks = self.tracks[self.tracks[('set', 'subset')] <= 'small']
        self.features = self.features.loc[self.small_tracks.index]

        print(f"Loaded {len(self.features)} tracks")

        self.available_features = self.features.columns.get_level_values(0).unique().tolist()
        print(f"Available feature categories: {self.available_features}")

    def analyze_discriminative_power(self):
        """
        Method 1: How well can each feature category distinguish between different genres:

        Intuition: If a feature category is good at telling genres apart, it's probably capturing something musically meaningful.
        """

        print("\n" + "="*80)
        print("METHOD 1: DISCRIMINATIVE POWER (Genre Classification)")
        print("="*80)
        print("\nQuestion: Which features best distinguish between different genres?")
        print("Intuition: Good features should separate different musical styles.\n")

        #Get genre labels
        genres = self.tracks.loc[self.features.index, ('track', 'genre_top')]

        #Only keep genres with atleast 100 songs
        genre_counts = genres.value_counts()
        common_genres = genre_counts[genre_counts >= 100].index

        mask = genres.isin(common_genres)
        genres_filtered = genres[mask]

        print(f"Using {len(common_genres)} genres with 100+ songs each:")
        print(f"{list(common_genres)}\n")

        # Test each feature category
        results = {}

        for feat_name in self.available_features:
            print(f"Testing {feat_name}... ", end='')

            # Get features for this category
            feat_data = self.features[feat_name].loc[genres_filtered.index].dropna()

            # Align with genres
            common_idx = feat_data.index.intersection(genres_filtered.index)
            x = feat_data.loc[common_idx]
            y = genres_filtered.loc[common_idx]

            if len(x) < 100:
                print("Not enough data")
                continue

            # Normalize
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)

            # Train a simple classifier
            from sklearn.model_selection import cross_val_score
            clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)

            # Cross-validation score
            scores = cross_val_score(clf, x_scaled, y, cv=3, scoring='accuracy')
            avg_scores = scores.mean()

            results[feat_name] = avg_scores
            print(f"Accuracy: {avg_scores:.3f}")

        # Sort by score
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

        print("\n" + "-"*80)
        print("RANKING (Higher = Better at distinguishing genres):")
        print("-"*80)
        
        for i, (feat, score) in enumerate(sorted_results, 1):
            bar = "█" * int(score * 50)
            print(f"{i:2d}. {feat:25s} {score:.3f} {bar}")
        
        print("\nINTERPRETATION:")
        print("- High score = This feature captures genre-specific characteristics")
        print("- Low score = This feature is similar across genres (less discriminative)")
        print("\nFor similarity search: High scoring features are usually important!")

        return sorted_results

    def analyze_variance_contribution(self):
        """
        Method 2: VARIANCE CONTRIBUTION

        Intuition: Features with high variance capture more differneces between songs.
        """
        print("\n" + "="*80)
        print("METHOD 2: VARIANCE CONTRIBUTION")
        print("="*80)
        print("\nQuestion: Which features vary the most across songs?")
        print("Intuition: High variance = more information content\n")

        results = {}

        for feat_name in self.available_features:
            feat_data = self.features[feat_name].dropna()

            if len(feat_data) < 100:
                continue

            # Normalize
            scaler = StandardScaler()
            normalized = scaler.fit_transform(feat_data)

            # Average variance across all sub-features in this category
            avg_variance = np.mean(np.var(normalized, axis=0))
            results[feat_name] = avg_variance

        sorted_results = sorted(results.items(), key=lambda x:x[1], reverse=True)

        print("RANKING (Higher = more variance between songs):")
        print("-"*80)

        # Normalize for visualization
        max_var = max(r[1] for r in sorted_results)

        for i, (feat, var) in enumerate(sorted_results, 1):
            normalized_var = var / max_var
            bar = "█" * int(normalized_var * 50)
            print(f"{i:2d}. {feat:25s} {var:.3f} {bar}")
        
        print("\nINTERPRETATION:")
        print("- High variance = Songs differ a lot in this feature")
        print("- Low variance = Songs are similar in this feature (less useful)")

        return sorted_results
    
    def analyze_pairwise_similarity_agreement(self, n_samples=500):
        """
        Method 3: Which features agree with each other about similarity?

        Intuition: If multiple feature agree that two songs are similar, they're probably capturing something real.
        """
        print("\n" + "="*80)
        print("METHOD 3: INTER FEATURE AGREEMENT")
        print("="*80)
        print("\nQuestion: Which features agree with each other about what's similar?")
        print("Intuition: Features that correlate with others are pronanlu reliable.\n")

        # Sample tracks for efficiency
        sample_indices = np.random.choice(self.features.index, min(n_samples, len(self.features)), replace=False)

        # Compute pairwise similarity for each feature
        similarity_matrices = {}

        print("Computing similarity matrices...")
        for feat_name in self.available_features[:10]: # Limit for speed
            print(f"     {feat_name}...", end='')

            feat_data = self.features[feat_name].loc[sample_indices].dropna()

            if len(feat_data) < 50:
                print("skipped (insufficient data)")
                continue

            # Normalize
            scaler = StandardScaler()
            normalized = scaler.fit_transform(feat_data)

            # Compute pairwise cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(normalized)

            similarity_matrices[feat_name] = sim_matrix.flatten()
            print("done")
            
        # Compute correlation between similarity matrices
        print("\nComputing inter-feature correlations...")

        feature_names = list(similarity_matrices.keys())
        n_features = len(feature_names)

        correlation_matrix = np.zeros((n_features, n_features))

        for i, feat1 in enumerate(feature_names):
            for j, feat2 in enumerate(feature_names):
                if i <= j:
                    corr = np.corrcoef(similarity_matrices[feat1], similarity_matrices[feat2])[0, 1]
                    correlation_matrix[i, j] = corr
                    correlation_matrix[j, i] = corr
        
        # Average correlation with other features

        avg_correlations = {}
        for i,feat in enumerate(feature_names):
            # Average correlation with all other features (excluding self)
            other_corrs = [correlation_matrix[i, j] for j in range(n_features) if j != i]
            avg_correlations[feat] = np.mean(other_corrs)

        sorted_results = sorted(avg_correlations.items(), key=lambda x:x[1], reverse=True)

        print("\nRANKING (Higher = More agreement with other features):")
        print("-"*80)

        for i, (feat, corr) in enumerate(sorted_results, 1):
            bar = "█" * int((corr +1 ) * 50) # +1 because correlation can be negative
            print(f"{i:2d}. {feat:25s} {corr:.3f} {bar}")

        print("\nINTERPRETATION:")
        print("- High Correlation = This feature agrees with others (reliable)")
        print("- Low Correlation = This feature is unique/independent (might be noise OR capturing unique aspect)")

        return sorted_results
    
    def analyze_incremental_value(self):
        """
        Method 4: How much does adding each feature improve results?

        Intuition: Start with the best feature, then see what each additional feature adds.
        """
        print("\n" + "="*80)
        print("METHOD 4: INCREMENTAL VALUE")
        print("="*80)
        print("\nQuestion: How much does each feature add when combined with others?")
        print("Intuition: Some features might be redundant; others add unique info.\n")

        # Get genre labels for classification task
        genres = self.tracks.loc[self.features.index, ('track', 'genre_top')]
        genre_counts = genres.value_counts()
        common_genres = genre_counts[genre_counts >= 100].index
        mask = genres.isin(common_genres)
        genres_filtered = genres[mask]

        # Start with empty set, incrementally add features
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        remaining_features = self.available_features.copy()
        selected_features = []
        scores = []

        print("Building feature set incrementally...")

        max_features = min(10, len(remaining_features))

        while len(selected_features) < max_features and remaining_features:
            best_score = 0
            best_feature = None

            features_to_test = remaining_features[:min(5, len(remaining_features))]

            for feat_name in features_to_test: # Limit for speed
                try:
                    # Try adding this feature
                    test_features = selected_features + [feat_name]

                    # Combine features
                    combined = pd.DataFrame()
                    for f in test_features:
                        feat_data = self.features[f].loc[genres_filtered.index]
                        combined = pd.concat([combined, feat_data], axis=1)

                    combined = combined.dropna()

                    if len(combined) < 100:
                        continue

                    # Align with labels
                    common_idx = combined.index.intersection(genres_filtered.index)
                    x = combined.loc[common_idx]
                    y = genres_filtered.loc[common_idx]

                    if len(x) < 100:
                        continue

                    # Normalize
                    scaler = StandardScaler()
                    x_scaled = scaler.fit_transform(x)

                    # Quick train/test split
                    x_train, x_test, y_train, y_test = train_test_split(
                        x_scaled, y, test_size=0.3, random_state=42
                    )

                    # Train classified
                    clf = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42)
                    clf.fit(x_train, y_train)
                    score = clf.score(x_test, y_test)

                    if score > best_score:
                        best_score = score
                        best_feature = feat_name
                
                except Exception as e:
                    print(f"   Error testing {feat_name}: {e}")
                    continue
                
                if best_feature:
                    selected_features.append(best_feature)
                    if best_feature in remaining_features:
                        remaining_features.remove(best_feature)
                    
                    scores.append(best_score)

                    improvement = scores[-1] - scores[-2] if len(scores) > 1 else scores[0]
                    print(f" Added '{best_feature}': accuracy = {best_score:.3f} (improvement: {improvement:.3f})")
                else:
                    print("    No further improvement possible, stoppinng")
                    break
            
            print("\nORDER OF IMPORTANCE (by incremental value):")
            print("-"*80)
            for i, (feat, score) in enumerate(zip(selected_features, scores), 1):
                improvement = score - scores[i-2] if i > 1 else score
                print(f"{i:2d}. {feat:25s} Accuracy: {score:.3f} (improvement: {improvement:.3f})")

        print("\nINTERPRETATION:")
        print("- Features at the top add the most unique information.")
        print("- Feature lower down might be redundant with earlier ones")
        print("- Diminishing returns suggest when to stop adding features.")

        return list(zip(selected_features, scores))
    
    def generate_recommendations(self):
        """
        Combine all analyses to generate concrete recommendations
        """
        print("\n" + "="*80)
        print("GENERATING RECOMMENDATIONS")
        print("="*80)
        
        # Run all analyses
        print("\nRunning comprehensive analysis (this may take a few minutes)...\n")
        
        discriminative = self.analyze_discriminative_power()
        variance = self.analyze_variance_contribution()
        agreement = self.analyze_pairwise_similarity_agreement()
        incremental = self.analyze_incremental_value()
        
        # Combine rankings (now with all 4 methods)
        print("\n" + "="*80)
        print("COMBINING ALL ANALYSES")
        print("="*80)
        
        # Convert each analysis to rankings (lower rank = better)
        rankings = {}
        
        # Method 1: Discriminative power
        for i, (feat, _) in enumerate(discriminative):
            if feat not in rankings:
                rankings[feat] = []
            rankings[feat].append(i)
        
        # Method 2: Variance
        for i, (feat, _) in enumerate(variance):
            if feat not in rankings:
                rankings[feat] = []
            rankings[feat].append(i)
        
        # Method 3: Agreement
        for i, (feat, _) in enumerate(agreement):
            if feat not in rankings:
                rankings[feat] = []
            rankings[feat].append(i)
        
        # Method 4: Incremental value
        for i, (feat, _) in enumerate(incremental):
            if feat not in rankings:
                rankings[feat] = []
            rankings[feat].append(i)
        
        # Compute average rank for each feature
        feature_scores = {}
        for feat, rank_list in rankings.items():
            # Average rank across all methods that evaluated this feature
            avg_rank = np.mean(rank_list)
            feature_scores[feat] = avg_rank
        
        # Sort by average rank (lower = better)
        sorted_recommendations = sorted(feature_scores.items(), key=lambda x: x[1])
        
        print("\nFEATURES RANKED BY COMBINED ANALYSIS:")
        print("-"*80)
        print(f"{'Rank':<6} {'Feature':<25} {'Avg Rank':<12} {'Methods Used'}")
        print("-"*80)
        
        for i, (feat, avg_rank) in enumerate(sorted_recommendations, 1):
            n_methods = len(rankings[feat])
            print(f"{i:<6} {feat:<25} {avg_rank:<12.2f} {n_methods}/4 methods")
        
        # Create tiers based on actual different features
        print("\n" + "="*80)
        print("FINAL RECOMMENDATIONS")
        print("="*80)
        
        # More intelligent tier assignment
        n_features = len(sorted_recommendations)
        
        tiers = {
            'Essential (Must Have)': sorted_recommendations[:max(2, n_features//4)],
            'Important (Recommended)': sorted_recommendations[max(2, n_features//4):max(4, n_features//2)],
            'Optional (Nice to Have)': sorted_recommendations[max(4, n_features//2):max(6, 3*n_features//4)],
            'Low Priority': sorted_recommendations[max(6, 3*n_features//4):]
        }
        
        suggested_weights = {}
        
        for tier_name, features in tiers.items():
            if not features:  # Skip empty tiers
                continue
                
            print(f"\n{tier_name}:")
            for feat, rank in features:
                # Suggest weights based on tier - more gradual
                if tier_name == 'Essential (Must Have)':
                    weight = 2.5
                elif tier_name == 'Important (Recommended)':
                    weight = 1.5
                elif tier_name == 'Optional (Nice to Have)':
                    weight = 0.8
                else:
                    weight = 0.3
                
                suggested_weights[feat] = weight
                print(f"  - {feat:<25} (avg rank: {rank:.2f}, suggested weight: {weight:.1f})")
        
        # Generate specific configurations with DIFFERENT features per tier
        print("\n" + "="*80)
        print("SUGGESTED FEATURE CONFIGURATIONS")
        print("="*80)
        
        # Ensure each config has different features
        essential_features = [f for f, _ in sorted_recommendations[:max(1, n_features//5)]]
        balanced_features = [f for f, _ in sorted_recommendations[:max(3, 2*n_features//5)]]
        comprehensive_features = [f for f, _ in sorted_recommendations[:max(5, 3*n_features//5)]]
        
        configs = {
            'minimal': {
                'features': {f: suggested_weights[f] for f in essential_features},
                'description': 'Fastest, top 1-2 most important features'
            },
            'balanced': {
                'features': {f: suggested_weights[f] for f in balanced_features},
                'description': 'Good balance of speed and quality (top 3-4 features)'
            },
            'comprehensive': {
                'features': {f: suggested_weights[f] for f in comprehensive_features},
                'description': 'High quality, slower but more accurate (top 5-6 features)'
            }
        }
        
        for config_name, config in configs.items():
            print(f"\n{config_name.upper()}:")
            print(f"  {config['description']}")
            print(f"  Features ({len(config['features'])} total):")
            for feat, weight in config['features'].items():
                print(f"    - {feat}: {weight}")
        
        # Show detailed breakdown
        print("\n" + "="*80)
        print("DETAILED RANKING BREAKDOWN")
        print("="*80)
        print("\nHow each feature ranked in each method:")
        print(f"{'Feature':<25} {'Discriminative':<15} {'Variance':<12} {'Agreement':<12} {'Incremental':<12}")
        print("-"*80)
        
        # Create lookup dictionaries for each method
        disc_ranks = {f: i+1 for i, (f, _) in enumerate(discriminative)}
        var_ranks = {f: i+1 for i, (f, _) in enumerate(variance)}
        agr_ranks = {f: i+1 for i, (f, _) in enumerate(agreement)}
        inc_ranks = {f: i+1 for i, (f, _) in enumerate(incremental)}
        
        for feat, _ in sorted_recommendations[:10]:  # Show top 10
            disc = disc_ranks.get(feat, 'N/A')
            var = var_ranks.get(feat, 'N/A')
            agr = agr_ranks.get(feat, 'N/A')
            inc = inc_ranks.get(feat, 'N/A')
            
            disc_str = f"#{disc}" if disc != 'N/A' else disc
            var_str = f"#{var}" if var != 'N/A' else var
            agr_str = f"#{agr}" if agr != 'N/A' else agr
            inc_str = f"#{inc}" if inc != 'N/A' else inc
            
            print(f"{feat:<25} {disc_str:<15} {var_str:<12} {agr_str:<12} {inc_str:<12}")
        
        return configs
            
if __name__ == "__main__":
        analyzer = FeatureImportanceAnalyzer()
        
        print("\n" + "🔬" * 50)
        print("COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS")
        print("🔬" * 50)
        print("\nThis will run 4 different analyses to determine feature importance:")
        print("  1. Discriminative Power - Can it distinguish genres?")
        print("  2. Variance Contribution - How much do songs vary?")
        print("  3. Inter-Feature Agreement - Do features agree on similarity?")
        print("  4. Incremental Value - What does each feature add?")
        print("\nThis will take 2-3 minutes to complete.\n")
        
        input("Press Enter to start the analysis...")
        
        # Run comprehensive analysis
        recommendations = analyzer.generate_recommendations()
        
        print("\n" + "💡" * 50)
        print("HOW TO USE THESE RESULTS")
        print("💡" * 50)
        print("""
    UNDERSTANDING THE RANKINGS:

    1. AVERAGE RANK: Lower is better
    - Rank 0-3: Consistently top performer across all methods
    - Rank 4-7: Good performer in most methods
    - Rank 8+: Lower priority

    2. METHODS USED: X/4 methods
    - 4/4: Feature was evaluated by all methods (most reliable)
    - 3/4: Feature was evaluated by most methods (reliable)
    - 2/4 or less: Limited data (take with grain of salt)

    3. CONFIGURATIONS:
    - MINIMAL: Start here for fastest results, then expand if needed
    - BALANCED: Recommended for most use cases
    - COMPREHENSIVE: Use if quality matters more than speed

    NEXT STEPS:

    1. Try the 'balanced' configuration first
    2. Test with tracks you know well
    3. Adjust weights based on results:
    - If results sound too similar → Increase weights
    - If results sound random → Decrease weights or remove features
    4. Compare with 'minimal' and 'comprehensive' to find your sweet spot

    Remember: These are data-driven suggestions. Your ears are the final judge!
        """)
        
        # Offer to save configuration
        save = input("\nWould you like to save the recommended configurations to a file? (y/n): ")
        if save.lower() == 'y':
            import json
            
            output = {
                'configurations': {
                    name: {
                        'description': config['description'],
                        'features': config['features']
                    }
                    for name, config in recommendations.items()
                },
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            filename = 'recommended_feature_weights.json'
            with open(filename, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"\n✓ Configurations saved to: {filename}")
            print("\nYou can load these in your similarity search with:")
            print(f"  import json")
            print(f"  with open('{filename}') as f:")
            print(f"      configs = json.load(f)")
            print(f"      weights = configs['configurations']['balanced']['features']")
        analyzer = FeatureImportanceAnalyzer()

        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        print("\nThis will run multiple times to help you decide which features matter most.")
        print("Each method looks at the problem from a different angle.\n")

        # Run comprehensive analysis
        recommendations = analyzer.generate_recommendations()

        print("\n" + "="*50)
        print("HOW TO USE THESE RECOMMENDATIONS")
        print("="*50)
        print("""
            1. Start with the 'balanced' configuration
            2. Test it on tracks you know well
            3. Adjust based on what you hear: 
                - Results too similar in timbre but wrong mood? -> Increase chroma/tempo weights
                - Results right mood but wrong sound? -> increase mfcc/spectral weights
            4. Use these recommendations as a starting point, not gosped!
            
            Remmember: Your ears are the final judge!
            """)