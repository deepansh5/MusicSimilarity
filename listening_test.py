import pandas as pd
import numpy as np
from weighted_similarity import WeightedMusicSimilarity
import os

class ListeningTest:
    def __init__(self):
        self.searcher = WeightedMusicSimilarity()
        self.test_results = []

    def load_recommended_features(self, filepath='recommended_feature_weights.json'):
        """
        Load features from the analyzer
        """
        import json

        if not os.path.exists(filepath):
            print(f"Recommended weights file not found: {filepath}") 
            print("Using default feature list instead")
            return ['mfcc', 'chroma_cqt', 'spectral_centroid', 'spectral_contrast', 'tonnetz']
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Get all unique features from comprehensive config
            features = list(data['configurations']['comprehensive']['features'].keys())

            print(f"\nLoaded {len(features)} recommended features from analyzer")
            print(f"   Features: {features}")

            return features
        
        except Exception as e:
            print(f"Error loading recommendations: {e}")
            return ['mfcc', 'chroma_cqt', 'spectral_centroid', 'spectral_contrast', 'tonnetz']
    
    def single_feature_test(self, track_id, feature_name):
        """Test a single feature category"""

        print(f"\n{'='*80}")
        print(f"LISTENING TEST: {feature_name}")
        print(f"{'=' * 80}")

        # Prepare this feature only
        weights = {feature_name: 1.0}
        self.searcher.prepare_weighted_features(weights, name=f"test_{feature_name}")

        # Get results
        results = self.searcher.find_similar(track_id, f"test_{feature_name}", n_similar=5)

        if not results:
            print(f"No results for {feature_name}")
            return
        
        similar_ids, scores = results

        # Display
        query_info = self.searcher.get_track_info(track_id)
        print(f"Query: {query_info['title']} - {query_info['artist']}")
        print(f"\nTop 5 similar (using only {feature_name}):\n")

        for i, (tid, score) in enumerate(zip(similar_ids, scores), 1):
            info = self.searcher.get_track_info(tid)
            if info:
                print(f"{i}. [{score:.3f}] {info['title'][:40]:40s} - {info['artist'][:30]:30s}")
                print(f"      Track ID: {tid}  |  Genre: {info['genre']}")
        
        print(f"\n{'-'*80}")
        print("LISTENING INSTRUCTIONS:")
        print("1. Listen to query track")
        print("2. Listen to each of the 5 similar tracks")
        print("3. Rate how well this feature captured similarity")
        print("4. Note what aspect of similarity it captured (if any)")
        print(f"{'-'*80}\n")

        # Get user rating
        while True:
            try:
                rating = input(f"How well did {feature_name} work? (1=terrible, 5=excellent):")
                rating = int(rating)
                if 1 <= rating <= 5:
                    break
                print("Please enter a number between 1 and 5")
            except: 
                print("Please enter a number")
        
        notes = input("What did this feature capture? (timbre/harmony/rhythm/mood/other): ")

        self.test_results.append({
            'feature': feature_name,
            'rating': rating,
            'notes': notes
        })

        return rating
    
    def comparative_test(self, track_id, features_to_compare):
        """Compare multiple features side-by-side"""

        print(f"\n{'='*80}")
        print(f"COMPARATIVE LISTENING TEST")
        print(f"{'='*80}\n")

        query_info = self.searcher.get_track_info(track_id)
        print(f"Query: {query_info['title']} - {query_info['artist']}")
        print(f"Genre: {query_info['genre']}\n")

        # Prepare all features
        for feat in features_to_compare:
            weights = {feat:1.0}
            self.searcher.prepare_weighted_features(weights, name=f"test_{feat}")

            # Get top 3 from each
            print("TOP # RECOMMENDATIONS FROM EACH FEATURE: ")
            print("="*80)

            for feat in features_to_compare:
                results = self.searcher.find_similar(track_id, f"test_{feat}", n_similar=3)

                if results:
                    print(f"\n{feat.upper()}:")
                    similar_ids, scores = results
                    
                    for i, (tid, score) in enumerate(zip(similar_ids, scores), 1):
                        info = self.searcher.get_track_info(tid)
                        if info:
                            print(f"{i}. {info['title'][:35]:35s} - {info['artist'][:25]:25s}")

            print(f"\n{'='*80}")
            print("Which feature gave the BEST results overall?")

            for i, feat in enumerate(features_to_compare, 1):
                print(f"{i}. {feat}")
            
            while True:
                try:
                    choice = input(f"\nEnter number (1-{len(features_to_compare)}):")
                    choice = int(choice)
                    if 1 <= choice <= len(features_to_compare):
                        winner = features_to_compare[choice - 1]
                        break
                except:
                    pass
            
            print(f"\nYou chose: {winner}")

            return winner
    
    def generate_report(self):
        """Summary of listening test results"""
        if not self.test_results:
            print("No test results yet!")
            return
        
        print(f"\n{'='*80}")
        print("LISTENING TEST REPORT")
        print(f"{'='*80}\n")

        # Sort by rating
        sorted_results = sorted(self.test_results, key=lambda x:x['rating'], reverse=True)

        print("FEATURES RANKED BY YOUR RATING:")
        print("-"*80)

        for i, result in enumerate(sorted_results, 1):
            stars = "*" * result['rating'] + "^" * (5 - result['rating'])
            print(f"{i}. {result['rating']:20s} {stars} ({result['rating']}/5)")
            if result['notes']:
                print(f"     Notes: {result['notes']}")
            print()

        #Suggested weights based on ratings
        print("SUGGESTED WEIGHTS (based on your ratings):")
        print("-"*80)

        max_rating = max(r['rating'] for r in self.test_results)

        for result in sorted_results:
            # Normallize ratings to weights (1-5 rating -> 0.2-2.0 weight)
            weight = (result['rating'] / max_rating) * 2.0
            print(f"{result['feature']:20s}: {weight:.1f}")

        # Generate config
        print(f"\n{"="*80}")
        print("PERSONALIZED CONFIGURATION:")
        print(f"{"="*80}\n")

        print("weight_config = {")
        for result in sorted_results:
            weight = (result['rating'] / max_rating) * 2.0
            print(f"      '{result['feature']}: {weight:.1f},")
        print("}")

        # Save to file
        print(f"\n{"="*80}")
        save = input("\nWould you like to save your personalized weights to a file (y/n)?")

        if save.lower() == 'y':
            import json

            personalized_weights = {}
            for result in sorted_results:
                weight = (result['rating'] / max_rating) * 2.0
                personalized_weights[result['feature']] = weight

            output = {
                'configurations' : {
                    'personalized': {
                        'description': 'Based on your listening test ratings',
                        'features': personalized_weights
                    }
                },
                'test_results': self.test_results,
                'timestamp': pd.Timestamp.now().isoformat()
            } 

            filename = 'personalized_feature_weights.json'
            with open(filename, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"\nPersonalized weights saved to: {filename}")
            print("\nYou can load these in weighted_similarity.py:")
            print(f"    searcher.load_weights_from_file('{filename}', 'personalized')")

if __name__ == "__main__":
    tester = ListeningTest()

    print("\n" + "=" * 50)
    print("PERSONALIZED FEATURE IMPORTANCE TEST")
    print("="*50)

    print("""
This tool helps you determine which features matter most TO YOU.
          
HOW IT WORKS:
          1. Pick a track you know well
          2. Test each feature individually
          3. Listen to the recommendations
          4. Rate how good each feature is
          5. Get personalized weight recommendations

NOTE: You'll need to acutally listen to the tracks!
      The track files are in fma_small/xxx/track_id.mp3
""")
    
    # Check if analyzer results exists
    if os.path.exists('recommended_feature_weights.json'):
        print("\nFound results from feature_analyzer.py")
        use_recommended = input("Use recommended features from analyzer (y/n) [default: y]?").strip.lower()

        if use_recommended != 'n':
            features_to_test = tester.load_recommended_features()
        else:
            features_to_test = ['mfcc', 'chroma_qrt','spectral_centroid', 'spectral_contrast', 'tonnetz']
    else:
        print(f"\n TIP: Run feature_importance_analyzer.py first to get data-driven recommendations")
        features_to_test = ['mfcc', 'chroma_cqt', 'spectral_centroid', 'spectral_contrast', 'tonnetz']
    
    print(f"\nWill test these {len(features_to_test)} features: {features_to_test}")

    #Get test track
    while True:
        try:
            track_id = int(input("\nEnter a track ID to test with (e.g. 2): "))
            info = tester.searcher.get_track_info(track_id)
            if info:
                print(f"\nSelected: {info['title']} - {info['artist']}")
                confirm = input("Is this correct? (y/n): ")
                if confirm.lower() == 'y':
                    break
                else:
                    print(f"Track {track_id} not found")
        except:
            print("Please enter a valid track ID")
    
    # Test individual features
    print("\n" + "-"*80)
    print("TESTING INDIVIDUAL FEATURES")
    print("="*80)

    for feat in features_to_test:
        tester.single_feature_test(track_id, feat)
    
    # Generate report
    tester.generate_report()

    # Compare with analyzer recommendations (if available)
    if os.path.exists('recommneded_feature_json', 'r'):
        print("\n" + "="*80)
        print("COMPARISON: Your Ratings vs Analyzer Recommendations")
        print("="*80)

        import json
        with open('recommended_feature_weight', 'r') as f:
            analyzer_data = json.load(f)

        analyzer_weights = analyzer_data['configuration']['balanced']['features']

        print(f"\n{'Feature'<25} {'Analyzer':<18} {'Your Rating':<15} {'Agreement'}")

        for result in tester.test_results:
            feat = result['feature']
            your_rating = result['rating']
            analyzer_weight = analyzer_weights.get(feat, 0)

            # Normalize to same scale for comparison
            # Analyzer: 0.3-2.5 -> Your rating: 1-5
            # Let's convert anaylzer to 1-5 scale
            if analyzer_weight > 0:
                normalized_analyzer = ((analyzer_weight - 0.3) / (2.5 - 0.3)) * 4 + 1
            else:
                normalized_analyzer = 0
            
            difference = abs(your_rating - normalized_analyzer)

            if difference < 1:
                agreement = "Strong"
            elif difference < 2:
                agreement = "Moderate"
            else:
                agreement = "Weak"
            
            analyzer_str = f"{analyzer_weight:.1f}" if analyzer_weight > 0 else "N/A"
            print(f"{feat:<25} {analyzer_str:<18} {your_rating:<15}/5 {agreement}")

            print("\nINTERPRETATION")
            print("- Strong agreement: Data and perception align (trust these!)")
            print("- Moderate agreement: Some difference (use your judgment)")
            print("- Weak agreement: Big difference (trust your ears or retest)")
    

    print("\n Testing complete!")
    print("\nNEXT STEPS:")
    print("1. Use your personalized weights in weighted_similarity.py")
    print("2. Or run weighted_similarity.py and load the saved configuration")
    print("3. Compare results with different weight schemes")



