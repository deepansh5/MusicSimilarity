Music Similarity Search System
A learning project for building a music recommendation system using audio feature analysis and machine learning techniques.

📋 Project Overview
This project explores how music similarity works by analyzing audio features from the Free Music Archive (FMA) dataset. It's designed as an educational journey through the concepts used by professional music recommendation systems like Spotify and Pandora.
What This System Does

Analyzes audio features (timbre, harmony, rhythm, energy) from music tracks
Finds songs that sound similar based on these features
Learns which features matter most through data-driven analysis
Allows you to customize similarity based on your preferences
Goal: Extract and compare features from any music files (not just the dataset)

What You'll Learn

Audio signal processing and feature extraction
Multi-dimensional similarity metrics
Machine learning for recommendation systems
The gap between technical and perceptual similarity
Why production systems use hybrid approaches (audio + metadata + collaborative filtering)


🎯 Current Status: Phase 1 Complete
What's Working:

✅ Data-driven feature importance analysis
✅ Weighted similarity search with configurable features
✅ Multiple preset configurations (minimal/balanced/comprehensive)
✅ Manual weight customization
✅ Perceptual validation through listening tests

Current Performance:

~3-5 out of 10 results feel similar (expected for audio-only features)
Cross-genre matching works (finds similar sound regardless of genre labels)
Consistent results across configurations

Key Learnings:

Audio features capture perceptual/mood similarity, not necessarily musical structure
Pure audio features have inherent limitations (don't capture melody, rhythm patterns, song structure)
Professional systems combine audio features with metadata and collaborative filtering


📁 Project Structure
music_similarity/
├── fma_metadata/                      # FMA dataset metadata
│   ├── tracks.csv                     # Track information
│   ├── features.csv                   # Pre-computed audio features
│   └── genres.csv                     # Genre taxonomy
│
├── fma_small/                         # Audio files (not included in repo)
│   └── [000-155]/                     # ~8000 30-second MP3 clips
│
├── feature_importance_analyzer.py     # Data-driven feature analysis
├── weighted_similarity.py             # Main similarity search system
├── listening_test.py                  # Perceptual validation tool
├── test_results_tracker.py           # Systematic testing framework
│
├── recommended_feature_weights.json   # Auto-generated recommendations
├── personalized_feature_weights.json  # Your custom weights (if created)
├── similarity_test_results.csv        # Test results log
│
└── README.md                          # This file

🚀 Getting Started
Prerequisites
Python 3.8+ with the following libraries:
bashpython -m pip install numpy pandas scikit-learn librosa matplotlib
```

**Required libraries:**
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning tools (PCA, classifiers, similarity metrics)
- `librosa` - Audio feature extraction
- `matplotlib` - Visualization (optional, for future phases)

### Dataset Setup

1. **Download FMA Small dataset:**
   - Metadata: https://os.unil.cloud.switch.ch/fma/fma_metadata.zip (~342 MB)
   - Audio: https://os.unil.cloud.switch.ch/fma/fma_small.zip (~7.2 GB)

2. **Extract to project directory:**
```
   music_similarity/
   ├── fma_metadata/
   └── fma_small/

Verify setup:

bash   # Should see tracks.csv, features.csv, genres.csv
   dir fma_metadata
   
   # Should see folders 000-155
   dir fma_small

🎵 How to Use
Step 1: Analyze Feature Importance
Determine which audio features are most informative:
bashpython feature_importance_analyzer.py
```

**What it does:**
- Runs 4 different analyses (discriminative power, variance, agreement, incremental value)
- Ranks features by importance
- Generates 3 recommended configurations (minimal/balanced/comprehensive)
- Saves results to `recommended_feature_weights.json`

**Output:**
```
FINAL RECOMMENDATIONS
════════════════════════════════════════════════════════════════════════════════

Essential (Must Have):
  - mfcc                      (avg rank: 1.50, suggested weight: 2.5)
  - spectral_contrast         (avg rank: 3.25, suggested weight: 2.5)

Important (Recommended):
  - chroma_cqt               (avg rank: 4.75, suggested weight: 1.5)
  - spectral_centroid        (avg rank: 5.50, suggested weight: 1.5)

...
Time: 2-3 minutes

Step 2: Test Similarity Search
Use the recommended weights to find similar songs:
bashpython weighted_similarity.py
```

**What it does:**
- Auto-detects `recommended_feature_weights.json`
- Offers minimal/balanced/comprehensive configurations
- Finds 10 similar songs for a given track
- Displays results with track info

**Example usage:**
```
Which configuration would you like to use?
  1. minimal - Fastest (1-2 features)
  2. balanced - Recommended (3-4 features)
  3. comprehensive - Highest quality (5-6 features)
  4. manual - Define your own weights

Enter choice (1-4) [default: 2]: 2

Enter track ID to test: 2

SIMILARITY SEARCH: Track 2
Feature weights: {'mfcc': 2.5, 'chroma_cqt': 1.5, ...}
================================================================================

Query Track:
  Title:  Food
  Artist: Electric Aura
  Genre:  Experimental

Top 10 Similar Tracks:
--------------------------------------------------------------------------------
 1. [Similarity: 0.892] Track 134
    AWOL
    The Golden Leers | Genre: Rock

 2. [Similarity: 0.847] Track 98
    Sunset
    ...
Time: A few seconds per search

Step 3 (Optional): Perceptual Validation
Validate the analyzer's recommendations with your own ears:
bashpython listening_test.py
What it does:

Tests each feature individually
Shows you top 5 similar songs using ONLY that feature
You rate how well it worked (1-5 stars)
Compares your ratings with analyzer's recommendations
Generates personalized weights based on your preferences

Use when:

You want deeply personalized weights
Analyzer recommendations don't match your perception
You have time to listen and rate (2-3 hours)


Step 4: Systematic Testing
Track performance across multiple tracks and genres:
bashpython test_results_tracker.py
```

**What it does:**
- Suggests diverse test tracks (different genres)
- Lets you test and rate results (0-10 similar)
- Tracks results across multiple tests
- Shows summary statistics by genre and config
- Saves results to `similarity_test_results.csv`

**Recommended:**
- Test 5-6 tracks from different genres
- Use same config for consistency (balanced recommended)
- Note patterns (which genres work better)

**Example output:**
```
TESTING SUMMARY
================================================================================

Total tests: 6
Average score: 0.45
Average similar: 4.5/10

By Configuration:
        mean  count
config              
balanced  0.45      6

By Genre:
             mean  count
track_genre              
Electronic   0.60      2
Hip-Hop      0.40      2
Rock         0.35      2

🧠 Understanding the Features
Audio Features Explained
MFCC (Mel-Frequency Cepstral Coefficients)

Captures: Timbre, texture, "color" of sound
Example: Distinguishes piano from guitar
Use when: You care about instrumentation similarity

Chroma (CQT, STFT, CENS)

Captures: Harmony, key, chord progressions
Example: Songs in C major vs D minor
Use when: You care about harmonic similarity

Spectral Features (Centroid, Contrast, Rolloff, Bandwidth)

Captures: Brightness, energy, frequency distribution
Example: Bright/sparkly vs dark/warm sounds
Use when: You care about sonic texture

Temporal Features (Tempo, Beat, Rhythm)

Captures: Speed, groove, rhythmic patterns
Example: Fast dance track vs slow ballad
Use when: You care about energy/movement

Tonnetz

Captures: Tonal relationships, harmonic space
Example: How chords relate to each other
Use when: You care about music theory relationships


🎛️ Configuration Guide
Preset Configurations
Minimal (1-2 features)

Fastest computation
Single most important feature
Use for: Quick searches, large-scale processing
Expected results: 2-3/10 similar

Balanced (3-4 features)

Recommended starting point
Good speed/quality tradeoff
Captures multiple dimensions (timbre + harmony + texture)
Expected results: 4-5/10 similar

Comprehensive (5-6 features)

Highest quality
Slower computation
Captures many aspects of similarity
Expected results: 5-6/10 similar

Custom Weights
You can define your own weights based on what matters to you:
pythonfrom weighted_similarity import WeightedMusicSimilarity

searcher = WeightedMusicSimilarity()

# Example: Emphasize timbre over harmony
custom_weights = {
    'mfcc': 3.0,              # Very important
    'spectral_centroid': 2.0,  # Important
    'chroma_cqt': 0.5,        # Less important
}

searcher.prepare_weighted_features(custom_weights, name='my_config')
searcher.display_results(track_id=2, feature_set_name='my_config', n_similar=10)
```

**Weight interpretation:**
- `3.0` = Very important (3x emphasis)
- `1.5` = Moderately important
- `1.0` = Standard importance
- `0.5` = Less important (half emphasis)
- `0.2` = Minimal importance

---

## 📊 Performance Expectations

### Realistic Results

**With audio features only:**
- Good: 4-5/10 similar
- Excellent: 6-7/10 similar
- Perfect (10/10): Nearly impossible

**Why?**
- Audio features capture perceptual similarity (how it sounds)
- Don't capture musical structure (melody, rhythm patterns, song form)
- Don't understand genre conventions or cultural context

### What the System Does Well

✅ **Cross-genre matching** - Finds similar sound regardless of genre labels  
✅ **Timbre/texture similarity** - Songs with similar instrumentation  
✅ **Mood/vibe matching** - Songs with similar energy/atmosphere  
✅ **Harmonic similarity** - Songs in same key or with similar chords  

### What the System Struggles With

❌ **Melody similarity** - Can't detect similar tunes  
❌ **Rhythmic patterns** - Doesn't capture groove or syncopation  
❌ **Song structure** - Doesn't understand verse-chorus patterns  
❌ **Genre conventions** - Might match across incompatible genres  
❌ **Lyrical themes** - No text/semantic understanding  

---

## 🔬 Technical Details

### Feature Weighting Mathematics

**Why we use `sqrt(weight)`:**

In cosine similarity, features are combined via dot product:
```
similarity = (A · B) / (||A|| × ||B||)
where A · B = sum(a_i × b_i)
```

If we scale by weight directly:
```
scaled = features × weight
similarity involves: (features × weight) × (features × weight) = features² × weight²
```

So we use sqrt to get linear weighting:
```
scaled = features × sqrt(weight)
similarity involves: (features × sqrt(weight))² = features² × weight  ✓
Similarity Metrics
Cosine Similarity:

Measures angle between feature vectors
Range: -1 (opposite) to +1 (identical)
Invariant to magnitude (only direction matters)
Used because: Ignores loudness, focuses on characteristics

Alternatives considered:

Euclidean distance: Sensitive to magnitude, not suitable for audio
Manhattan distance: Similar issues
Mahalanobis distance: More complex, requires covariance matrix


🚧 Current Limitations
Known Issues

Audio features alone aren't enough

Need metadata (genre, era) for better results
Professional systems use collaborative filtering


Dataset limitations

Only 8,000 tracks (limited diversity)
30-second clips (not full songs)
Pre-computed features (can't add custom ones easily)


No melody/rhythm analysis

Features don't capture these aspects well
Would need specialized algorithms


Genre imbalance

Some genres have more tracks than others
Results may be biased toward common genres



Workarounds
For better results:

Use balanced or comprehensive config
Filter by genre (Phase 2)
Add metadata weighting (Phase 2)
Provide user feedback (Phase 3)


🗺️ Roadmap
✅ Phase 1: Feature Weighting (Complete)

 Data-driven feature analysis
 Weighted similarity search
 Preset configurations
 Perceptual validation
 Systematic testing framework

🔄 Phase 2: Hybrid Similarity (Next)
Goal: Improve from 4-5/10 to 6-7/10
Features to add:

Genre compatibility scoring
Era/decade similarity
Artist similarity
Weighted hybrid scoring

Estimated time: 4-6 hours
🎯 Phase 3: Feature Extraction Pipeline (Main Goal)
Goal: Apply to YOUR music files
Features to build:

Extract features from any MP3 using librosa
Normalize features to match FMA dataset
Compare your songs to FMA database
Compare your own songs to each other
Build reusable CLI tool

Estimated time: 4-5 hours
🔮 Phase 4: Learning & Refinement (Optional)
Features:

User feedback collection
Online weight adjustment
A/B testing framework
Evaluation metrics

Estimated time: 3-5 hours

📚 Learning Resources
Concepts Covered
Audio Signal Processing:

Short-Time Fourier Transform (STFT)
Mel-Frequency Cepstrum
Chromagram analysis
Spectral features

Machine Learning:

Feature engineering
Dimensionality reduction (PCA)
Similarity metrics
Multi-criteria decision making
Hybrid systems

Music Information Retrieval:

Audio feature extraction
Music similarity
Content-based recommendation
Genre classification

Further Reading
Academic Papers:

"The FMA: A Dataset for Music Analysis" (Defferrard et al., 2016)
"Music Similarity Estimation" (Logan & Salomon, 2001)

Libraries:

librosa documentation: https://librosa.org/
scikit-learn user guide: https://scikit-learn.org/

Datasets:

FMA: https://github.com/mdeff/fma
Million Song Dataset: http://millionsongdataset.com/


🤝 Contributing / Extending
This is a personal learning project, but you can extend it:
Ideas for Extensions

Add more features:

Extract tempo/beat features
Add loudness dynamics
Compute melody contours


Improve similarity:

Try different distance metrics
Implement collaborative filtering
Add content-based filtering with lyrics


Build interfaces:

Web UI for searching
CLI tool for batch processing
API for integration


Add visualizations:

t-SNE plots of feature space
Genre distributions
Similarity networks




📝 Version History
v1.0 - Phase 1 Complete (Current)

Feature importance analyzer with 4 analysis methods
Weighted similarity search with 3 preset configs
Listening test for perceptual validation
Systematic testing framework
Documentation and learnings

Upcoming:

v2.0 - Phase 2: Hybrid similarity with metadata
v3.0 - Phase 3: Feature extraction pipeline
v4.0 - Phase 4: Learning and refinement


🙏 Acknowledgments
Datasets:

Free Music Archive (FMA) - Defferrard et al.
Creative Commons licensed music

Libraries:

librosa - Audio analysis
scikit-learn - Machine learning
pandas - Data manipulation

Inspiration:

Spotify's recommendation system
Pandora's Music Genome Project
Academic MIR research


📄 License
This is a personal learning project. The code is provided as-is for educational purposes.
FMA Dataset License: Creative Commons (see FMA documentation)

📧 Contact / Questions
This is a solo learning project, but feel free to:

Document your own learnings
Extend the code for your purposes
Share insights about what worked/didn't work


🎓 Key Takeaways
What I Learned

Audio features have limitations

Can't capture everything that makes music similar
Perceptual ≠ Musical similarity
Need hybrid approaches for production systems


Data-driven vs perceptual validation

Statistics don't always match human perception
Both perspectives are valuable
Iterative testing is essential


Feature engineering matters

Different features capture different aspects
Weighting is crucial
No single "best" configuration


Realistic expectations

4-5/10 similar is good for audio-only
Professional systems use many signals
Perfect similarity is impossible



Next Steps

 Complete systematic testing (5-6 tracks)
 Decide on best configuration for my taste
 Move to Phase 2 (hybrid similarity)
 Build feature extraction pipeline (Phase 3)
 Apply to my own music library