import pandas as pd
import numpy as np
from collections import defaultdict

class GenreCompatibility:
    def __init__(self):
        """Initialize genre compatibility system"""
        print("Loading genre data...")
        
        try:
            # Load tracks to get genre information
            self.tracks = pd.read_csv('fma_metadata/tracks.csv', index_col=0, header=[0, 1])
            print(f"✓ Loaded tracks: {len(self.tracks)} rows")
            
            self.genres_df = pd.read_csv('fma_metadata/genres.csv', index_col=0)
            print(f"✓ Loaded genres CSV: {len(self.genres_df)} rows")
            print(f"  Columns: {self.genres_df.columns.tolist()}")
            
            # Build genre hierarchy and relationships
            self._build_genre_graph()
            self._build_compatibility_matrix()
            
            print(f"✓ Initialization complete")
            
        except Exception as e:
            print(f"Error during initialization: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _build_genre_graph(self):
        """Build hierarchical genre relationships"""
        
        print("\nBuilding genre graph...")
        
        # Get all genres
        self.genre_list = self.genres_df.index.tolist()
        print(f"  Found {len(self.genre_list)} genres")
        
        # Build parent-child relationships from genre hierarchy
        self.genre_parents = {}
        self.genre_children = defaultdict(list)
        
        for idx, row in self.genres_df.iterrows():
            if 'parent' in row and pd.notna(row['parent']) and row['parent'] != 0:
                try:
                    parent_id = int(row['parent'])
                    self.genre_parents[idx] = parent_id
                    self.genre_children[parent_id].append(idx)
                except:
                    pass
        
        print(f"✓ Built {len(self.genre_parents)} parent-child relationships")
    
    def _build_compatibility_matrix(self):
        """Build compatibility scores between genres"""
        
        print("\nBuilding compatibility matrix...")
        
        # CRITICAL: Initialize as empty dict FIRST
        self.genre_names = {}
        print(f"  Initialized genre_names as: {type(self.genre_names)}")
        
        # Get genre titles (names)
        print("  Extracting genre names from dataframe...")
        for idx, row in self.genres_df.iterrows():
            if 'title' in row:
                self.genre_names[idx] = row['title']
        
        print(f"  After extraction, genre_names is: {type(self.genre_names)}")
        print(f"  Extracted {len(self.genre_names)} genre names")
        
        # Verify it's still a dict
        if not isinstance(self.genre_names, dict):
            print(f"ERROR: genre_names became {type(self.genre_names)} instead of dict!")
            print(f"Content: {self.genre_names}")
            raise TypeError("genre_names must be a dict")
        
        # Create reverse mapping (name -> id)
        print("  Creating reverse mapping...")
        self.genre_ids = {}
        for k, v in self.genre_names.items():
            self.genre_ids[v] = k
        
        print(f"✓ Created {len(self.genre_ids)} genre ID mappings")
        
        # Define known genre relationships
        self.genre_relationships = {
            # Rock family
            ('Rock', 'Pop'): 0.6,
            ('Rock', 'Indie-Rock'): 0.8,
            ('Rock', 'Alternative'): 0.7,
            ('Rock', 'Punk'): 0.7,
            ('Rock', 'Metal'): 0.6,
            
            # Electronic family
            ('Electronic', 'Techno'): 0.8,
            ('Electronic', 'House'): 0.8,
            ('Electronic', 'Ambient'): 0.7,
            ('Electronic', 'Experimental'): 0.6,
            ('Electronic', 'Dance'): 0.8,
            
            # Hip-Hop family
            ('Hip-Hop', 'Electronic'): 0.5,
            ('Hip-Hop', 'Pop'): 0.5,
            ('Hip-Hop', 'Soul-RnB'): 0.6,
            
            # Folk family
            ('Folk', 'Country'): 0.7,
            ('Folk', 'Rock'): 0.5,
            ('Folk', 'Indie-Rock'): 0.6,
            ('Folk', 'Blues'): 0.6,
            
            # Jazz/Classical
            ('Jazz', 'Classical'): 0.4,
            ('Jazz', 'Blues'): 0.7,
            ('Jazz', 'Soul-RnB'): 0.6,
            
            # Pop connections
            ('Pop', 'Electronic'): 0.6,
            ('Pop', 'Indie-Rock'): 0.5,
            
            # Experimental/Avant-garde
            ('Experimental', 'Avant-Garde'): 0.8,
            ('Experimental', 'Electronic'): 0.6,
        }
        
        # Make relationships symmetric
        symmetric_relationships = {}
        for (g1, g2), score in self.genre_relationships.items():
            symmetric_relationships[(g1, g2)] = score
            symmetric_relationships[(g2, g1)] = score
        
        self.genre_relationships = symmetric_relationships
        
        print(f"✓ Built {len(self.genre_relationships)} genre relationship pairs")
    
    def get_compatibility(self, genre1, genre2):
        """Calculate compatibility score between two genres"""
        
        # Handle same genre
        if genre1 == genre2:
            return 1.0
        
        # Handle None/missing genres
        if pd.isna(genre1) or pd.isna(genre2):
            return 0.5
        
        # Convert to strings if needed
        if isinstance(genre1, (int, np.integer)):
            genre1 = self.genre_names.get(genre1, str(genre1))
        if isinstance(genre2, (int, np.integer)):
            genre2 = self.genre_names.get(genre2, str(genre2))
        
        # Check manual relationships first
        if (genre1, genre2) in self.genre_relationships:
            return self.genre_relationships[(genre1, genre2)]
        
        # Check hierarchical relationships
        genre1_id = self.genre_ids.get(genre1)
        genre2_id = self.genre_ids.get(genre2)
        
        if genre1_id and genre2_id:
            # Check if one is parent of other
            if genre1_id in self.genre_parents and self.genre_parents[genre1_id] == genre2_id:
                return 0.8
            if genre2_id in self.genre_parents and self.genre_parents[genre2_id] == genre1_id:
                return 0.8
            
            # Check if siblings (same parent)
            if (genre1_id in self.genre_parents and genre2_id in self.genre_parents):
                if self.genre_parents[genre1_id] == self.genre_parents[genre2_id]:
                    return 0.6
        
        # Default: different genres
        return 0.2
    
    def get_compatible_genres(self, genre, min_compatibility=0.5):
        """Get all genres compatible with the given genre"""
        
        compatible = []
        
        # Get all unique genre names
        all_genres = set(self.genre_names.values())
        
        for other_genre in all_genres:
            if other_genre == genre:
                continue
            
            score = self.get_compatibility(genre, other_genre)
            
            if score >= min_compatibility:
                compatible.append((other_genre, score))
        
        # Sort by compatibility (highest first)
        compatible.sort(key=lambda x: x[1], reverse=True)
        
        return compatible
    
    def visualize_genre_network(self, genre, max_genres=10):
        """Show compatible genres in a readable format"""
        
        print(f"\n{'='*80}")
        print(f"GENRE COMPATIBILITY FOR: {genre}")
        print(f"{'='*80}\n")
        
        compatible = self.get_compatible_genres(genre, min_compatibility=0.4)
        
        if not compatible:
            print("No compatible genres found")
            return
        
        print(f"Found {len(compatible)} compatible genres\n")
        print(f"{'Rank':<6} {'Genre':<25} {'Compatibility':<15} {'Relationship'}")
        print("-" * 80)
        
        for i, (other_genre, score) in enumerate(compatible[:max_genres], 1):
            # Describe relationship
            if score >= 0.8:
                relationship = "Very Close"
            elif score >= 0.6:
                relationship = "Related"
            elif score >= 0.4:
                relationship = "Somewhat Related"
            else:
                relationship = "Distant"
            
            bar = "█" * int(score * 20)
            print(f"{i:<6} {other_genre:<25} {score:<15.2f} {bar} {relationship}")


if __name__ == "__main__":
    print("\n" + "+" * 50)
    print("GENRE COMPATIBILITY SYSTEM")
    print("+" * 50 + "\n")
    
    try:
        gc = GenreCompatibility()
        
        print("\n" + "="*80)
        print("VERIFICATION")
        print("="*80)
        print(f"genre_names type: {type(gc.genre_names)}")
        print(f"genre_names length: {len(gc.genre_names)}")
        print(f"genre_ids type: {type(gc.genre_ids)}")
        print(f"genre_ids length: {len(gc.genre_ids)}")
        print(f"\nSample genres: {list(gc.genre_names.values())[:10]}")
        
        # Test with some genres
        print("\n" + "="*80)
        print("TESTING GENRE COMPATIBILITY")
        print("="*80)
        
        test_genres = ['Rock', 'Electronic', 'Hip-Hop']
        
        for genre in test_genres:
            gc.visualize_genre_network(genre, max_genres=5)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()