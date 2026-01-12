import pandas as pd
import os

features = pd.read_csv('fma_metadata/features.csv', header=[0, 1, 2], index_col=0)
tracks = pd.read_csv('fma_metadata/tracks.csv', header=[0, 1], index_col=0)

print("Features shape:", features.shape)
print("Tracks shape:", tracks.shape)
print("\n" + "-"*50)
print("Features categories available:")
print(features.columns.get_level_values(0).unique())
print("\n" + "-"*50)
print("Sample track info:")
print(tracks.head())