import pickle
import numpy as np
import glob
import os
import json

db_path = r"c:\Users\USER\Desktop\MY_PICTURES\model\face_embeddings.pkl"
with open(db_path, "rb") as f:
    db = pickle.load(f)

embs = np.array(db["embeddings"])
print(f"Total embeddings: {len(embs)}")
print(f"Person name: {db.get('person_name')}")
print(f"Person ID: {db.get('person_id')}")
print(f"Stored threshold: {db.get('threshold')}")

# Compute pairwise distances between all Caleb embeddings to see the real intra-class variance!
norms = np.linalg.norm(embs, axis=1, keepdims=True)
embs_norm = embs / norms

# Pairwise cosine similarity matrix
sim_matrix = np.dot(embs_norm, embs_norm.T)
dist_matrix = 1.0 - sim_matrix

print(f"Intra-class Cosine Distance Stats for Caleb:")
print(f"  Min dist (non-diagonal): {np.min(dist_matrix[dist_matrix > 0.0001]):.4f}")
print(f"  Mean dist: {np.mean(dist_matrix):.4f}")
print(f"  Median dist: {np.median(dist_matrix):.4f}")
print(f"  95th percentile dist: {np.percentile(dist_matrix, 95):.4f}")
print(f"  Max dist between Caleb photos: {np.max(dist_matrix):.4f}")
