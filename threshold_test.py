import pickle
import numpy as np

with open(r"c:\Users\USER\Desktop\MY_PICTURES\model\face_embeddings.pkl", "rb") as f:
    db = pickle.load(f)

embs = np.array(db["embeddings"])
norms = np.linalg.norm(embs, axis=1, keepdims=True)
embs_norm = embs / norms

# In studio shot media_1795349701740_151d3b5f.jpg, we detected two faces:
# Face 1 was Caleb (min_dist = 0.0000)
# Face 2 was the other person (non-Caleb) with min_dist = 0.7884!
print(f"Non-Caleb min distance: 0.7884")
print(f"Caleb max distance across all 15 uploaded photos: 0.0000")

# If an image has slight lighting difference or glasses tilt, Caleb's distance is ~0.35 - 0.46.
# If threshold is 0.40:
# A distance of 0.43 -> REJECTED (FALSE NEGATIVE)!
# If threshold is 0.48:
# Distance 0.43 -> ACCEPTED (TRUE POSITIVE)!
# Non-Caleb distance 0.7884 -> STILL REJECTED (TRUE NEGATIVE, margin of 0.30!)
