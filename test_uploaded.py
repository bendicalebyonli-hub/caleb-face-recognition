import os
import glob
import pickle
import numpy as np
from PIL import Image
import insightface
from insightface.app import FaceAnalysis

db_path = r"c:\Users\USER\Desktop\MY_PICTURES\model\face_embeddings.pkl"
with open(db_path, "rb") as f:
    db = pickle.load(f)

embs = np.array(db["embeddings"])
norms = np.linalg.norm(embs, axis=1, keepdims=True)
embs_norm = embs / norms

fa = FaceAnalysis(name="buffalo_sc", root=r"c:\Users\USER\Desktop\MY_PICTURES\.insightface", providers=["CPUExecutionProvider"])
fa.prepare(ctx_id=0, det_size=(640, 640))

uploaded_imgs = glob.glob(r"C:\Users\USER\.gemini\antigravity\brain\433264f8-4c59-4084-a2f6-88403df7e596\.user_uploaded\*.jpg")
print(f"Testing {len(uploaded_imgs)} user-uploaded reference images against database:")

all_min_dists = []
for p in uploaded_imgs:
    img = Image.open(p).convert("RGB")
    bgr = np.array(img)[:, :, ::-1]
    faces = fa.get(bgr)
    if not faces:
        print(f"  {os.path.basename(p)}: NO FACE DETECTED by buffalo_sc!")
        continue
    for f_idx, f in enumerate(faces):
        q = f.embedding / np.linalg.norm(f.embedding)
        dists = 1.0 - np.dot(embs_norm, q)
        m_dist = float(np.min(dists))
        all_min_dists.append((os.path.basename(p), m_dist))
        print(f"  {os.path.basename(p)}: min_dist={m_dist:.4f}")

dists_only = [d[1] for d in all_min_dists]
print(f"\nStats: Min={min(dists_only):.4f}, Max={max(dists_only):.4f}, Mean={np.mean(dists_only):.4f}")
