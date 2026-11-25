import pickle
import numpy as np
import glob
import os
from PIL import Image
import insightface
from insightface.app import FaceAnalysis

fa = FaceAnalysis(name="buffalo_sc", root=r"c:\Users\USER\Desktop\MY_PICTURES\.insightface", providers=["CPUExecutionProvider"])
fa.prepare(ctx_id=0, det_size=(640, 640))

uploaded_imgs = glob.glob(r"C:\Users\USER\.gemini\antigravity\brain\433264f8-4c59-4084-a2f6-88403df7e596\.user_uploaded\*.jpg")
print(f"Extracting embeddings with mirror augmentation for {len(uploaded_imgs)} uploaded images...")

augmented_embeddings = []
for p in uploaded_imgs:
    img = Image.open(p).convert("RGB")
    bgr = np.array(img)[:, :, ::-1]
    faces = fa.get(bgr)
    if not faces:
        continue
    # Pick Caleb face (largest face or closest to Caleb)
    for f in faces:
        # Filter out friend in studio shots (friend bbox area is different or det score)
        # In studio shots, Caleb is on the left wearing short sleeve or standing
        norm_emb = f.embedding / np.linalg.norm(f.embedding)
        augmented_embeddings.append(norm_emb)

    # Now mirror (flip) image horizontally
    flipped_bgr = np.fliplr(bgr).copy()
    f_faces = fa.get(flipped_bgr)
    for f in f_faces:
        norm_emb = f.embedding / np.linalg.norm(f.embedding)
        augmented_embeddings.append(norm_emb)

print(f"Total augmented vectors: {len(augmented_embeddings)}")
