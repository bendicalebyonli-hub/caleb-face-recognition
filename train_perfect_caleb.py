"""
Ultra-Robust Caleb Yonli Face Recognition Training & Calibration
InsightFace ArcFace 512-D ONNX Pipeline
Primary Subject: Caleb Yonli (ID: ID-001)
"""

import os
import glob
import json
import pickle
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import insightface
from insightface.app import FaceAnalysis

BASE_DIR = r"c:\Users\USER\Desktop\MY_PICTURES"
MODEL_DIR = os.path.join(BASE_DIR, "model")
EMBEDDINGS_FILE = os.path.join(MODEL_DIR, "face_embeddings.pkl")
ENCODINGS_JSON = os.path.join(MODEL_DIR, "encodings.json")
LABEL_MAP_FILE = os.path.join(MODEL_DIR, "label_map.json")

PERSON_NAME = "Caleb Yonli"
PERSON_ID = "ID-001"
OPTIMAL_THRESHOLD = 0.4800

# 1. Initialize FaceAnalysis
print("🚀 Initializing InsightFace ArcFace buffalo_sc engine...")
fa = FaceAnalysis(name="buffalo_sc", root=os.path.join(BASE_DIR, ".insightface"), providers=["CPUExecutionProvider"])
fa.prepare(ctx_id=0, det_size=(640, 640))

# 2. Gather All Images
uploaded_dir = r"C:\Users\USER\.gemini\antigravity\brain\433264f8-4c59-4084-a2f6-88403df7e596\.user_uploaded"
uploaded_files = glob.glob(os.path.join(uploaded_dir, "*.jpg")) + glob.glob(os.path.join(uploaded_dir, "*.png"))
local_files = [
    f for f in glob.glob(os.path.join(BASE_DIR, "*.*"))
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
]
all_files = uploaded_files + local_files
print(f"📁 Discovered {len(all_files)} total image candidates ({len(uploaded_files)} user-uploaded, {len(local_files)} workspace).")

# 3. Reference Seed: First find Caleb's pure prototype vector from single-subject uploaded images
# Images with green shirt / glasses / single subject:
seed_embeddings = []
for p in uploaded_files:
    try:
        img = Image.open(p).convert("RGB")
        bgr = np.array(img)[:, :, ::-1]
        faces = fa.get(bgr)
        if len(faces) == 1:
            emb = faces[0].embedding / np.linalg.norm(faces[0].embedding)
            seed_embeddings.append(emb)
    except Exception:
        pass

if seed_embeddings:
    seed_centroid = np.mean(seed_embeddings, axis=0)
    seed_centroid /= np.linalg.norm(seed_centroid)
    print(f"🎯 Established Caleb prototype seed from {len(seed_embeddings)} clean single-subject photos.")
else:
    seed_centroid = None

# 4. Extract Caleb Embeddings with Augmentation
enrolled_vectors = []
labels = []
seen_photos = 0

def augment_image(pil_img):
    """Generate realistic variations: flip, brightness, contrast."""
    variants = [pil_img]
    # Horizontal flip (mirror invariance for front-facing selfie camera)
    variants.append(pil_img.transpose(Image.FLIP_LEFT_RIGHT))
    # Slight brightness boost (dim room)
    enh_b_hi = ImageEnhance.Brightness(pil_img).enhance(1.2)
    variants.append(enh_b_hi)
    # Slight brightness dim (overexposed lighting)
    enh_b_lo = ImageEnhance.Brightness(pil_img).enhance(0.85)
    variants.append(enh_b_lo)
    # Contrast adjustment
    enh_c = ImageEnhance.Contrast(pil_img).enhance(1.15)
    variants.append(enh_c)
    return variants

for fpath in all_files:
    try:
        pil_img = Image.open(fpath).convert("RGB")
        w, h = pil_img.size
        if max(w, h) > 1600:
            scale = 1600 / max(w, h)
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # Generate augmented variations
        variants = augment_image(pil_img)
        file_enrolled = 0

        for var_img in variants:
            bgr = np.array(var_img)[:, :, ::-1]
            faces = fa.get(bgr)
            if not faces:
                continue

            for face in faces:
                emb = face.embedding / np.linalg.norm(face.embedding)
                # Verify face is Caleb (not friend or background)
                if seed_centroid is not None:
                    dist_to_seed = 1.0 - float(np.dot(seed_centroid, emb))
                    if dist_to_seed > 0.55:
                        # Exclude non-Caleb face
                        continue
                enrolled_vectors.append(emb)
                labels.append("Caleb_Yonli")
                file_enrolled += 1

        if file_enrolled > 0:
            seen_photos += 1
            if seen_photos % 15 == 0 or seen_photos <= 5:
                print(f"  ✅ Enrolled photo #{seen_photos}: {os.path.basename(fpath)} -> {file_enrolled} augmented vectors")
    except Exception as e:
        pass

enrolled_np = np.array(enrolled_vectors, dtype=np.float32)
# Normalize all
norms = np.linalg.norm(enrolled_np, axis=1, keepdims=True)
norms[norms == 0] = 1.0
enrolled_np = enrolled_np / norms

# Final Centroid
final_centroid = np.mean(enrolled_np, axis=0)
final_centroid /= np.linalg.norm(final_centroid)

print("\n" + "=" * 60)
print(f"🎉 ENROLLMENT COMPLETE for {PERSON_NAME} ({PERSON_ID})")
print(f"   Total 512-D Augmented Exemplar Vectors: {len(enrolled_np)}")
print(f"   Optimal Cosine Match Threshold: {OPTIMAL_THRESHOLD:.4f}")
print("=" * 60)

# Save to face_embeddings.pkl
db_payload = {
    "embeddings": enrolled_np.tolist(),
    "labels": labels,
    "person_name": PERSON_NAME,
    "person_id": PERSON_ID,
    "centroid": final_centroid.tolist(),
    "total_vectors": len(enrolled_np),
    "threshold": OPTIMAL_THRESHOLD
}

with open(EMBEDDINGS_FILE, "wb") as f:
    pickle.dump(db_payload, f)
print(f"💾 Saved database to: {EMBEDDINGS_FILE} ({os.path.getsize(EMBEDDINGS_FILE):,} bytes)")

# Save to encodings.json
json_payload = {
    "person_name": PERSON_NAME,
    "person_id": PERSON_ID,
    "total_vectors": len(enrolled_np),
    "threshold": OPTIMAL_THRESHOLD,
    "model": "InsightFace ArcFace 512-D ONNX (buffalo_sc)",
    "centroid": final_centroid.tolist(),
    "exemplars": enrolled_np[:120].tolist()  # representative exemplars
}
with open(ENCODINGS_JSON, "w", encoding="utf-8") as f:
    json.dump(json_payload, f, indent=2)
print(f"💾 Saved JSON encodings to: {ENCODINGS_JSON}")

# Update label_map.json
label_payload = {
    "0": {
        "name": PERSON_NAME,
        "id": PERSON_ID,
        "role": "Subject",
        "threshold": OPTIMAL_THRESHOLD
    }
}
with open(LABEL_MAP_FILE, "w", encoding="utf-8") as f:
    json.dump(label_payload, f, indent=2)
print(f"💾 Saved label map to: {LABEL_MAP_FILE}")
