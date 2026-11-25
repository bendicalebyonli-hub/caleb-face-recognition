"""
Caleb Yonli Biometric Face ID - Training & Encodings Pipeline
InsightFace ArcFace 512-D ONNX (buffalo_sc)
Subject: Caleb Yonli (ID: ID-001)
"""

import os
import sys
import glob
import json
import pickle
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

PERSON_NAME = "Caleb Yonli"
PERSON_ID = "ID-001"
LABEL_KEY = "Caleb_Yonli"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
EMBEDDINGS_FILE = os.path.join(MODEL_DIR, "face_embeddings.pkl")
ENCODINGS_JSON = os.path.join(MODEL_DIR, "encodings.json")
LABEL_MAP_FILE = os.path.join(MODEL_DIR, "label_map.json")

os.makedirs(MODEL_DIR, exist_ok=True)

def get_face_app():
    local_root = os.path.join(BASE_DIR, ".insightface")
    root_path = local_root if os.path.exists(local_root) else None
    app = FaceAnalysis(name="buffalo_sc", root=root_path, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def load_image(img_path, max_size=1280):
    pil_img = Image.open(str(img_path)).convert("RGB")
    w, h = pil_img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return bgr

def main():
    print("=" * 60)
    print(f"🚀 TRAINING PIPELINE: {PERSON_NAME} ({PERSON_ID})")
    print(f"   Architecture: InsightFace ArcFace 512-D ONNX")
    print("=" * 60)

    app = get_face_app()

    supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [
        f for f in sorted(Path(BASE_DIR).iterdir())
        if f.is_file() and f.suffix.lower() in supported_ext
    ]
    print(f"Found {len(image_paths)} images to train in {BASE_DIR}")

    embeddings = []
    labels = []

    for img_path in image_paths:
        try:
            bgr = load_image(img_path)
            faces = app.get(bgr)
            if faces:
                best_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                norm_emb = best_face.embedding / np.linalg.norm(best_face.embedding)
                embeddings.append(norm_emb.tolist())
                labels.append(LABEL_KEY)
                print(f"  ✅ Enrolled {img_path.name} (det={best_face.det_score:.2f})")
        except Exception as e:
            print(f"  ❌ Error in {img_path.name}: {e}")

    if not embeddings:
        print("⚠️ No faces detected. Exiting.")
        return

    all_vecs = np.array(embeddings)
    raw_c = np.mean(all_vecs, axis=0)
    centroid = raw_c / np.linalg.norm(raw_c)

    db_payload = {
        "embeddings": embeddings,
        "labels": labels,
        "person_name": PERSON_NAME,
        "person_id": PERSON_ID,
        "centroid": centroid.tolist(),
        "total_vectors": len(embeddings),
        "threshold": 0.40
    }

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(db_payload, f)

    label_map = {"0": LABEL_KEY, "display_name": PERSON_NAME, "id": PERSON_ID}
    with open(LABEL_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    json_enc = {
        "person_name": PERSON_NAME,
        "person_id": PERSON_ID,
        "threshold": 0.40,
        "total_enrolled": len(embeddings),
        "centroid": centroid.tolist(),
        "exemplars": embeddings[:35]
    }
    with open(ENCODINGS_JSON, "w", encoding="utf-8") as f:
        json.dump(json_enc, f, indent=2)

    print(f"\n🎉 Successfully trained & saved {len(embeddings)} 512-D vectors for {PERSON_NAME}!")

if __name__ == "__main__":
    main()
