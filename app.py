"""
Caleb Yonli Biometric Face ID Engine - Flask Backend API
InsightFace ArcFace 512-D ONNX Feature Extraction
Primary Subject: Caleb Yonli (ID: ID-001)
"""

import os
import sys

# Prevent OpenBLAS / thread contention on multi-core / Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import io
import time
import json
import pickle
import base64
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_DIR = os.path.join(BASE_DIR, "model")
EMBEDDINGS_FILE = os.path.join(MODEL_DIR, "face_embeddings.pkl")
ENCODINGS_JSON = os.path.join(MODEL_DIR, "encodings.json")
LABEL_MAP_FILE = os.path.join(MODEL_DIR, "label_map.json")

# Ensure template/static folders exist
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATES_DIR)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global Face Engine & Encodings Cache
face_app = None
face_db = None
stored_embeddings = None
stored_labels = None
stored_centroid = None
PRIMARY_NAME = "Caleb Yonli"
PRIMARY_ID = "ID-001"
THRESHOLD = 0.4800
SERVER_START_TIME = time.time()


def init_face_app():
    """Initialize InsightFace buffalo_sc engine with robust fallback paths."""
    global face_app
    import insightface
    from insightface.app import FaceAnalysis

    # Prefer workspace local model dir if available
    local_root = os.path.join(BASE_DIR, ".insightface")
    if os.path.exists(os.path.join(local_root, "models", "buffalo_sc")):
        root_path = local_root
    else:
        root_path = None

    print(f"📦 Initializing InsightFace ArcFace buffalo_sc (root={root_path})...")
    fa = FaceAnalysis(name="buffalo_sc", root=root_path, providers=["CPUExecutionProvider"])
    fa.prepare(ctx_id=0, det_size=(640, 640))
    face_app = fa
    print("✅ InsightFace Engine Ready!")


def load_database():
    """Load pre-computed 512-D face embeddings database."""
    global face_db, stored_embeddings, stored_labels, stored_centroid, PRIMARY_NAME, PRIMARY_ID, THRESHOLD

    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(EMBEDDINGS_FILE, "rb") as f:
                face_db = pickle.load(f)
            
            raw_embs = face_db.get("embeddings", [])
            if raw_embs:
                stored_embeddings = np.array(raw_embs, dtype=np.float32)
                # Ensure normalized vectors
                norms = np.linalg.norm(stored_embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                stored_embeddings = stored_embeddings / norms
                
                stored_labels = face_db.get("labels", ["Caleb_Yonli"] * len(raw_embs))
                PRIMARY_NAME = face_db.get("person_name", "Caleb Yonli")
                PRIMARY_ID = face_db.get("person_id", "ID-001")
                THRESHOLD = float(face_db.get("threshold", 0.4800))
                
                raw_c = face_db.get("centroid", None)
                if raw_c is not None:
                    stored_centroid = np.array(raw_c, dtype=np.float32)
                    stored_centroid /= np.linalg.norm(stored_centroid)
                
                print(f"✅ Loaded {len(stored_embeddings)} 512-D vectors for '{PRIMARY_NAME}' ({PRIMARY_ID})")
                return
        except Exception as e:
            print(f"⚠️ Error reading pickle database: {e}")

    # Fallback to encodings.json if present
    if os.path.exists(ENCODINGS_JSON):
        try:
            with open(ENCODINGS_JSON, "r", encoding="utf-8") as f:
                ej = json.load(f)
            PRIMARY_NAME = ej.get("person_name", "Caleb Yonli")
            PRIMARY_ID = ej.get("person_id", "ID-001")
            THRESHOLD = float(ej.get("threshold", 0.4000))
            vectors = ej.get("exemplars", [])
            if vectors:
                stored_embeddings = np.array(vectors, dtype=np.float32)
                norms = np.linalg.norm(stored_embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                stored_embeddings = stored_embeddings / norms
                stored_labels = ["Caleb_Yonli"] * len(vectors)
                print(f"✅ Loaded {len(stored_embeddings)} exemplars from encodings.json for {PRIMARY_NAME}")
                return
        except Exception as e:
            print(f"⚠️ Error reading JSON encodings: {e}")

    print("⚠️  Warning: No trained encodings database found! Please run train.py.")


def pil_to_bgr(pil_img, max_size=1280):
    w, h = pil_img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def recognize_face(image_input):
    """
    Core Biometric Recognition Pipeline:
    1. Detects faces with InsightFace RetinaFace/SCRFD
    2. Aligns and crops facial landmarks
    3. Extracts 512-D ArcFace feature vector
    4. Computes minimum Cosine Distance against enrolled Caleb Yonli vectors
    5. Returns rich JSON with bounding box, det_score, cosine_dist, confidence, PASS/FAIL
    """
    t0 = time.perf_counter()

    if face_app is None:
        init_face_app()

    if stored_embeddings is None or len(stored_embeddings) == 0:
        return {
            "matched": False,
            "status": "FAIL",
            "name": "Database Empty",
            "confidence": 0.0,
            "error": "NO ENROLLMENTS",
            "message": "Enrolled database is empty. Run train.py first."
        }

    try:
        if isinstance(image_input, Image.Image):
            pil_img = image_input
        else:
            pil_img = Image.open(image_input).convert("RGB")
        
        orig_w, orig_h = pil_img.size
        img_bgr = pil_to_bgr(pil_img)
        faces = face_app.get(img_bgr)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        if not faces:
            return {
                "matched": False,
                "status": "FAIL",
                "name": "Unknown Subject",
                "id": "UNAUTHORIZED",
                "confidence": 0.0,
                "det_score": 0.0,
                "cosine_dist": 1.0,
                "error": "NO FACE DETECTED",
                "message": "No human face detected in viewfinder. Adjust lighting and angle.",
                "image_size": [orig_w, orig_h],
                "total_encodings": len(stored_embeddings),
                "latency_ms": elapsed_ms
            }

        # Select most prominent / largest face in frame
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        # 512-D Normalized Vector
        q_emb = face.embedding / np.linalg.norm(face.embedding)

        # Cosine distance to all stored embeddings: 1.0 - (u . v)
        cosine_dists = 1.0 - np.dot(stored_embeddings, q_emb)
        min_dist = float(np.min(cosine_dists))
        best_idx = int(np.argmin(cosine_dists))
        best_label = stored_labels[best_idx] if stored_labels else "Caleb_Yonli"

        # Test-Time Mirror Invariance (handles front selfie camera vs rear lens flip)
        try:
            flipped_bgr = cv2.flip(img_bgr, 1)
            f_faces = face_app.get(flipped_bgr)
            if f_faces:
                f_face = max(f_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                f_q = f_face.embedding / np.linalg.norm(f_face.embedding)
                f_dists = 1.0 - np.dot(stored_embeddings, f_q)
                f_min = float(np.min(f_dists))
                if f_min < min_dist:
                    min_dist = f_min
                    q_emb = f_q
        except Exception:
            pass

        # Detection quality score
        det_score = round(float(getattr(face, "det_score", 0.95)) * 100, 1)
        
        # Scale bounding box back to original image dimensions if resized
        cur_h, cur_w = img_bgr.shape[:2]
        scale_x = orig_w / cur_w
        scale_y = orig_h / cur_h
        raw_box = face.bbox
        bbox = [
            max(0, int(raw_box[0] * scale_x)),
            max(0, int(raw_box[1] * scale_y)),
            min(orig_w, int(raw_box[2] * scale_x)),
            min(orig_h, int(raw_box[3] * scale_y))
        ]

        # Calculate high-accuracy confidence percentage
        if min_dist <= THRESHOLD:
            # Calibrated high confidence for confirmed Caleb Yonli match (80% - 99.9%)
            conf = max(80.0, min(99.9, 100.0 - (min_dist / THRESHOLD) * 20.0))
            is_match = True
            status_str = "PASS"
            display_name = PRIMARY_NAME
            subj_id = PRIMARY_ID
            msg = f"Biometric Match Confirmed: {PRIMARY_NAME} ({PRIMARY_ID})"
        else:
            conf = max(5.0, min(50.0, (1.0 - min_dist) * 100.0))
            is_match = False
            status_str = "FAIL"
            display_name = "Unknown Subject"
            subj_id = "UNAUTHORIZED"
            msg = "Subject not enrolled in biometric database"

        result = {
            "matched": is_match,
            "status": status_str,
            "name": display_name,
            "id": subj_id,
            "confidence": round(conf, 1),
            "cosine_dist": round(min_dist, 4),
            "det_score": det_score,
            "bbox": bbox,
            "image_size": [orig_w, orig_h],
            "total_encodings": len(stored_embeddings),
            "threshold": THRESHOLD,
            "message": msg,
            "latency_ms": elapsed_ms,
            "timestamp": int(time.time() * 1000)
        }
        return result

    except Exception as e:
        return {
            "matched": False,
            "status": "FAIL",
            "name": "Processing Error",
            "confidence": 0.0,
            "error": str(e),
            "message": f"Server processing error: {str(e)}"
        }


# ====================================================
# REST API ROUTES
# ====================================================

@app.route("/status", methods=["GET"])
def status():
    """Health & Biometric Model Status Endpoint"""
    uptime_sec = int(time.time() - SERVER_START_TIME)
    return jsonify({
        "status": "ready" if face_app and stored_embeddings is not None else "initializing",
        "system": "Caleb Yonli Biometric Face Recognition System",
        "primary_subject": PRIMARY_NAME,
        "primary_id": PRIMARY_ID,
        "model_architecture": "InsightFace ArcFace 512-D ONNX (buffalo_sc)",
        "total_encodings": len(stored_embeddings) if stored_embeddings is not None else 0,
        "known_persons": [PRIMARY_NAME],
        "threshold": THRESHOLD,
        "uptime_seconds": uptime_sec,
        "endpoints": {
            "/status": "GET - System health and model info",
            "/recognize": "POST - Multipart image recognition (file: image)",
            "/recognize_base64": "POST - Base64 JSON image recognition",
            "/enroll": "POST - Dynamic photo enrollment"
        }
    })


@app.route("/recognize", methods=["POST"])
def recognize():
    """Multipart Image Recognition Endpoint"""
    if "image" not in request.files:
        return jsonify({
            "matched": False,
            "status": "FAIL",
            "name": "Error",
            "error": "NO_IMAGE_UPLOADED",
            "message": "Missing 'image' file field in multipart form-data"
        }), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({
            "matched": False,
            "status": "FAIL",
            "name": "Error",
            "error": "EMPTY_FILENAME",
            "message": "Empty file uploaded"
        }), 400

    try:
        img = Image.open(file.stream).convert("RGB")
        res = recognize_face(img)
        return jsonify(res)
    except Exception as e:
        return jsonify({
            "matched": False,
            "status": "FAIL",
            "name": "Error",
            "error": "DECODE_ERROR",
            "message": f"Failed to parse uploaded image: {str(e)}"
        }), 400


@app.route("/recognize_base64", methods=["POST"])
def recognize_base64():
    """JSON Base64 Image Recognition Endpoint (for Live Stream Camera)"""
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({
            "matched": False,
            "status": "FAIL",
            "name": "Error",
            "error": "NO_IMAGE_DATA",
            "message": "Missing 'image' base64 string in JSON payload"
        }), 400

    try:
        raw_b64 = data["image"]
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(raw_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        res = recognize_face(img)
        return jsonify(res)
    except Exception as e:
        return jsonify({
            "matched": False,
            "status": "FAIL",
            "name": "Error",
            "error": "BASE64_DECODE_ERROR",
            "message": f"Failed to decode base64 image: {str(e)}"
        }), 400


@app.route("/enroll", methods=["POST"])
def enroll():
    """Dynamic Face Enrollment Endpoint"""
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400
    
    person_name = request.form.get("name", PRIMARY_NAME).strip()
    person_id = request.form.get("id", PRIMARY_ID).strip()
    file = request.files["image"]

    try:
        pil_img = Image.open(file.stream).convert("RGB")
        bgr = pil_to_bgr(pil_img)
        faces = face_app.get(bgr)
        if not faces:
            return jsonify({"success": False, "error": "No face found in photo"}), 400

        best_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        norm_emb = (best_face.embedding / np.linalg.norm(best_face.embedding)).tolist()

        global stored_embeddings, stored_labels
        new_emb = np.array(norm_emb, dtype=np.float32)
        if stored_embeddings is not None:
            stored_embeddings = np.vstack([stored_embeddings, new_emb])
            stored_labels.append(person_name.replace(" ", "_"))
        else:
            stored_embeddings = np.array([new_emb])
            stored_labels = [person_name.replace(" ", "_")]

        # Save to disk
        db_payload = {
            "embeddings": stored_embeddings.tolist(),
            "labels": stored_labels,
            "person_name": person_name,
            "person_id": person_id,
            "threshold": THRESHOLD
        }
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(db_payload, f)

        return jsonify({
            "success": True,
            "message": f"Successfully enrolled new sample for {person_name}",
            "total_encodings": len(stored_embeddings)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ====================================================
# STATIC FILE & PWA SERVING
# ====================================================

@app.route("/")
def index():
    # If index.html is in root or templates
    if os.path.exists(os.path.join(BASE_DIR, "index.html")):
        return send_from_directory(BASE_DIR, "index.html")
    return render_template("index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    # Check root workspace first (for PWA files, icons, style.css, app.js)
    if os.path.exists(os.path.join(BASE_DIR, filename)):
        return send_from_directory(BASE_DIR, filename)
    if os.path.exists(os.path.join(STATIC_DIR, filename)):
        return send_from_directory(STATIC_DIR, filename)
    return jsonify({"error": "Not Found"}), 404


# Initialize on import/start
init_face_app()
load_database()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print(f"⚡ Caleb Yonli Production Biometric Server Active")
    print(f"🌐 Local Address: http://localhost:{port}")
    print(f"📱 REST Endpoints: /status | /recognize | /recognize_base64")
    print(f"🔒 Server Engine: Production WSGI (Waitress Multi-Threaded)")
    print("=" * 60)
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=port, threads=6)
    except ImportError:
        import logging
        import flask.cli
        flask.cli.show_server_banner = lambda *args: None
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        app.run(debug=False, host="0.0.0.0", port=port)
