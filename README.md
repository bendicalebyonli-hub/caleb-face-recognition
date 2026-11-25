# Caleb Yonli · Mobile Biometric Face Recognition System (PWA v3.0)

[![Engine](https://img.shields.io/badge/Model-InsightFace%20ArcFace%20512--D-00f2fe.svg)](#)
[![Primary Subject](https://img.shields.io/badge/Enrolled-Caleb%20Yonli%20(ID--001)-00f5a0.svg)](#)
[![PWA](https://img.shields.io/badge/PWA-Installable%20Mobile%20App-ff0055.svg)](#)
[![Deployment](https://img.shields.io/badge/Deploy-Render%20%2B%20Vercel-blue.svg)](#)

A mobile web application (Progressive Web App) and REST API for **Caleb Yonli (ID: ID-001)** biometric facial recognition, powered by **InsightFace ArcFace 512-D ONNX** deep feature extraction.

---

## ⚡ System Architecture

```
MY_PICTURES/
├── index.html                 # Mobile Web App Shell (PWA)
├── style.css                  # Dark Cyberpunk HUD & Mobile Theme
├── app.js                     # Mobile Engine (Camera HUD, Auto-Scan, Auto-Log, Sound, JSON)
├── sw.js                      # Service Worker (Offline caching & PWA)
├── manifest.webmanifest       # PWA Manifest (Name, theme, icons)
├── manifest.json              # Fallback Manifest
├── vercel.json                # Vercel Frontend Deployment Configuration
├── render.yaml                # Render Backend Service Configuration
├── Procfile                   # Gunicorn WSGI Start Command
├── requirements.txt           # Python Production Dependencies
├── app.py                     # Flask REST API (/status, /recognize, /recognize_base64, /enroll)
├── train.py                   # Auto-labeling & 512-D ArcFace Training Pipeline
│
├── model/                     # Enrolled Biometric Database
│   ├── face_embeddings.pkl    # 109 Enrolled 512-D ArcFace vectors (Pickle cache)
│   ├── encodings.json         # Portable JSON encodings with Centroid & Exemplars
│   └── label_map.json         # Identity mapping: "0": "Caleb_Yonli" (ID-001)
│
├── icons/                     # High-Resolution App Icons
│   ├── favicon.svg            # Biometric reticle SVG
│   ├── icon-192.png           # 192x192 PWA Icon
│   ├── icon-512.png           # 512x512 PWA Icon
│   ├── icon-maskable.png      # Android Adaptive Maskable Icon
│   └── apple-touch-icon.png   # iOS Safari Home Screen Icon
│
├── frontend/                  # Standalone directory ready for direct Vercel import
├── templates/                 # Flask HTML templates
├── static/                    # Flask static assets
└── android-apk/               # Complete Android Studio project & APK guide
```

---

## 📊 Training & Biometric Performance Metrics

The model auto-labeled and trained across **109 diverse, high-resolution photos** of Caleb Yonli (desk, outdoor, office, formal with tie, studio, smiling, glasses on/off, and downward head tilt angles).

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Primary Identity** | **Caleb Yonli** | Authorized Principal |
| **Biometric ID** | **ID-001** | Primary Enrolled Subject |
| **Vector Space** | **512-D ArcFace** | InsightFace ONNX (`buffalo_sc`) |
| **Enrolled Vectors** | **109 Vectors** | Multi-angle ensemble coverage |
| **Min Cosine Dist** | **0.1124** | **88.76%** Peak Similarity |
| **Mean Cosine Dist** | **0.2176** | **78.24%** Average Intra-Class Similarity |
| **Max Cosine Dist** | **0.3628** | **63.72%** Worst-Case Angle Similarity |
| **Cosine Threshold** | **0.4000** | Strict ArcFace Verification Boundary |
| **Caleb Verification**| **PASS (100.0%)** | 100% verified across all test samples |
| **Imposter Rejection**| **FAIL (21.2%)** | Distance: **0.7884** (Strongly Rejected) |
| **Inference Latency** | **45ms - 85ms** | Real-time on standard CPU |

---

## 🛠️ Mobile Web App (PWA) Features

### 1. 📷 Live Camera HUD
- **Real-Time Viewfinder**: High-definition video feed with glowing cyan reticles and animated sweeping biometric laser beam.
- **Auto-Scan Toggle**: Hands-free continuous verification looping every 2.5 seconds with live result feedback.
- **Manual Snap**: High-resolution frame capture with audio shutter sound and flash animation.
- **Bounding Box Reticle**: Canvas projection dynamically draws neon targeting brackets around Caleb Yonli's face with label `TARGET LOCK · CALEB YONLI (ID-001)`.
- **Hardware Controls**: Instant flip camera (front/back lens) and torch/flashlight toggle.
- **Demo Scan Simulation**: 1-click test button pre-loaded with Caleb Yonli's signature sample for testing on non-HTTPS environments or devices without webcams.

### 2. 🖼️ Photo Upload Recognition
- Drag-and-drop or tap to select any high-resolution image (JPG, PNG, WEBP, HEIC up to 25MB).
- Instant preview with laser sweep animation.
- Bounding box accurately projected over facial coordinates.
- Quick test presets: Desk (Green Shirt), Studio (Blue Pants), Formal (Tie & Glasses).

### 3. 🎯 Distinct PASS / FAIL Status
- **PASS (Verified Match)**:
  - Green neon badge: `IDENTITY VERIFIED · PASS`
  - Glowing avatar halo with initials `CY`
  - Name: `Caleb Yonli`
  - Subtitle: `ID: ID-001 · ENROLLED PRINCIPAL`
  - Animated green confidence bar (e.g. `99.4%`)
  - Audio: Harmonic 3-tone major chime
  - Haptic: Double-pulse confirmation vibration
- **FAIL (Access Denied / Unrecognized)**:
  - Electric red badge: `ACCESS DENIED · FAIL`
  - Warning pulse avatar `?`
  - Name: `Unknown Subject`
  - Subtitle: `UNAUTHORIZED · NOT IN DATABASE`
  - Audio: Low sawtooth warning buzz
  - Haptic: Extended warning vibration

### 4. 📄 Live Raw JSON Response Viewer
- Direct `{ } View Raw JSON` button on the bottom sheet and on each history record.
- Displays full syntax-highlighted payload:
  ```json
  {
    "matched": true,
    "status": "PASS",
    "name": "Caleb Yonli",
    "id": "ID-001",
    "confidence": 100.0,
    "cosine_dist": 0.0000,
    "det_score": 84.4,
    "bbox": [275, 81, 366, 208],
    "image_size": [682, 1024],
    "threshold": 0.4,
    "total_encodings": 109,
    "latency_ms": 85.8,
    "timestamp": 1795352197746
  }
  ```
- 1-Click "Copy JSON" to clipboard with toast confirmation and "Download JSON" button.

### 5. 📝 Attendance History Auto-Log
- Automatically logs every single verification scan (camera snap, auto-scan loop, upload).
- Real-time counter badge on the bottom navigation tab.
- Summary metric cards: Total Scans, Verified (PASS), Rejected (FAIL), Pass Rate %.
- Filter records by: `All`, `Pass Only`, `Fail Only`.
- One-click **Download CSV** (`caleb_yonli_attendance_log.csv`) formatted for Excel and HR systems.

---

## 🚀 Deployment Instructions

### 1. Deploy Frontend to Vercel
1. Push your repository to **GitHub**.
2. Go to **[https://vercel.com](https://vercel.com)** -> Click **Add New Project**.
3. Import your GitHub repository (`bendicalebyonli-hub/caleb-face-recognition`).
4. (Optional) Set Root Directory to `frontend` (or leave at root `/` since `vercel.json` and `index.html` are at both locations).
5. Click **Deploy**.
6. Your PWA will be live at `https://your-project.vercel.app`!

### 2. Deploy Backend to Render
1. Go to **[https://render.com](https://render.com)** -> Click **New +** -> **Web Service**.
2. Connect your GitHub repository.
3. Configure settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -w 1 -b 0.0.0.0:$PORT --timeout 120 app:app`
4. Add Environment Variable:
   - `OPENBLAS_NUM_THREADS` = `1`
5. Click **Deploy Web Service**.
6. Copy your Render service URL (e.g. `https://caleb-face-recognition.onrender.com`) and paste it into the app's Settings modal (`⚙️`).

---

## 📱 How to Download & Install the Android APK

### Option A: 1-Click Cloud APK via PWABuilder (Fastest)
1. Go to **[https://www.pwabuilder.com](https://www.pwabuilder.com)**.
2. Enter your live Vercel URL and click **Start**.
3. Click **Package for Stores** -> Select **Android**.
4. Click **Generate / Download Package**.
5. Transfer the downloaded `.apk` to your phone and tap to install!

### Option B: Add to Home Screen (Instant PWA)
1. Open your Vercel URL in **Chrome** or **Edge** on your Android phone.
2. Tap the bottom banner: **"Install Caleb FaceID Mobile App on Home Screen"** (or tap `⋮` -> **"Add to Home Screen"**).
3. The app icon appears on your home screen and launches full-screen without any browser address bar!
