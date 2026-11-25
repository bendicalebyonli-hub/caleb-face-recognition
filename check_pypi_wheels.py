import urllib.request
import json

packages = [
    "flask", "flask-cors", "gunicorn", "insightface", "onnxruntime",
    "opencv-python-headless", "pillow", "pillow-heif", "numpy", "scipy", "scikit-image", "tqdm", "requests"
]

for p in packages:
    try:
        req = urllib.request.urlopen(f"https://pypi.org/pypi/{p}/json", timeout=10)
        data = json.loads(req.read().decode())
        print(f"PyPI has {p}: latest={data.get('info', {}).get('version')}")
    except Exception as e:
        print(f"PyPI ERROR for {p}: {e}")
