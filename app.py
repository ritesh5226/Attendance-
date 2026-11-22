"""
Web-based Real-Time Attendance Monitoring System (Face + LBPH features)
Run:  python app.py      then open  http://127.0.0.1:5050

The browser captures the webcam; each frame is sent to this Flask server,
which detects the face (Haar Cascade), extracts LBPH features, recognises
the person and writes attendance to attendance.csv.
"""

import base64
import csv
import json
import os
import re
import shutil
import threading
import webbrowser
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_FILE = os.path.join(BASE_DIR, "trainer.yml")
LABELS_FILE = os.path.join(BASE_DIR, "labels.json")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")

FACE_SIZE = (200, 200)
CONFIDENCE_THRESHOLD = 70  # LBPH distance: LOWER = better match
NAME_RE = re.compile(r"^[A-Za-z0-9 _-]{1,40}$")
PORT = 5050  # 5000 is used by AirPlay on recent macOS versions

app = Flask(__name__)
lock = threading.Lock()
state = {"rec": None, "labels": {}, "date": None, "done": set()}

cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)

if cascade.empty():
    raise SystemExit(
        "ERROR: Face detector file not found. Install a 4.x OpenCV build:\n"
        "  pip uninstall -y opencv-python opencv-contrib-python\n"
        "  pip install 'opencv-contrib-python>=4.8,<5'"
    )


# ---------------------------------------------------------------- helpers
def decode(data_url):
    try:
        b64 = data_url.split(",", 1)[1]
        arr = np.frombuffer(base64.b64decode(b64), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def detect_faces(gray):
    gray = cv2.equalizeHist(gray)
    return cascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))


def preprocess(gray, box):
    x, y, w, h = box
    face = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
    return cv2.equalizeHist(face)


def sanitize(name):
    name = (name or "").strip()
    return name if NAME_RE.match(name) else None


def read_rows():
    if not os.path.exists(ATTENDANCE_FILE):
        return []
    with open(ATTENDANCE_FILE, newline="") as f:
        return list(csv.DictReader(f))


def refresh_done():
    """Keep an in-memory set of names already marked today."""
    today = datetime.now().strftime("%Y-%m-%d")
    if state["date"] != today:
        state["date"] = today
        state["done"] = {r["Name"] for r in read_rows() if r["Date"] == today}


def mark_attendance(name):
    new_file = not os.path.exists(ATTENDANCE_FILE)
    now = datetime.now()
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["Name", "Date", "Time", "Status"])
        w.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), "Present"])
    print(f"[ATTENDANCE] {name} marked present at {now.strftime('%H:%M:%S')}")


def load_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(LABELS_FILE):
        rec = cv2.face.LBPHFaceRecognizer_create()
        rec.read(MODEL_FILE)
        with open(LABELS_FILE) as f:
            state["labels"] = {int(k): v for k, v in json.load(f).items()}
        state["rec"] = rec
    else:
        state["rec"] = None


def list_people():
    if not os.path.isdir(DATASET_DIR):
        return []
    people = []
    for p in sorted(os.listdir(DATASET_DIR)):
        pdir = os.path.join(DATASET_DIR, p)
        if os.path.isdir(pdir):
            n = len([f for f in os.listdir(pdir) if f.lower().endswith(".jpg")])
            people.append({"name": p, "samples": n})
    return people


# ------------------------------------------------------------------ pages
@app.get("/")
def index():
    return render_template("index.html")


# -------------------------------------------------------------------- API
@app.post("/api/register")
def api_register():
    data = request.get_json(silent=True) or {}
    name = sanitize(data.get("name"))
    if not name:
        return jsonify(error="Invalid name"), 400
    frame = decode(data.get("image", ""))
    if frame is None:
        return jsonify(error="Bad image"), 400
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    if len(faces) == 0:
        return jsonify(saved=False, box=None)
    box = max(faces, key=lambda b: b[2] * b[3])
    pdir = os.path.join(DATASET_DIR, name)
    os.makedirs(pdir, exist_ok=True)
    with lock:
        idx = len([f for f in os.listdir(pdir) if f.lower().endswith(".jpg")]) + 1
        cv2.imwrite(os.path.join(pdir, f"{idx}.jpg"), preprocess(gray, box))
    x, y, w, h = (int(v) for v in box)
    return jsonify(saved=True, box={"x": x, "y": y, "w": w, "h": h})


@app.post("/api/train")
def api_train():
    images, labels, label_map = [], [], {}
    label = 0
    for person in [p["name"] for p in list_people()]:
        pdir = os.path.join(DATASET_DIR, person)
        count_before = len(images)
        for f in os.listdir(pdir):
            if f.lower().endswith(".jpg"):
                img = cv2.imread(os.path.join(pdir, f), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(cv2.resize(img, FACE_SIZE))
                    labels.append(label)
        if len(images) > count_before:
            label_map[label] = person
            label += 1
    if not images:
        return jsonify(error="No face samples found. Register at least one person first."), 400
    rec = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    rec.train(images, np.array(labels))
    with lock:
        rec.write(MODEL_FILE)
        with open(LABELS_FILE, "w") as f:
            json.dump(label_map, f)
        load_model()
    return jsonify(message=f"Trained on {len(images)} images of {len(label_map)} people.")


@app.post("/api/recognize")
def api_recognize():
    data = request.get_json(silent=True) or {}
    frame = decode(data.get("image", ""))
    if frame is None:
        return jsonify(error="Bad image"), 400
    with lock:
        if state["rec"] is None:
            load_model()
        if state["rec"] is None:
            return jsonify(error="Model not trained yet. Register people and click Train Model.")
        refresh_done()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = []
        for box in detect_faces(gray):
            label, conf = state["rec"].predict(preprocess(gray, box))
            x, y, w, h = (int(v) for v in box)
            item = {"x": x, "y": y, "w": w, "h": h, "conf": round(float(conf)),
                    "name": "Unknown", "marked": False}
            if conf < CONFIDENCE_THRESHOLD:
                name = state["labels"].get(label, "Unknown")
                item["name"] = name
                if name != "Unknown" and name not in state["done"]:
                    mark_attendance(name)
                    state["done"].add(name)
                    item["marked"] = True
            results.append(item)
        return jsonify(faces=results, present=len(state["done"]))


@app.get("/api/summary")
def api_summary():
    with lock:
        refresh_done()
        people = list_people()
        names = [p["name"] for p in people]
        present = sorted(n for n in state["done"] if n in names)
        absent = [n for n in names if n not in state["done"]]
    return jsonify(people=people, registered=len(people), present=present, absent=absent)


@app.get("/api/attendance")
def api_attendance():
    date = request.args.get("date")
    rows = read_rows()
    if date:
        rows = [r for r in rows if r["Date"] == date]
    return jsonify(rows=list(reversed(rows)))


@app.delete("/api/people/<name>")
def api_delete_person(name):
    name = sanitize(name)
    if not name:
        return jsonify(error="Invalid name"), 400
    pdir = os.path.join(DATASET_DIR, name)
    if os.path.isdir(pdir):
        shutil.rmtree(pdir)
    return jsonify(message=f"Deleted {name}. Click Train Model to update the model.")


@app.get("/download")
def download():
    if not os.path.exists(ATTENDANCE_FILE):
        return "No attendance records yet.", 404
    return send_file(ATTENDANCE_FILE, as_attachment=True, download_name="attendance.csv")


if __name__ == "__main__":
    url = f"http://127.0.0.1:{PORT}"
    print(f"\nAttendance website running at {url}  (Ctrl+C to stop)\n")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
