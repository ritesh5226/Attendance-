"""
Real-Time Attendance Monitoring System based on Face and Features Recognition
-----------------------------------------------------------------------------
Face detection   : Haar Cascade (Viola-Jones)
Feature extract. : LBPH (Local Binary Patterns Histograms)
Recognition      : LBPH face recognizer (OpenCV contrib)
Storage          : CSV file (attendance.csv)

Commands:
    python attendance.py register --name "Rahul"   # capture face samples
    python attendance.py train                      # train the recognizer
    python attendance.py run                        # real-time attendance
    python attendance.py report                     # show attendance records
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_FILE = os.path.join(BASE_DIR, "trainer.yml")
LABELS_FILE = os.path.join(BASE_DIR, "labels.json")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")

FACE_SIZE = (200, 200)
SAMPLES_PER_PERSON = 60
# LBPH: LOWER value = better match. Increase if known people show "Unknown",
# decrease if strangers get recognised.
CONFIDENCE_THRESHOLD = 70

cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)

if cascade.empty():
    raise SystemExit(
        "ERROR: Face detector file not found. Install a 4.x OpenCV build:\n"
        "  pip uninstall -y opencv-python opencv-contrib-python\n"
        "  pip install 'opencv-contrib-python>=4.8,<5'"
    )


def open_camera():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        sys.exit(
            "ERROR: Cannot open camera.\n"
            "Mac: System Settings > Privacy & Security > Camera > enable your "
            "Terminal / VS Code, then restart it."
        )
    return cam


def detect_faces(gray):
    gray = cv2.equalizeHist(gray)
    return cascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))


def preprocess(gray, box):
    x, y, w, h = box
    face = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
    return cv2.equalizeHist(face)


# ------------------------------------------------------------------ REGISTER
def register(name):
    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    cam = open_camera()
    count = len(os.listdir(person_dir))
    target = count + SAMPLES_PER_PERSON
    frame_no = 0
    print(f"Capturing faces for '{name}'. Look at the camera, move your head slightly. Press Q to stop.")

    while count < target:
        ok, frame = cam.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)
        frame_no += 1
        if len(faces) > 0:
            box = max(faces, key=lambda b: b[2] * b[3])  # largest face
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if frame_no % 3 == 0:
                count += 1
                cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), preprocess(gray, box))
        cv2.putText(frame, f"{name}: {count}/{target}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow("Register Face (Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} samples for '{name}'. Now run: python attendance.py train")


# --------------------------------------------------------------------- TRAIN
def train():
    if not os.path.isdir(DATASET_DIR) or not os.listdir(DATASET_DIR):
        sys.exit("No dataset found. Register at least one person first.")
    images, labels, label_map = [], [], {}
    for idx, person in enumerate(sorted(os.listdir(DATASET_DIR))):
        pdir = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(pdir):
            continue
        label_map[idx] = person
        for f in os.listdir(pdir):
            img = cv2.imread(os.path.join(pdir, f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(cv2.resize(img, FACE_SIZE))
                labels.append(idx)
    if not images:
        sys.exit("No images found in dataset.")
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(images, np.array(labels))
    recognizer.write(MODEL_FILE)
    with open(LABELS_FILE, "w") as f:
        json.dump(label_map, f)
    print(f"Trained on {len(images)} images of {len(label_map)} people: {list(label_map.values())}")


# ---------------------------------------------------------------- ATTENDANCE
def marked_today():
    today = datetime.now().strftime("%Y-%m-%d")
    names = set()
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, newline="") as f:
            for row in csv.DictReader(f):
                if row["Date"] == today:
                    names.add(row["Name"])
    return names


def mark_attendance(name):
    new_file = not os.path.exists(ATTENDANCE_FILE)
    now = datetime.now()
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["Name", "Date", "Time", "Status"])
        w.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), "Present"])
    print(f"[ATTENDANCE] {name} marked present at {now.strftime('%H:%M:%S')}")


def run():
    if not (os.path.exists(MODEL_FILE) and os.path.exists(LABELS_FILE)):
        sys.exit("Model not found. Run: python attendance.py train")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)
    with open(LABELS_FILE) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    done = marked_today()
    cam = open_camera()
    print("Attendance running. Press Q to quit.")

    while True:
        ok, frame = cam.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for box in detect_faces(gray):
            x, y, w, h = box
            label, conf = recognizer.predict(preprocess(gray, box))
            if conf < CONFIDENCE_THRESHOLD:
                name = label_map.get(label, "Unknown")
                color = (0, 255, 0)
                if name not in done:
                    mark_attendance(name)
                    done.add(name)
                text = f"{name} ({conf:.0f})"
            else:
                color, text = (0, 0, 255), "Unknown"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(frame, f"Present today: {len(done)}  |  {datetime.now().strftime('%H:%M:%S')}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Real-Time Attendance (Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


# -------------------------------------------------------------------- REPORT
def report():
    if not os.path.exists(ATTENDANCE_FILE):
        print("No attendance records yet.")
        return
    with open(ATTENDANCE_FILE, newline="") as f:
        rows = list(csv.reader(f))
    for r in rows:
        print(f"{r[0]:<20}{r[1]:<14}{r[2]:<10}{r[3]}")


def main():
    p = argparse.ArgumentParser(description="Face Recognition Attendance System")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register", help="capture face samples for a person")
    r.add_argument("--name", required=True)
    sub.add_parser("train", help="train the recognizer")
    sub.add_parser("run", help="start real-time attendance")
    sub.add_parser("report", help="print attendance records")
    args = p.parse_args()
    {"register": lambda: register(args.name), "train": train, "run": run, "report": report}[args.cmd]()


if __name__ == "__main__":
    main()
