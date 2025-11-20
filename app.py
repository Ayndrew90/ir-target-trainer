import cv2
import threading
import time
from dotenv import load_dotenv
import os
from flask import Flask, Response, request, jsonify, render_template
import math

load_dotenv()

RTSP_USER = os.getenv("RTSP_USER")
RTSP_PASSW = os.getenv("RTSP_PASSW")
RTSP_IP = os.getenv("RTSP_IP")
RTSP_PORT = os.getenv("RTSP_PORT")

RTSP_URL = f"rtsp://{RTSP_USER}:{RTSP_PASSW}@{RTSP_IP}:{RTSP_PORT}/stream1"
STREAM_WIDTH = 640
STREAM_HEIGHT = 360
THRESHOLD_VALUE = 25
MIN_CONTOUR_AREA = 15
MIN_TIME_BETWEEN_HITS = 0.2
HIT_MARKER_RADIUS = 8

TARGET_DESIGN_SIZE = 1600.0
RADIUS_10 = 161.0
RADIUS_9 = 280.0
RADIUS_8 = 401.0
RADIUS_7 = 520.0
RADIUS_6 = 642.0

app = Flask(__name__)

roi = None
hits = []
last_detection_time = 0.0
prev_frame = None
total_score = 0


class Camera:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.capture = cv2.VideoCapture(self.rtsp_url)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True

        if not self.capture.isOpened():
            print("Warning: could not open RTSP stream.")

    def start(self):
        thread = threading.Thread(target=self.update, daemon=True)
        thread.start()

    def update(self):
        while self.running:
            if self.capture.isOpened():
                ret, frame = self.capture.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.5)

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False
        try:
            self.capture.release()
        except Exception:
            pass


camera = Camera(RTSP_URL)
camera.start()


def compute_score(hit_x, hit_y, roi_box):
    if roi_box is None:
        return 0

    x, y, w, h = roi_box
    if w <= 0 or h <= 0:
        return 0

    cx = x + w / 2.0
    cy = y + h / 2.0

    scale_x = w / TARGET_DESIGN_SIZE
    scale_y = h / TARGET_DESIGN_SIZE

    dx_design = (hit_x - cx) / scale_x
    dy_design = (hit_y - cy) / scale_y
    r = math.hypot(dx_design, dy_design)

    if r <= RADIUS_10:
        return 10
    if r <= RADIUS_9:
        return 9
    if r <= RADIUS_8:
        return 8
    if r <= RADIUS_7:
        return 7
    if r <= RADIUS_6:
        return 6
    return 0


def detect_hit_and_draw(frame):
    global roi, hits, last_detection_time, prev_frame, total_score

    if frame is None:
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is None:
        prev_frame = gray.copy()
        return frame

    diff = cv2.absdiff(gray, prev_frame)
    prev_frame = gray.copy()
    diff = cv2.GaussianBlur(diff, (7, 7), 0)
    _, thresh = cv2.threshold(diff, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    if roi is not None:
        x, y, w, h = roi
        h_frame, w_frame = thresh.shape[:2]
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(1, min(w, w_frame - x))
        h = max(1, min(h, h_frame - y))
        roi_thresh = thresh[y:y + h, x:x + w]
        offset_x, offset_y = x, y
    else:
        roi_thresh = thresh
        offset_x, offset_y = 0, 0

    contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_c = None
    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_CONTOUR_AREA and area > max_area:
            max_area = area
            max_c = c

    if max_c is not None:
        now = time.time()
        if now - last_detection_time > MIN_TIME_BETWEEN_HITS:
            M = cv2.moments(max_c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cx += offset_x
                cy += offset_y

                score = compute_score(cx, cy, roi)
                hits.append({"x": cx, "y": cy, "score": score})
                total_score += score
                last_detection_time = now

    for hit in hits:
        hx = hit["x"]
        hy = hit["y"]
        cv2.circle(frame, (hx, hy), HIT_MARKER_RADIUS, (0, 0, 255), 2)
        cv2.circle(frame, (hx, hy), 3, (0, 0, 255), -1)

    return frame


@app.route("/")
def index():
    return render_template("index.html", base_width=STREAM_WIDTH, base_height=STREAM_HEIGHT)


@app.route("/video_feed")
def video_feed():
    def gen():
        global roi

        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            frame = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT))
            frame = detect_hit_and_draw(frame)

            display_frame = frame
            if roi is not None:
                x, y, w, h = roi
                h_frame, w_frame = frame.shape[:2]
                x = max(0, min(x, w_frame - 1))
                y = max(0, min(y, h_frame - 1))
                w = max(1, min(w, w_frame - x))
                h = max(1, min(h, h_frame - y))

                cropped = frame[y:y + h, x:x + w]

                if cropped.size != 0:
                    ch, cw = cropped.shape[:2]
                    scale = min(STREAM_WIDTH / cw, STREAM_HEIGHT / ch)
                    new_w = int(cw * scale)
                    new_h = int(ch * scale)
                    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                    pad_left = (STREAM_WIDTH - new_w) // 2
                    pad_right = STREAM_WIDTH - new_w - pad_left
                    pad_top = (STREAM_HEIGHT - new_h) // 2
                    pad_bottom = STREAM_HEIGHT - new_h - pad_top

                    display_frame = cv2.copyMakeBorder(
                        resized,
                        pad_top,
                        pad_bottom,
                        pad_left,
                        pad_right,
                        borderType=cv2.BORDER_CONSTANT,
                        value=[0, 0, 0],
                    )
            else:
                display_frame = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT))

            ret, buffer = cv2.imencode('.jpg', display_frame)
            if not ret:
                continue
            jpg_bytes = buffer.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n'
            )

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/set_roi", methods=["POST"])
def set_roi():
    global roi
    data = request.get_json(force=True)
    x = int(data.get("x", 0))
    y = int(data.get("y", 0))
    w = int(data.get("w", 0))
    h = int(data.get("h", 0))

    x = max(0, x)
    y = max(0, y)
    w = max(1, w)
    h = max(1, h)

    roi = (x, y, w, h)
    return jsonify({"status": "ok", "x": x, "y": y, "w": w, "h": h})


@app.route("/clear_roi", methods=["POST"])
def clear_roi():
    global roi
    roi = None
    return jsonify({"status": "ok"})


@app.route("/clear_hits", methods=["POST"])
def clear_hits():
    global hits, total_score
    hits = []
    total_score = 0
    return jsonify({"status": "ok"})


@app.route("/stats")
def stats():
    hits_count = len(hits)
    average = float(total_score) / hits_count if hits_count > 0 else 0.0
    return jsonify({"hits": hits_count, "score": total_score, "average": average})


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=9999, threaded=True)
    finally:
        camera.stop()