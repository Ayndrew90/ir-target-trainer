import os
import json
import cv2
import threading
import time
import math
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()

from flask import (
    Blueprint,
    render_template,
    request,
    current_app,
    jsonify,
    url_for,
    flash,
    redirect
)

from .data_handler import load_data, save_data, get_next_id

bp_laser = Blueprint("bp_laser", __name__, template_folder="templates")

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

roi = None
hits = []
last_detection_time = 0.0
prev_frame = None
total_score = 0
last_display_frame = None
max_hits_allowed = 0
detection_paused = False
lock_state = threading.Lock()


class Camera:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.capture = cv2.VideoCapture(self.rtsp_url)
        self.lock = threading.Lock()
        self.capture_lock = threading.Lock()
        self.frame = None
        self.running = True

        if not self.capture.isOpened():
            print("Warning: could not open RTSP stream for laser mode.")

    def start(self):
        thread = threading.Thread(target=self.update, daemon=True)
        thread.start()

    def update(self):
        while self.running:
            with self.capture_lock:
                capture = self.capture
                opened = capture is not None and capture.isOpened()
                if opened:
                    ret, frame = capture.read()
                else:
                    ret = False
                    frame = None

            if opened and ret:
                with self.lock:
                    self.frame = frame
            elif opened and not ret:
                time.sleep(0.05)
            else:
                time.sleep(0.5)

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False
        with self.capture_lock:
            try:
                self.capture.release()
            except Exception:
                pass

    def reset(self):
        with self.capture_lock:
            try:
                if self.capture is not None:
                    self.capture.release()
            except Exception:
                pass

            self.capture = cv2.VideoCapture(self.rtsp_url)

        with self.lock:
            self.frame = None

        if not self.capture.isOpened():
            print("Warning: could not reopen RTSP stream for laser mode.")


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
    global roi, hits, last_detection_time, prev_frame, total_score, last_display_frame, detection_paused

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

    with lock_state:
        paused_due_to_limit = detection_paused or (
            max_hits_allowed > 0 and len(hits) >= max_hits_allowed
        )

    if paused_due_to_limit:
        with lock_state:
            for hval in hits:
                hx = hval["x"]
                hy = hval["y"]
                cv2.circle(frame, (hx, hy), HIT_MARKER_RADIUS, (0, 0, 255), 2)
                cv2.circle(frame, (hx, hy), 3, (0, 0, 255), -1)
            last_display_frame = frame.copy()
        return frame

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

                if roi is not None:
                    rx, ry, rw, rh = roi
                    roi_dict = {"x": rx, "y": ry, "w": rw, "h": rh}
                else:
                    roi_dict = None

                with lock_state:
                    hits.append({"x": cx, "y": cy, "score": score, "roi": roi_dict})
                    total = 0
                    for hval in hits:
                        total += hval["score"]
                    globals()["total_score"] = total
                    if max_hits_allowed > 0 and len(hits) >= max_hits_allowed:
                        globals()["detection_paused"] = True
                last_detection_time = now

    with lock_state:
        for hval in hits:
            hx = hval["x"]
            hy = hval["y"]
            cv2.circle(frame, (hx, hy), HIT_MARKER_RADIUS, (0, 0, 255), 2)
            cv2.circle(frame, (hx, hy), 3, (0, 0, 255), -1)
        last_display_frame = frame.copy()

    return frame


def get_json_paths():
    base = current_app.config["JSON_FOLDER"]
    return {
        "sessions": os.path.join(base, "laser_visits.json"),
        "rounds": os.path.join(base, "laser_rounds.json"),
        "users": os.path.join(base, "laser_users.json"),
        "shots": os.path.join(base, "laser_shots.json"),
        "guns": os.path.join(base, "laser_guns.json"),
        "targets": os.path.join(base, "laser_targets.json"),
    }


@bp_laser.route("/")
def index():
    return render_template("laser_index.html")


@bp_laser.route("/range_visits")
def range_visits():
    paths = get_json_paths()
    sessions = load_data(paths["sessions"])
    sessions.sort(key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"), reverse=True)

    guns = load_data(paths["guns"])

    def get_gun_name(g_id):
        for g in guns:
            if g["id"] == g_id:
                return g["name"]
        return "Unknown"

    return render_template("laser_range_visits.html", sessions=sessions, get_gun_name=get_gun_name)


@bp_laser.route("/delete_visit/<int:session_id>", methods=["POST"])
def delete_visit(session_id):
    paths = get_json_paths()
    sessions = load_data(paths["sessions"])
    rounds = load_data(paths["rounds"])
    shots_data = load_data(paths["shots"])

    session_obj = next((s for s in sessions if s["id"] == session_id), None)
    if not session_obj:
        flash("Laser session not found!", "danger")
        return redirect(url_for("bp_laser.range_visits"))

    rounds_to_delete = [r for r in rounds if r["session_id"] == session_id]
    round_ids = {r["id"] for r in rounds_to_delete}

    upload_folder = current_app.config["UPLOAD_FOLDER"]
    for r in rounds_to_delete:
        if r.get("photo"):
            path = os.path.join(upload_folder, r["photo"])
            if os.path.exists(path):
                os.remove(path)
        rounds.remove(r)

    shots_data = [s for s in shots_data if s["round_id"] not in round_ids]

    sessions.remove(session_obj)

    save_data(paths["sessions"], sessions)
    save_data(paths["rounds"], rounds)
    save_data(paths["shots"], shots_data)

    flash("Laser session deleted!", "success")
    return redirect(url_for("bp_laser.range_visits"))


@bp_laser.route("/view_visit/<int:session_id>")
def view_visit(session_id):
    paths = get_json_paths()
    sessions = load_data(paths["sessions"])
    rounds_all = load_data(paths["rounds"])
    users = load_data(paths["users"])
    targets = load_data(paths["targets"])
    shots_data = load_data(paths["shots"])

    session_obj = next((s for s in sessions if s["id"] == session_id), None)
    if not session_obj:
        return render_template(
            "laser_view_visit.html",
            session=None,
            rounds=[],
            get_user_name=lambda _id: "Unknown",
            get_target_name=lambda _id: "Unknown",
            stream_width=STREAM_WIDTH,
            stream_height=STREAM_HEIGHT,
        )

    shots_map = {entry["round_id"]: entry for entry in shots_data}

    session_rounds = []
    for r in rounds_all:
        if r["session_id"] != session_id:
            continue
        entry = shots_map.get(r["id"])
        hits_list = entry["hits"] if entry else []
        total_shots = entry["total_shots"] if entry else 0
        r_copy = dict(r)
        r_copy["hits"] = hits_list
        r_copy["total_shots"] = total_shots
        session_rounds.append(r_copy)

    def get_user_name(u_id):
        for u in users:
            if u["id"] == u_id:
                return u["name"]
        return "Unknown"

    def get_target_name(t_id):
        for t in targets:
            if t["id"] == t_id:
                return t["name"]
        return "Unknown"

    return render_template(
        "laser_view_visit.html",
        session=session_obj,
        rounds=session_rounds,
        get_user_name=get_user_name,
        get_target_name=get_target_name,
        stream_width=STREAM_WIDTH,
        stream_height=STREAM_HEIGHT,
    )


@bp_laser.route("/settings", methods=["GET", "POST"])
def settings():
    paths = get_json_paths()
    users_json = paths["users"]
    guns_json = paths["guns"]
    targets_json = paths["targets"]

    if request.method == "POST":
        form_type = request.form.get("form_type")

        if form_type == "user":
            name = request.form.get("user_name")
            if name:
                users = load_data(users_json)
                new_id = get_next_id(users)
                users.append({"id": new_id, "name": name})
                save_data(users_json, users)

        elif form_type == "gun":
            name = request.form.get("gun_name")
            if name:
                guns = load_data(guns_json)
                new_id = get_next_id(guns)
                guns.append({"id": new_id, "name": name})
                save_data(guns_json, guns)

        elif form_type == "target":
            name = request.form.get("target_name")
            if name:
                targets = load_data(targets_json)
                new_id = get_next_id(targets)
                targets.append({"id": new_id, "name": name})
                save_data(targets_json, targets)

    all_users = load_data(users_json)
    all_guns = load_data(guns_json)
    all_targets = load_data(targets_json)

    return render_template(
        "laser_settings.html",
        all_users=all_users,
        all_guns=all_guns,
        all_targets=all_targets,
    )


@bp_laser.route("/statistics", methods=["GET", "POST"])
def statistics():
    paths = get_json_paths()
    sessions = load_data(paths["sessions"])
    rounds = load_data(paths["rounds"])
    users = load_data(paths["users"])
    guns = load_data(paths["guns"])
    targets = load_data(paths["targets"])
    shots = load_data(paths["shots"])

    filter_user = request.form.get("filter_user")
    filter_gun = request.form.get("filter_gun")
    filter_target = request.form.get("filter_target")
    filter_distance = request.form.get("filter_distance")
    filter_recents = request.form.get("filter_recents")

    def to_int_or_none(val):
        if not val or val in ("all", "alle"):
            return None
        try:
            return int(val)
        except ValueError:
            return None

    filter_user_id = to_int_or_none(filter_user)
    filter_gun_id = to_int_or_none(filter_gun)
    filter_target_id = to_int_or_none(filter_target)

    if not filter_distance or filter_distance == "all":
        filter_distance_val = None
    else:
        filter_distance_val = filter_distance

    if not filter_recents or filter_recents == "all":
        filter_recents_val = None
    else:
        try:
            filter_recents_val = int(filter_recents)
        except ValueError:
            filter_recents_val = None

    for s in sessions:
        s["_dt"] = datetime.strptime(s["date"], "%Y-%m-%d")
    sessions.sort(key=lambda x: x["_dt"], reverse=True)

    if filter_recents_val:
        sessions = sessions[:filter_recents_val]

    if filter_gun_id is not None:
        sessions = [s for s in sessions if s.get("gun_id") == filter_gun_id]

    session_map = {s["id"]: s for s in sessions}
    allowed_session_ids = set(session_map.keys())

    filtered_rounds = []
    for r in rounds:
        if r["session_id"] not in allowed_session_ids:
            continue

        if filter_user_id is not None and r.get("user_id") != filter_user_id:
            continue

        if filter_target_id is not None and r.get("target_type_id") != filter_target_id:
            continue

        if filter_distance_val is not None and r.get("distance") != filter_distance_val:
            continue

        filtered_rounds.append(r)

    date_score_map = {}
    for r in filtered_rounds:
        sess = session_map.get(r["session_id"])
        if not sess:
            continue
        d_str = sess["date"]
        date_score_map.setdefault(d_str, []).append(r["score"])

    date_avg_score = []
    for d_str, scores in date_score_map.items():
        dt = datetime.strptime(d_str, "%Y-%m-%d")
        avg_s = sum(scores) / len(scores) if scores else 0
        date_avg_score.append((dt, avg_s))

    date_avg_score.sort(key=lambda x: x[0])
    labels_avg_score = [x[0].strftime("%Y-%m-%d") for x in date_avg_score]
    data_avg_score = [x[1] for x in date_avg_score]

    visits_for_chart = []
    sessions_sorted_asc = sorted(sessions, key=lambda x: x["_dt"])

    for s in sessions_sorted_asc:
        s_rounds = [rnd for rnd in filtered_rounds if rnd["session_id"] == s["id"]]
        if s_rounds:
            avg_score = sum(r["score"] for r in s_rounds) / len(s_rounds)
            visits_for_chart.append((s["date"], avg_score))

    labels_visit_chart = [x[0] for x in visits_for_chart]
    data_visit_chart = [x[1] for x in visits_for_chart]

    filtered_round_ids = {r["id"] for r in filtered_rounds}
    shots_for_heatmap = []

    scale_x = 400.0 / float(STREAM_WIDTH)
    scale_y = 400.0 / float(STREAM_HEIGHT)

    for entry in shots:
        if entry["round_id"] not in filtered_round_ids:
            continue
        for h in entry.get("hits", []):
            x_stream = h.get("x")
            y_stream = h.get("y")
            if x_stream is None or y_stream is None:
                continue
            shots_for_heatmap.append({
                "x": x_stream * scale_x,
                "y": y_stream * scale_y
            })

    return render_template(
        "laser_statistics.html",
        all_users=users,
        all_guns=guns,
        all_targets=targets,
        filter_user=filter_user,
        filter_gun=filter_gun,
        filter_target=filter_target,
        filter_distance=filter_distance,
        filter_recents=filter_recents,
        labels_avg_score=labels_avg_score,
        data_avg_score=data_avg_score,
        labels_visit_chart=labels_visit_chart,
        data_visit_chart=data_visit_chart,
        shots_for_heatmap=shots_for_heatmap,
    )


@bp_laser.route("/new_range_visit", methods=["GET", "POST"])
def new_range_visit():
    paths = get_json_paths()
    guns = load_data(paths["guns"])
    targets = load_data(paths["targets"])
    users = load_data(paths["users"])

    if request.method == "POST":
        shots_per_round = int(request.form.get("shots_per_round", 0))
        gun_id = int(request.form.get("gun_id"))
        rounds_json_str = request.form.get("rounds_json", "[]")

        try:
            rounds_payload = json.loads(rounds_json_str)
        except json.JSONDecodeError:
            rounds_payload = []

        sessions = load_data(paths["sessions"])
        rounds = load_data(paths["rounds"])
        shots_data = load_data(paths["shots"])

        new_session_id = get_next_id(sessions)
        today_str = date.today().strftime("%Y-%m-%d")
        new_session = {
            "id": new_session_id,
            "date": today_str,
            "gun_id": gun_id,
            "shots_per_round": shots_per_round,
        }
        sessions.append(new_session)

        for r_obj in rounds_payload:
            user_id = int(r_obj.get("user_id"))
            target_id = int(r_obj.get("target_id"))
            distance = r_obj.get("distance")
            raw_score = float(r_obj.get("raw_score", 0.0))
            hits_list = r_obj.get("hits", [])
            photo_filename = r_obj.get("photo")

            score_per_shot = raw_score / shots_per_round if shots_per_round > 0 else 0.0

            new_round_id = get_next_id(rounds)
            new_round = {
                "id": new_round_id,
                "session_id": new_session_id,
                "user_id": user_id,
                "target_type_id": target_id,
                "distance": distance,
                "score": score_per_shot,
                "photo": photo_filename,
            }
            rounds.append(new_round)

            shots_entry = {
                "round_id": new_round_id,
                "total_shots": shots_per_round,
                "hits": hits_list,
            }
            shots_data.append(shots_entry)

        save_data(paths["sessions"], sessions)
        save_data(paths["rounds"], rounds)
        save_data(paths["shots"], shots_data)

        return render_template("laser_new_visit_done.html", session_id=new_session_id)

    today_str = date.today().strftime("%Y-%m-%d")

    return render_template(
        "laser_new_visit.html",
        all_guns=guns,
        all_targets=targets,
        all_users=users,
        today=today_str,
    )


@bp_laser.route("/video_feed")
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

            ret, buffer = cv2.imencode(".jpg", display_frame)
            if not ret:
                continue
            jpg_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
            )

    return current_app.response_class(
        gen(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@bp_laser.route("/set_roi", methods=["POST"])
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


@bp_laser.route("/clear_roi", methods=["POST"])
def clear_roi():
    global roi
    roi = None
    return jsonify({"status": "ok"})


@bp_laser.route("/reset_camera", methods=["POST"])
def reset_camera():
    global prev_frame, last_display_frame

    camera.reset()
    with lock_state:
        prev_frame = None
        last_display_frame = None

    return jsonify({"status": "ok"})


@bp_laser.route("/clear_hits", methods=["POST"])
def clear_hits():
    global hits, total_score, detection_paused
    with lock_state:
        hits = []
        total_score = 0
        detection_paused = False
    return jsonify({"status": "ok"})


@bp_laser.route("/set_hits_limit", methods=["POST"])
def set_hits_limit():
    global max_hits_allowed, detection_paused
    data = request.get_json(force=True) or {}

    try:
        limit = int(data.get("limit", 0))
    except (TypeError, ValueError):
        limit = 0

    limit = max(0, limit)

    with lock_state:
        max_hits_allowed = limit

        current_hits = len(hits)

        if detection_paused and current_hits > 0:
            pass
        elif max_hits_allowed == 0:
            detection_paused = False
        elif current_hits >= max_hits_allowed:
            detection_paused = True
        else:
            detection_paused = False

    return jsonify(
        {
            "status": "ok",
            "limit": max_hits_allowed,
            "hits": current_hits,
            "paused": detection_paused,
        }
    )


@bp_laser.route("/stats")
def stats():
    with lock_state:
        count_hits = len(hits)
        total = total_score
        avg = float(total) / count_hits if count_hits > 0 else 0.0
        hits_copy = list(hits)
    return jsonify({"hits": count_hits, "score": total, "average": avg, "hits_list": hits_copy})


@bp_laser.route("/save_snapshot", methods=["POST"])
def save_snapshot():
    global roi, hits

    frame = camera.get_frame()
    if frame is None:
        return jsonify({"status": "error", "message": "no frame"}), 400

    frame = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT))

    with lock_state:
        hits_copy = list(hits)
        roi_local = roi

    for hval in hits_copy:
        cv2.circle(frame, (hval["x"], hval["y"]), HIT_MARKER_RADIUS, (0, 0, 255), 2)
        cv2.circle(frame, (hval["x"], hval["y"]), 3, (0, 0, 255), -1)

    if roi_local is not None:
        x, y, w, h = roi_local
        h_frame, w_frame = frame.shape[:2]
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(1, min(w, w_frame - x))
        h = max(1, min(h, h_frame - y))
        cropped = frame[y:y + h, x:x + w]
    else:
        cropped = frame

    upload_folder = current_app.config["UPLOAD_FOLDER"]
    os.makedirs(upload_folder, exist_ok=True)

    ts = int(time.time() * 1000)
    filename = f"laser_{ts}.jpg"
    path = os.path.join(upload_folder, filename)

    cv2.imwrite(path, cropped)

    return jsonify({"status": "ok", "filename": filename})