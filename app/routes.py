import os
import uuid
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from werkzeug.utils import secure_filename
from datetime import datetime

from .data_handler import load_data, save_data, get_next_id

bp_main = Blueprint("bp_main", __name__, template_folder="templates")

def save_photo(file_storage):
    if not file_storage:
        return None
    filename = secure_filename(file_storage.filename)
    if filename == "":
        return None

    ext = filename.split(".")[-1]
    unique_name = f"{uuid.uuid4()}.{ext}"
    upload_path = os.path.join(current_app.config["UPLOAD_FOLDER"], unique_name)
    file_storage.save(upload_path)
    return unique_name

@bp_main.route("/")
def index():
    return render_template("index.html")

@bp_main.route("/settings", methods=["GET", "POST"])
def settings():
    ranges_json = os.path.join(current_app.config["JSON_FOLDER"], "ranges.json")
    guns_json = os.path.join(current_app.config["JSON_FOLDER"], "guns.json")
    targets_json = os.path.join(current_app.config["JSON_FOLDER"], "target_types.json")

    if request.method == "POST":
        form_type = request.form.get("form_type")

        if form_type == "range":
            name = request.form.get("range_name")
            file_storage = request.files.get("range_photo")
            photo_filename = save_photo(file_storage)

            if name:
                all_ranges = load_data(ranges_json)
                new_id = get_next_id(all_ranges)
                new_range = {
                    "id": new_id,
                    "name": name,
                    "photo": photo_filename
                }
                all_ranges.append(new_range)
                save_data(ranges_json, all_ranges)
                flash("New shooting range added!", "success")

        elif form_type == "gun":
            name = request.form.get("gun_name")
            file_storage = request.files.get("gun_photo")
            photo_filename = save_photo(file_storage)

            if name:
                all_guns = load_data(guns_json)
                new_id = get_next_id(all_guns)
                new_gun = {
                    "id": new_id,
                    "name": name,
                    "photo": photo_filename
                }
                all_guns.append(new_gun)
                save_data(guns_json, all_guns)
                flash("New gun added!", "success")

        elif form_type == "target":
            name = request.form.get("target_name")
            file_storage = request.files.get("target_photo")
            photo_filename = save_photo(file_storage)

            if name:
                all_targets = load_data(targets_json)
                new_id = get_next_id(all_targets)
                new_target = {
                    "id": new_id,
                    "name": name,
                    "photo": photo_filename
                }
                all_targets.append(new_target)
                save_data(targets_json, all_targets)
                flash("New target added!", "success")

        return redirect(url_for("bp_main.settings"))

    all_ranges = load_data(ranges_json)
    all_guns = load_data(guns_json)
    all_targets = load_data(targets_json)

    return render_template("settings.html", 
                           all_ranges=all_ranges,
                           all_guns=all_guns, 
                           all_targets=all_targets)

@bp_main.route("/new_range_visit", methods=["GET", "POST"])
def new_range_visit():
    ranges_json = os.path.join(current_app.config["JSON_FOLDER"], "ranges.json")
    visits_json = os.path.join(current_app.config["JSON_FOLDER"], "range_visits.json")
    rounds_json = os.path.join(current_app.config["JSON_FOLDER"], "rounds.json")
    guns_json = os.path.join(current_app.config["JSON_FOLDER"], "guns.json")
    targets_json = os.path.join(current_app.config["JSON_FOLDER"], "target_types.json")

    all_ranges = load_data(ranges_json)
    all_guns = load_data(guns_json)
    all_targets = load_data(targets_json)

    if request.method == "POST":
        visit_date = request.form.get("visit_date")
        range_id = request.form.get("range_id")
        ammo_used = request.form.get("ammo_used")

        range_visits = load_data(visits_json)
        new_visit_id = get_next_id(range_visits)
        new_visit = {
            "id": new_visit_id,
            "date": visit_date,
            "range_id": int(range_id),
            "ammo_used": int(ammo_used)
        }
        range_visits.append(new_visit)
        save_data(visits_json, range_visits)

        rounds_list = load_data(rounds_json)
        round_count = int(request.form.get("round_count", 0))
        for i in range(1, round_count + 1):
            gun_field = f"gun_{i}"
            target_field = f"target_{i}"
            score_field = f"score_{i}"
            photo_field = f"photo_{i}"
            distance_field = f"distance_{i}"

            gun_id = request.form.get(gun_field)
            target_type_id = request.form.get(target_field)
            score = request.form.get(score_field)
            distance = request.form.get(distance_field)
            file_storage = request.files.get(photo_field)
            photo_filename = save_photo(file_storage)

            if gun_id and target_type_id and score:
                new_round_id = get_next_id(rounds_list)
                new_round = {
                    "id": new_round_id,
                    "range_visit_id": new_visit_id,
                    "gun_id": int(gun_id),
                    "target_type_id": int(target_type_id),
                    "score": float(score),
                    "photo": photo_filename,
                    "distance": distance
                }
                rounds_list.append(new_round)

        save_data(rounds_json, rounds_list)
        flash("New range visit was created!", "success")
        return redirect(url_for("bp_main.range_visits"))

    return render_template("new_visit.html",
                           all_ranges=all_ranges,
                           all_guns=all_guns,
                           all_targets=all_targets)

@bp_main.route("/range_visits")
def range_visits():
    visits_json = os.path.join(current_app.config["JSON_FOLDER"], "range_visits.json")
    range_visits = load_data(visits_json)

    range_visits.sort(
        key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"),
        reverse=True
    )

    ranges_json = os.path.join(current_app.config["JSON_FOLDER"], "ranges.json")
    all_ranges = load_data(ranges_json)

    def get_range_name(r_id):
        for r in all_ranges:
            if r["id"] == r_id:
                return r["name"]
        return "Unknown"

    return render_template("range_visits.html", 
                           visits=range_visits, 
                           get_range_name=get_range_name)

@bp_main.route("/delete_visit/<int:visit_id>", methods=["POST"])
def delete_visit(visit_id):
    visits_json = os.path.join(current_app.config["JSON_FOLDER"], "range_visits.json")
    rounds_json = os.path.join(current_app.config["JSON_FOLDER"], "rounds.json")

    range_visits = load_data(visits_json)
    rounds_list = load_data(rounds_json)

    visit_to_delete = next((v for v in range_visits if v["id"] == visit_id), None)
    if not visit_to_delete:
        flash("Range visit not found!", "danger")
        return redirect(url_for("bp_main.range_visits"))

    rounds_to_delete = [r for r in rounds_list if r["range_visit_id"] == visit_id]
    for r in rounds_to_delete:
        if r["photo"]:
            photo_path = os.path.join(current_app.config["UPLOAD_FOLDER"], r["photo"])
            if os.path.exists(photo_path):
                os.remove(photo_path)
        rounds_list.remove(r)

    range_visits.remove(visit_to_delete)

    save_data(visits_json, range_visits)
    save_data(rounds_json, rounds_list)
    flash("Range visit deleted!", "success")
    return redirect(url_for("bp_main.range_visits"))

@bp_main.route("/view_visit/<int:visit_id>")
def view_visit(visit_id):
    visits_json = os.path.join(current_app.config["JSON_FOLDER"], "range_visits.json")
    rounds_json = os.path.join(current_app.config["JSON_FOLDER"], "rounds.json")
    ranges_json = os.path.join(current_app.config["JSON_FOLDER"], "ranges.json")
    guns_json = os.path.join(current_app.config["JSON_FOLDER"], "guns.json")
    targets_json = os.path.join(current_app.config["JSON_FOLDER"], "target_types.json")

    range_visits = load_data(visits_json)
    rounds_list = load_data(rounds_json)
    all_ranges = load_data(ranges_json)
    all_guns = load_data(guns_json)
    all_targets = load_data(targets_json)

    visit = next((v for v in range_visits if v["id"] == visit_id), None)
    if not visit:
        flash("Range visit not found!", "danger")
        return redirect(url_for("bp_main.range_visits"))

    visit_rounds = [r for r in rounds_list if r["range_visit_id"] == visit_id]

    def get_range_name(r_id):
        found = next((rng for rng in all_ranges if rng["id"] == r_id), None)
        return found["name"] if found else "Unknown"

    def get_gun_name(g_id):
        found = next((g for g in all_guns if g["id"] == g_id), None)
        return found["name"] if found else "Unknown"

    def get_target_name(t_id):
        found = next((t for t in all_targets if t["id"] == t_id), None)
        return found["name"] if found else "Unknown"

    return render_template("view_visit.html",
                           visit=visit,
                           visit_rounds=visit_rounds,
                           get_range_name=get_range_name,
                           get_gun_name=get_gun_name,
                           get_target_name=get_target_name)

@bp_main.route("/statistics", methods=["GET", "POST"])
def statistics():
    visits_json = os.path.join(current_app.config["JSON_FOLDER"], "range_visits.json")
    rounds_json = os.path.join(current_app.config["JSON_FOLDER"], "rounds.json")
    ranges_json = os.path.join(current_app.config["JSON_FOLDER"], "ranges.json")
    guns_json = os.path.join(current_app.config["JSON_FOLDER"], "guns.json")
    targets_json = os.path.join(current_app.config["JSON_FOLDER"], "target_types.json")
    shots_json_path = os.path.join(current_app.config["JSON_FOLDER"], "shots.json")

    all_visits = load_data(visits_json)
    all_rounds = load_data(rounds_json)
    all_ranges = load_data(ranges_json)
    all_guns = load_data(guns_json)
    all_targets = load_data(targets_json)
    all_shots = load_data(shots_json_path)

    filter_range = request.form.get("filter_range")
    filter_gun = request.form.get("filter_gun")
    filter_target = request.form.get("filter_target")
    filter_distance = request.form.get("filter_distance")
    filter_recents = request.form.get("filter_recents")

    def to_int_or_none(value):
        if not value or value == "alle":
            return None
        return int(value)

    filter_range_id = to_int_or_none(filter_range)
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
        except:
            filter_recents_val = None

    for v in all_visits:
        v["_dt"] = datetime.strptime(v["date"], "%Y-%m-%d")
    all_visits.sort(key=lambda x: x["_dt"], reverse=True)

    if filter_recents_val:
        all_visits = all_visits[:filter_recents_val]

    if filter_range_id is not None:
        all_visits = [v for v in all_visits if v["range_id"] == filter_range_id]

    visit_map = {v["id"]: v for v in all_visits}

    filtered_rounds = []
    for r in all_rounds:
        if r["range_visit_id"] not in visit_map:
            continue

        if filter_gun_id is not None and r["gun_id"] != filter_gun_id:
            continue

        if filter_target_id is not None and r["target_type_id"] != filter_target_id:
            continue

        if filter_distance_val is not None and r.get("distance") != filter_distance_val:
            continue

        filtered_rounds.append(r)


    date_score_map = {}
    for r in filtered_rounds:
        visit_obj = visit_map[r["range_visit_id"]]
        date_str = visit_obj["date"]
        if date_str not in date_score_map:
            date_score_map[date_str] = []
        date_score_map[date_str].append(r["score"])

    date_avg_score = []
    for d_str, scores in date_score_map.items():
        dt = datetime.strptime(d_str, "%Y-%m-%d")
        avg_s = sum(scores) / len(scores) if scores else 0
        date_avg_score.append((dt, avg_s))

    date_avg_score.sort(key=lambda x: x[0])
    labels_avg_score = [x[0].strftime("%Y-%m-%d") for x in date_avg_score]
    data_avg_score = [x[1] for x in date_avg_score]


    visits_for_chart = []
    all_visits.sort(key=lambda x: x["_dt"])

    for v in all_visits:
        v_rnds = [rnd for rnd in filtered_rounds if rnd["range_visit_id"] == v["id"]]
        if v_rnds:
            avg_score = sum(r["score"] for r in v_rnds) / len(v_rnds)
            visits_for_chart.append((v["date"], avg_score))

    labels_visit_chart = [x[0] for x in visits_for_chart]
    data_visit_chart = [x[1] for x in visits_for_chart]

    filtered_round_ids = {r["id"] for r in filtered_rounds}

    shots_for_heatmap = []
    for entry in all_shots:
        if entry["round_id"] in filtered_round_ids:
            for h in entry["hits"]:
                shots_for_heatmap.append({"x": h["x"], "y": h["y"]})

    return render_template("statistics.html",
                           all_ranges=all_ranges,
                           all_guns=all_guns,
                           all_targets=all_targets,
                           filter_range=filter_range,
                           filter_gun=filter_gun,
                           filter_target=filter_target,
                           filter_distance=filter_distance,
                           filter_recents=filter_recents,
                           labels_avg_score=labels_avg_score,
                           data_avg_score=data_avg_score,
                           labels_visit_chart=labels_visit_chart,
                           data_visit_chart=data_visit_chart,
                           shots_for_heatmap=shots_for_heatmap)


@bp_main.route("/interactive_target/<int:round_id>", methods=["GET", "POST"])
def interactive_target(round_id):
    shots_json_path = os.path.join(current_app.config["JSON_FOLDER"], "shots.json")
    shots_data = load_data(shots_json_path)
    
    existing_entry = next((item for item in shots_data if item["round_id"] == round_id), None)

    if request.method == "POST":
        total_shots = request.form.get("total_shots", type=int)
        hits_json_str = request.form.get("hits_json")

        import json
        try:
            hits_array = json.loads(hits_json_str) if hits_json_str else []
        except:
            hits_array = []

        if existing_entry:
            existing_entry["total_shots"] = total_shots
            existing_entry["hits"] = hits_array
        else:
            new_entry = {
                "round_id": round_id,
                "total_shots": total_shots,
                "hits": hits_array
            }
            shots_data.append(new_entry)

        save_data(shots_json_path, shots_data)
        flash("Interactive target data saved!", "success")
        return redirect(url_for("bp_main.view_visit", visit_id=_find_visit_id_from_round(round_id)))

    if not existing_entry:
        existing_entry = {
            "round_id": round_id,
            "total_shots": 0,
            "hits": []
        }

    return render_template("interactive_target.html",
                           round_id=round_id,
                           total_shots=existing_entry["total_shots"],
                           hits=existing_entry["hits"])

def _find_visit_id_from_round(round_id):
    rounds_json = os.path.join(current_app.config["JSON_FOLDER"], "rounds.json")
    all_rounds = load_data(rounds_json)
    this_round = next((r for r in all_rounds if r["id"] == round_id), None)
    return this_round["range_visit_id"] if this_round else None

