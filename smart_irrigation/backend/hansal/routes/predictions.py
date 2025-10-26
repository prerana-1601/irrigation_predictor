
from flask import Blueprint, jsonify
import json, os

bp = Blueprint('predictions', __name__, url_prefix='/api/predictions')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
FIELDS_PATH = os.path.join(DATA_DIR, 'fields.json')

def _load_fields():
    with open(FIELDS_PATH) as f:
        return json.load(f)

@bp.get('/')
def predictions():
    # Naive logic for demo: if latest soil_moisture < 25 => irrigation needed
    out = []
    for f in _load_fields():
        latest = f['sensorData'][-1] if f['sensorData'] else None
        needed = False
        if latest and latest.get('soil_moisture') is not None:
            needed = latest['soil_moisture'] < 25.0
        out.append({
            "field_id": f['id'],
            "field_name": f['name'],
            "irrigation_needed": bool(needed)
        })
    return jsonify(out)

# Admin demo route
admin_bp = Blueprint('admin', __name__, url_prefix='/api/admin')

USERS_PATH = os.path.join(DATA_DIR, 'users.json')

def _load_users():
    with open(USERS_PATH) as f:
        return json.load(f)

@admin_bp.get('/users')
def users():
    users = _load_users()
    for u in users:
        u.pop('password', None)
    return jsonify(users)
