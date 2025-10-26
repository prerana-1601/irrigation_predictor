from flask import Blueprint, jsonify, request
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key
from decimal import Decimal
import boto3
import os
import time
from typing import Any, Dict, List

from ..auth_utils import require_auth  # Python 3.9 safe relative import

bp = Blueprint("admin", __name__, url_prefix="/api/admin")

REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
USERS_TABLE_NAME = os.getenv("DDB_USERS_TABLE", "users")
FIELDS_TABLE_NAME = os.getenv("DDB_FIELDS_TABLE", "fields")

ddb = boto3.resource("dynamodb", region_name=REGION)
users_table = ddb.Table(USERS_TABLE_NAME)
fields_table = ddb.Table(FIELDS_TABLE_NAME)


# ----------------- helpers -----------------
def _to_json(v):
    if isinstance(v, list):
        return [_to_json(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_json(x) for k, x in v.items()}
    if isinstance(v, Decimal):
        return int(v) if v % 1 == 0 else float(v)
    return v


def _to_ddb(v: Any):
    if isinstance(v, float):
        return Decimal(str(v))
    if isinstance(v, list):
        return [_to_ddb(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_ddb(x) for k, x in v.items()}
    return v


def _user_public(u: dict) -> dict:
    return {
        "id": int(u["user_id"]),
        "name": u.get("name", ""),
        "email": u.get("email", ""),
        "role": u.get("role", "user"),
        "fields": [int(x) for x in u.get("fields", [])],
    }


def _user_name(user_id: int) -> str:
    try:
        r = users_table.get_item(Key={"user_id": int(user_id)})
        item = r.get("Item")
        return (item or {}).get("name", "")
    except ClientError:
        return ""


def _add_field_to_user(user_id: int, field_id: int):
    """Ensure field_id exists in user's fields list."""
    r = users_table.get_item(Key={"user_id": user_id})
    item = r.get("Item")
    if not item:
        return
    fields = list(map(int, item.get("fields", [])))
    if field_id not in fields:
        fields.append(field_id)
        users_table.update_item(
            Key={"user_id": user_id},
            UpdateExpression="SET #f=:f",
            ExpressionAttributeNames={"#f": "fields"},
            ExpressionAttributeValues={":f": fields},
        )


def _remove_field_from_user(user_id: int, field_id: int):
    r = users_table.get_item(Key={"user_id": user_id})
    item = r.get("Item")
    if not item:
        return
    fields = [int(x) for x in item.get("fields", []) if int(x) != field_id]
    users_table.update_item(
        Key={"user_id": user_id},
        UpdateExpression="SET #f=:f",
        ExpressionAttributeNames={"#f": "fields"},
        ExpressionAttributeValues={":f": fields},
    )


# ----------------- users list (unchanged) -----------------
@bp.get("/users")
@require_auth(role="admin")
def users():
    """List all non-admin users (for Admin dashboard)."""
    items: List[Dict] = []
    try:
        resp = users_table.scan()
        items.extend(resp.get("Items", []))
        while "LastEvaluatedKey" in resp:
            resp = users_table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"])
            items.extend(resp.get("Items", []))
    except ClientError as e:
        return jsonify({"error": f"DDB scan failed: {e.response['Error']['Message']}"}), 500

    out = []
    for u in items:
        if u.get("role") == "admin":
            continue
        out.append(_user_public(u))
    return jsonify(out)


# ----------------- read fields (with owner_name) -----------------
@bp.get("/fields")
@require_auth(role="admin")
def all_fields():
    """Return all fields (as before) + owner_name."""
    try:
        items = []
        resp = fields_table.scan()
        items.extend(resp.get("Items", []))
        while "LastEvaluatedKey" in resp:
            resp = fields_table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"])
            items.extend(resp.get("Items", []))

        owner_ids = sorted({int(it.get("owner_user_id", 0)) for it in items if "owner_user_id" in it})
        owner_name_by_id = {uid: _user_name(uid) for uid in owner_ids}

        data = []
        for it in items:
            owner_id = int(it.get("owner_user_id", 0))
            data.append(
                {
                    "id": int(it["field_id"]),
                    "name": it.get("name", ""),
                    "location": it.get("location", ""),
                    "owner_user_id": owner_id,
                    "owner_name": owner_name_by_id.get(owner_id, ""),
                    "irrigation_needed": bool(it.get("irrigation_needed", False)),
                    "sensorData": _to_json(it.get("sensorData", [])),
                }
            )
        return jsonify(data)
    except ClientError as e:
        return jsonify({"error": f"DDB scan failed: {e.response['Error']['Message']}"}), 500


@bp.get("/users/<int:user_id>/fields")
@require_auth(role="admin")
def fields_for_user(user_id: int):
    """Return one user's info + only their fields."""
    try:
        user_resp = users_table.get_item(Key={"user_id": user_id})
        user_item = user_resp.get("Item")
        if not user_item:
            return jsonify({"error": "User not found"}), 404

        f_resp = fields_table.query(
            IndexName="owner-index",
            KeyConditionExpression=Key("owner_user_id").eq(user_id),
        )
        owner_name = user_item.get("name", "")
        fields = []
        for it in f_resp.get("Items", []):
            fields.append(
                {
                    "id": int(it["field_id"]),
                    "name": it.get("name", ""),
                    "location": it.get("location", ""),
                    "owner_user_id": int(it.get("owner_user_id", 0)),
                    "owner_name": owner_name,
                    "irrigation_needed": bool(it.get("irrigation_needed", False)),
                    "sensorData": _to_json(it.get("sensorData", [])),
                }
            )
        return jsonify({"user": _user_public(user_item), "fields": fields})
    except ClientError as e:
        return jsonify({"error": f"DDB error: {e.response['Error']['Message']}"}), 500


# ----------------- CRUD: create / update / delete fields -----------------
def _new_field_id() -> int:
    # simple unique-ish int id for demo
    return int(time.time() * 1000) % 1_000_000_000


@bp.post("/fields")
@require_auth(role="admin")
def create_field():
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    location = (data.get("location") or "").strip()
    owner_user_id = int(data.get("owner_user_id") or 0)
    irrigation_needed = bool(data.get("irrigation_needed", False))
    sensor_data = data.get("sensorData", [])

    if not name or not owner_user_id:
        return jsonify({"error": "name and owner_user_id are required"}), 400

    field_id = int(data.get("id") or _new_field_id())

    item = {
        "field_id": field_id,
        "name": name,
        "location": location,
        "owner_user_id": owner_user_id,
        "irrigation_needed": irrigation_needed,
        "sensorData": sensor_data,
    }
    try:
        fields_table.put_item(Item=_to_ddb(item), ConditionExpression="attribute_not_exists(field_id)")
        _add_field_to_user(owner_user_id, field_id)
        # return normalized json
        item["owner_name"] = _user_name(owner_user_id)
        item["sensorData"] = _to_json(item["sensorData"])
        return jsonify(item), 201
    except ClientError as e:
        return jsonify({"error": f"Create failed: {e.response['Error']['Message']}"}), 500


@bp.put("/fields/<int:field_id>")
@require_auth(role="admin")
def update_field(field_id: int):
    data = request.get_json(force=True) or {}
    # fetch existing
    try:
        r = fields_table.get_item(Key={"field_id": field_id})
        old = r.get("Item")
        if not old:
            return jsonify({"error": "Not found"}), 404
    except ClientError as e:
        return jsonify({"error": f"Read failed: {e.response['Error']['Message']}"}), 500

    # compute new values
    name = data.get("name", old.get("name"))
    location = data.get("location", old.get("location"))
    irrigation_needed = data.get("irrigation_needed", old.get("irrigation_needed", False))
    owner_user_id_new = int(data.get("owner_user_id", old.get("owner_user_id", 0)))
    sensor_data = data.get("sensorData", old.get("sensorData", []))

    # write back
    try:
        fields_table.put_item(
            Item=_to_ddb(
                {
                    "field_id": field_id,
                    "name": name,
                    "location": location,
                    "owner_user_id": owner_user_id_new,
                    "irrigation_needed": bool(irrigation_needed),
                    "sensorData": sensor_data,
                }
            )
        )
        # if owner changed, update users arrays
        owner_user_id_old = int(old.get("owner_user_id", 0))
        if owner_user_id_new != owner_user_id_old:
            if owner_user_id_old:
                _remove_field_from_user(owner_user_id_old, field_id)
            if owner_user_id_new:
                _add_field_to_user(owner_user_id_new, field_id)

        out = {
            "id": field_id,
            "name": name,
            "location": location,
            "owner_user_id": owner_user_id_new,
            "owner_name": _user_name(owner_user_id_new),
            "irrigation_needed": bool(irrigation_needed),
            "sensorData": _to_json(sensor_data),
        }
        return jsonify(out)
    except ClientError as e:
        return jsonify({"error": f"Update failed: {e.response['Error']['Message']}"}), 500


@bp.delete("/fields/<int:field_id>")
@require_auth(role="admin")
def delete_field(field_id: int):
    try:
        r = fields_table.get_item(Key={"field_id": field_id})
        item = r.get("Item")
        if not item:
            return jsonify({"ok": True})  # already gone

        owner_user_id = int(item.get("owner_user_id", 0))
        fields_table.delete_item(Key={"field_id": field_id})
        if owner_user_id:
            _remove_field_from_user(owner_user_id, field_id)
        return jsonify({"ok": True})
    except ClientError as e:
        return jsonify({"error": f"Delete failed: {e.response['Error']['Message']}"}), 500


# ----------------- assign existing field(s) to a user -----------------
@bp.post("/users/<int:user_id>/fields/assign")
@require_auth(role="admin")
def assign_fields(user_id: int):
    """
    Body: {"field_id": 123}  OR  {"field_ids":[1,2,3]}
    Reassigns each field to this user_id.
    """
    payload = request.get_json(force=True) or {}
    ids = payload.get("field_ids")
    if ids is None and "field_id" in payload:
        ids = [payload["field_id"]]
    if not ids:
        return jsonify({"error": "field_id or field_ids is required"}), 400

    changed: List[int] = []
    for fid in ids:
        fid = int(fid)
        # read current
        r = fields_table.get_item(Key={"field_id": fid})
        item = r.get("Item")
        if not item:
            continue
        prev_owner = int(item.get("owner_user_id", 0))
        # write new owner
        item["owner_user_id"] = user_id
        fields_table.put_item(Item=_to_ddb(item))
        if prev_owner:
            _remove_field_from_user(prev_owner, fid)
        _add_field_to_user(user_id, fid)
        changed.append(fid)

    return jsonify({"ok": True, "reassigned": changed})
