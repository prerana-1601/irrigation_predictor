from flask import Blueprint, jsonify
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
from decimal import Decimal
import boto3, os

from ..auth_utils import require_auth, parse_token

bp = Blueprint('fields', __name__, url_prefix='/api/fields')

REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
FIELDS_TABLE_NAME = os.getenv("DDB_FIELDS_TABLE", "fields")

ddb = boto3.resource("dynamodb", region_name=REGION)
fields_table = ddb.Table(FIELDS_TABLE_NAME)

def _to_json(v):
    if isinstance(v, list):
        return [_to_json(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_json(x) for k, x in v.items()}
    if isinstance(v, Decimal):
        # render as int if it’s whole number, else float
        return int(v) if v % 1 == 0 else float(v)
    return v

def _field_public(item):
    # normalize DynamoDB item -> response shape you already use on the frontend
    return {
        "id": int(item["field_id"]),
        "name": item.get("name", ""),
        "location": item.get("location", ""),
        "owner_user_id": int(item.get("owner_user_id", 0)),
        "irrigation_needed": bool(item.get("irrigation_needed", False)),
        "sensorData": _to_json(item.get("sensorData", [])),
    }

@bp.get("/")
@require_auth()  # anyone logged in
def list_fields_for_me():
    """
    Users -> only their fields (query by owner_user_id via owner-index).
    Admins -> all fields (scan).
    """
    auth = parse_token()
    user_id = int(auth["sub"])
    role = auth.get("role", "user")

    try:
        items = []
        if role == "admin":
            # admins see everything
            resp = fields_table.scan()
            items.extend(resp.get("Items", []))
            while "LastEvaluatedKey" in resp:
                resp = fields_table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"])
                items.extend(resp.get("Items", []))
        else:
            # query by owner_user_id (requires GSI: owner-index)
            resp = fields_table.query(
                IndexName="owner-index",
                KeyConditionExpression=Key("owner_user_id").eq(user_id),
            )
            items.extend(resp.get("Items", []))

        data = [_field_public(it) for it in items]
        return jsonify(data)

    except ClientError as e:
        return jsonify({"error": f"DDB error: {e.response['Error']['Message']}"}), 500

@bp.get("/<int:field_id>")
@require_auth()
def get_field(field_id: int):
    """
    Return one field. Users can only read their own field; admins can read any.
    """
    auth = parse_token()
    user_id = int(auth["sub"])
    role = auth.get("role", "user")

    try:
        resp = fields_table.get_item(Key={"field_id": field_id})
        item = resp.get("Item")
        if not item:
            return jsonify({"error": "Not found"}), 404

        # enforce ownership for non-admins
        owner_id = int(item.get("owner_user_id", 0))
        if role != "admin" and owner_id != user_id:
            return jsonify({"error": "Forbidden"}), 403

        return jsonify(_field_public(item))

    except ClientError as e:
        return jsonify({"error": f"DDB error: {e.response['Error']['Message']}"}), 500
