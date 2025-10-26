from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
import boto3, os, uuid, time

# ⬇️ relative import (fixes unresolved reference)
from ..auth_utils import make_token

bp = Blueprint('auth', __name__, url_prefix='/api/auth')

# ---- Config / AWS ----
REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
TABLE_NAME = os.getenv("DDB_USERS_TABLE", "users")

ddb = boto3.resource("dynamodb", region_name=REGION)
users_table = ddb.Table(TABLE_NAME)

# ---- Helpers ----
def _now_ms() -> int:
    return int(time.time() * 1000)

def _user_public(u: dict) -> dict:
    return {
        "id": int(u["user_id"]),
        "name": u.get("name", ""),
        "email": u.get("email", ""),
        "role": u.get("role", "user"),
        "fields": [int(x) for x in u.get("fields", [])],
    }

def _hash_password(password: str) -> str:
    # Force PBKDF2 so we don't depend on scrypt availability
    return generate_password_hash(password, method="pbkdf2:sha256", salt_length=16)

def _password_ok(stored: str, plain: str) -> bool:
    # Werkzeug hashes contain ":" (e.g., "pbkdf2:sha256:...")
    if ":" in (stored or ""):
        return check_password_hash(stored, plain)
    # Legacy/plaintext fallback
    return stored == plain

# ---- Routes ----
@bp.post('/signup')
def signup():
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "email and password required"}), 400

    # Check if email already exists (requires GSI: email-index on "email")
    try:
        existing = users_table.query(
            IndexName="email-index",
            KeyConditionExpression=Key("email").eq(email),
            Limit=1,
        )
        if existing.get("Items"):
            return jsonify({"error": "email already registered"}), 409
    except ClientError as e:
        return jsonify({"error": f"DDB query failed: {e.response['Error']['Message']}"}), 500

    user_id = uuid.uuid4().int >> 96  # short numeric ID
    item = {
        "user_id": user_id,
        "name": name or "User",
        "email": email,
        "password": _hash_password(password),
        "role": "user",
        "fields": [],
        "created_at": _now_ms(),
    }
    try:
        users_table.put_item(
            Item=item,
            ConditionExpression="attribute_not_exists(user_id)"
        )
    except ClientError as e:
        return jsonify({"error": f"Create failed: {e.response['Error']['Message']}"}), 500

    token = make_token(user_id, item["role"])
    return jsonify({"message": "ok", "user": _user_public(item), "token": token}), 201


@bp.post('/login')
def login():
    data = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify({"error": "email and password required"}), 400

    try:
        resp = users_table.query(
            IndexName="email-index",
            KeyConditionExpression=Key("email").eq(email),
            Limit=1,
        )
        items = resp.get("Items", [])
        if not items:
            return jsonify({"error": "Invalid credentials"}), 401

        user = items[0]
        stored = user.get("password", "") or ""
        ok = _password_ok(stored, password)

        # If legacy plaintext matched, upgrade it to PBKDF2 immediately
        if ok and ":" not in stored:
            try:
                new_hash = _hash_password(password)
                users_table.update_item(
                    Key={"user_id": user["user_id"]},
                    UpdateExpression="SET #p = :hpw",
                    ExpressionAttributeNames={"#p": "password"},
                    ExpressionAttributeValues={":hpw": new_hash},
                )
                user["password"] = new_hash
            except ClientError:
                # Non-fatal: auth succeeded; we'll just try again next login
                pass

        if not ok:
            return jsonify({"error": "Invalid credentials"}), 401

    except ClientError as e:
        return jsonify({"error": f"DDB query failed: {e.response['Error']['Message']}"}), 500

    token = make_token(int(user["user_id"]), user.get("role", "user"))
    return jsonify({"token": token, "user": _user_public(user)}), 200


# (Optional) quick diagnostics while debugging; remove in production.
@bp.get('/_diag')
def _diag():
    try:
        meta = users_table.meta.client.describe_table(TableName=TABLE_NAME)
        has_index = any(
            i.get("IndexName") == "email-index"
            for i in meta["Table"].get("GlobalSecondaryIndexes", [])
        )
        sample_email = (request.args.get("email") or "demo@hansal.ai").lower()
        q = users_table.query(
            IndexName="email-index",
            KeyConditionExpression=Key("email").eq(sample_email),
            Limit=1,
        )
        return jsonify({
            "region": os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")),
            "table": TABLE_NAME,
            "email_index_present": has_index,
            "queried_email": sample_email,
            "found_users": len(q.get("Items", [])),
        })
    except Exception as e:
        return jsonify({"error": str(e), "table": TABLE_NAME}), 500
