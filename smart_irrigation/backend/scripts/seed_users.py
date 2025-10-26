#!/usr/bin/env python3
"""
Seed/demo users for the Hansal Smart Irrigation app.

Usage (from the repo's backend/ folder):
  source .venv/bin/activate
  export AWS_REGION=us-east-1
  export DDB_USERS_TABLE=users
  python scripts/seed_users.py

Requires: boto3, werkzeug (already in requirements.txt)
"""

import os
import time
import uuid
from typing import List, Dict

import boto3
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key
from werkzeug.security import generate_password_hash

REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
TABLE_NAME = os.getenv("DDB_USERS_TABLE", "users")

ddb = boto3.resource("dynamodb", region_name=REGION)
table = ddb.Table(TABLE_NAME)

PBKDF2 = dict(method="pbkdf2:sha256", salt_length=16)
now_ms = lambda: int(time.time() * 1000)

# ---------------------------------------------------------------------------
# Demo dataset
# Passwords are plaintext here; script will hash them before writing.
# ---------------------------------------------------------------------------
DEMO_USERS: List[Dict] = [
    # Admin
    {"name": "System Admin", "email": "admin@hansal.ai", "password": "admin123", "role": "admin", "fields": []},

    # Normal users
    {"name": "Demo User",   "email": "demo@hansal.ai",   "password": "hansal123", "role": "user", "fields": [1, 2]},
    {"name": "Sarah Farmer","email": "sarah@hansal.ai",  "password": "sarah123",  "role": "user", "fields": [3]},
    {"name": "Miguel Rivera","email":"miguel@hansal.ai", "password": "miguel123", "role": "user", "fields": [4, 5]},
    {"name": "Priya Patel", "email": "priya@hansal.ai",  "password": "priya123",  "role": "user", "fields": [6, 7]},
    {"name": "Chen Li",     "email": "chen@hansal.ai",   "password": "chen123",   "role": "user", "fields": [8]},
    {"name": "Aisha Khan",  "email": "aisha@hansal.ai",  "password": "aisha123",  "role": "user", "fields": [9, 10]},
    {"name": "Luca Romano", "email": "luca@hansal.ai",   "password": "luca123",   "role": "user", "fields": [11]},
    {"name": "Grace Kim",   "email": "grace@hansal.ai",  "password": "grace123",  "role": "user", "fields": [12, 13]},
]

def short_user_id() -> int:
    """Generate a short-ish numeric user_id (compatible with app)."""
    return uuid.uuid4().int >> 96

def ensure_email_index():
    """Warn if email-index GSI is missing."""
    try:
        meta = table.meta.client.describe_table(TableName=TABLE_NAME)
    except ClientError as e:
        raise SystemExit(f"DescribeTable failed: {e.response['Error']['Message']}")
    gsis = (meta["Table"].get("GlobalSecondaryIndexes") or [])
    if not any(g.get("IndexName") == "email-index" for g in gsis):
        print("WARNING: GSI 'email-index' not found on table "
              f"'{TABLE_NAME}'. The seeder will still run, but lookups by email will fail.")
        print("Create a GSI on attribute 'email' (HASH) named exactly: email-index\n")

def upsert_user(u: Dict) -> str:
    """
    If email exists: update name/role/fields/password, set updated_at.
    If not: create a new item with user_id and created_at.
    Returns 'created' or 'updated'.
    """
    email = u["email"].strip().lower()
    # Lookup by email (via GSI)
    resp = table.query(IndexName="email-index", KeyConditionExpression=Key("email").eq(email), Limit=1)
    items = resp.get("Items", [])
    hashed = generate_password_hash(u["password"], **PBKDF2)

    if items:
        user = items[0]
        # Update existing
        table.update_item(
            Key={"user_id": user["user_id"]},
            UpdateExpression="SET #n=:n, #r=:r, #f=:f, #p=:p, updated_at=:ts",
            ExpressionAttributeNames={"#n": "name", "#r": "role", "#f": "fields", "#p": "password"},
            ExpressionAttributeValues={
                ":n": u["name"],
                ":r": u.get("role", "user"),
                ":f": list(map(int, u.get("fields", []))),
                ":p": hashed,
                ":ts": now_ms(),
            },
        )
        return "updated"
    else:
        # Create new
        item = {
            "user_id": short_user_id(),
            "name": u["name"],
            "email": email,
            "password": hashed,
            "role": u.get("role", "user"),
            "fields": list(map(int, u.get("fields", []))),
            "created_at": now_ms(),
        }
        table.put_item(Item=item)
        return "created"

def main():
    print(f"Seeding users into DynamoDB table '{TABLE_NAME}' in region '{REGION}'...")
    ensure_email_index()
    created = updated = 0
    for u in DEMO_USERS:
        try:
            result = upsert_user(u)
            if result == "created":
                created += 1
            else:
                updated += 1
            print(f"  {u['email']:<24} -> {result}")
        except ClientError as e:
            print(f"  {u['email']:<24} -> ERROR: {e.response['Error']['Message']}")
    print(f"Done. Created: {created}, Updated: {updated}")

if __name__ == "__main__":
    main()
