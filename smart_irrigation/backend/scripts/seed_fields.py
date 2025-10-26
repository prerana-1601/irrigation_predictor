#!/usr/bin/env python3
import os, json, math, random
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

REGION            = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
USERS_TABLE_NAME  = os.getenv("DDB_USERS_TABLE", "users")
FIELDS_TABLE_NAME = os.getenv("DDB_FIELDS_TABLE", "fields")

ddb = boto3.resource("dynamodb", region_name=REGION)
ddbc = boto3.client("dynamodb", region_name=REGION)
users_table  = ddb.Table(USERS_TABLE_NAME)
fields_table = ddb.Table(FIELDS_TABLE_NAME)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
JSON_PATH = os.path.join(DATA_DIR, "fields.json")

FIELDS_SPEC: List[Dict[str, Any]] = [
    {"id": 1,  "name": "Field Alpha",   "location": "North Farm",  "owner_email": "demo@hansal.ai"},
    {"id": 2,  "name": "Field Beta",    "location": "North Farm",  "owner_email": "demo@hansal.ai"},
    {"id": 3,  "name": "Field Gamma",   "location": "East Farm",   "owner_email": "sarah@hansal.ai"},
    {"id": 4,  "name": "Field Delta",   "location": "West Farm",   "owner_email": "miguel@hansal.ai"},
    {"id": 5,  "name": "Field Epsilon", "location": "West Farm",   "owner_email": "miguel@hansal.ai"},
    {"id": 6,  "name": "Field Zeta",    "location": "South Farm",  "owner_email": "priya@hansal.ai"},
    {"id": 7,  "name": "Field Eta",     "location": "South Farm",  "owner_email": "priya@hansal.ai"},
    {"id": 8,  "name": "Field Theta",   "location": "Hill Farm",   "owner_email": "chen@hansal.ai"},
    {"id": 9,  "name": "Field Iota",    "location": "River Farm",  "owner_email": "aisha@hansal.ai"},
    {"id": 10, "name": "Field Kappa",   "location": "River Farm",  "owner_email": "aisha@hansal.ai"},
    {"id": 11, "name": "Field Lambda",  "location": "Valley Farm", "owner_email": "luca@hansal.ai"},
    {"id": 12, "name": "Field Mu",      "location": "Plateau",     "owner_email": "grace@hansal.ai"},
    {"id": 13, "name": "Field Nu",      "location": "Plateau",     "owner_email": "grace@hansal.ai"},
]

def get_user_id_by_email(email: str) -> int:
    resp = users_table.query(
        IndexName="email-index",
        KeyConditionExpression=Key("email").eq(email.lower().strip()),
        Limit=1,
    )
    items = resp.get("Items", [])
    if not items:
        raise RuntimeError(f"user not found for email: {email} (did you seed users first?)")
    return int(items[0]["user_id"])

def make_series(hours: int = 48, step_hours: int = 4) -> List[Dict[str, float]]:
    now = datetime.now(timezone.utc)
    points: List[Dict[str, float]] = []
    base_moisture = random.uniform(28, 45)
    for i in range(0, hours + 1, step_hours):
        t = now - timedelta(hours=(hours - i))
        temp = 24 + 6 * math.sin(2 * math.pi * (t.hour / 24.0)) + random.uniform(-1.2, 1.2)
        humidity = 70 + random.uniform(-8, 8) - (temp - 24) * 0.8
        humidity = max(35, min(95, humidity))
        rainfall = random.choice([0, 0, 0, 0, random.uniform(0.5, 6.0)])
        drift = -0.25 * (temp - 24) + 0.9 * rainfall + random.uniform(-0.8, 0.8)
        base_moisture = max(8, min(60, base_moisture + drift * 0.2))
        points.append({
            "timestamp": t.isoformat(),
            "temperature": round(temp, 1),
            "humidity": round(humidity, 1),
            "soil_moisture": round(base_moisture, 1),
            "rainfall": round(rainfall, 2),
        })
    return points

def to_irrigation_needed(series: List[Dict[str, float]]) -> bool:
    return bool(series and series[-1]["soil_moisture"] < 25.0)

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def write_json(payload: List[Dict[str, Any]]):
    ensure_data_dir()
    with open(JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote demo fields JSON -> {JSON_PATH}")

def to_ddb(value):
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, list):
        return [to_ddb(v) for v in value]
    if isinstance(value, dict):
        return {k: to_ddb(v) for k, v in value.items()}
    return value

def ensure_fields_table():
    try:
        ddbc.describe_table(TableName=FIELDS_TABLE_NAME)
        return
    except ddbc.exceptions.ResourceNotFoundException:
        print(f"Creating DynamoDB table '{FIELDS_TABLE_NAME}' in {REGION} ...")
        ddbc.create_table(
            TableName=FIELDS_TABLE_NAME,
            AttributeDefinitions=[
                {"AttributeName": "field_id", "AttributeType": "N"},
                {"AttributeName": "owner_user_id", "AttributeType": "N"},
            ],
            KeySchema=[{"AttributeName": "field_id", "KeyType": "HASH"}],
            BillingMode="PAY_PER_REQUEST",
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "owner-index",
                    "KeySchema": [{"AttributeName": "owner_user_id", "KeyType": "HASH"}],
                    "Projection": {"ProjectionType": "ALL"},
                }
            ],
        )
        ddbc.get_waiter("table_exists").wait(TableName=FIELDS_TABLE_NAME)
        print("Table is ACTIVE.")

def upsert_fields_ddb(items: List[Dict[str, Any]]):
    ensure_fields_table()
    print(f"Upserting {len(items)} fields into DynamoDB table '{FIELDS_TABLE_NAME}' (region {REGION}) ...")
    for it in items:
        ddb_item = to_ddb({
            "field_id": it["id"],
            "name": it["name"],
            "location": it["location"],
            "owner_user_id": it["owner_user_id"],
            "irrigation_needed": it["irrigation_needed"],
            "sensorData": it["sensorData"],
        })
        fields_table.put_item(Item=ddb_item)
    print("DynamoDB upsert complete.")

def main():
    full: List[Dict[str, Any]] = []
    for spec in FIELDS_SPEC:
        owner_id = get_user_id_by_email(spec["owner_email"])
        series = make_series(hours=48, step_hours=4)
        full.append({
            "id": int(spec["id"]),
            "name": spec["name"],
            "location": spec["location"],
            "owner_user_id": owner_id,
            "sensorData": series,
            "irrigation_needed": to_irrigation_needed(series),
        })

    write_json(full)

    try:
        upsert_fields_ddb(full)
    except ClientError as e:
        print(f"Skipping DynamoDB upsert due to error: {e.response['Error']['Message']}")
        print("Tip: check IAM perms and region/account.")

if __name__ == "__main__":
    main()
