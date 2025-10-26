'''import os
import json
import datetime
import urllib.request
import urllib.parse
import boto3

# --------- Region & Clients ----------
REGION = os.environ.get("AWS_REGION", "us-east-1")
secrets = boto3.client("secretsmanager", region_name=REGION)
ses = boto3.client("ses", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)

# --------- Required Environment Variables ----------
# Coordinates & app config
LAT = os.environ["LAT"]                 # e.g., "-37.8142454"
LON = os.environ["LON"]                 # e.g., "144.9631732"
CNT = int(os.environ.get("CNT", "1"))   # 1 = tomorrow
UNITS = os.environ.get("UNITS", "metric")
FARM_NAME = os.environ.get("FARM_NAME", "Hansal Kasuweri – Pilot Farm")

# Email
SENDER_EMAIL = os.environ["SENDER_EMAIL"]  # verified in SES (same region)
RECIPIENTS = [e.strip() for e in os.environ["RECIPIENTS"].split(",")]  # verified while in sandbox

# Secrets Manager
WEATHER_SECRET_NAME = os.environ["WEATHER_SECRET_NAME"]  # e.g., "openweather/api_key"

# Bedrock
MODEL_ID = os.environ.get("MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")


# --------- Helpers ----------
def _get_openweather_key() -> str:
    """Fetch the OpenWeather API key from Secrets Manager."""
    resp = secrets.get_secret_value(SecretId=WEATHER_SECRET_NAME)
    s = resp.get("SecretString")
    try:
        return json.loads(s).get("api_key", "").strip()
    except Exception:
        return (s or "").strip()


def _fetch_climate(lat: str, lon: str, cnt: int = 1, units: str = "metric") -> dict:
    """
    Fetch climate forecast from OpenWeather PRO.
    NOTE: PRO climate often returns Kelvin even if 'units' specified;
    we handle that when interpreting temperatures.
    """
    api_key = _get_openweather_key()
    base = "https://pro.openweathermap.org/data/2.5/forecast/climate"
    qs = urllib.parse.urlencode({
        "lat": str(lat),
        "lon": str(lon),
        "cnt": str(cnt),
        "appid": api_key,
        "units": units
    })
    url = f"{base}?{qs}"
    with urllib.request.urlopen(url, timeout=25) as r:
        body = r.read().decode("utf-8")
        return json.loads(body)


def _k_to_c(val):
    if val is None:
        return None
    v = float(val)
    # If it's Kelvin (>200), convert; otherwise assume already Celsius.
    return round(v - 273.15, 1) if v > 200 else round(v, 1)


def _extract_forecast(data: dict) -> dict:
    """Pull out the fields we care about for 'tomorrow'."""
    city = (data.get("city") or {}).get("name", "Unknown")
    days = data.get("list", [])
    if not days:
        raise RuntimeError("OpenWeather returned no daily entries.")

    day = days[0]  # CNT=1 => next day
    t = day.get("temp", {})
    desc = (day.get("weather", [{}])[0] or {}).get("description", "")

    out = {
        "city": city,
        "date": datetime.date.fromtimestamp(day.get("dt", 0)).isoformat(),
        "temp_day_c": _k_to_c(t.get("day")),
        "temp_min_c": _k_to_c(t.get("min")),
        "temp_max_c": _k_to_c(t.get("max")),
        "humidity_pct": day.get("humidity"),
        "pressure_hpa": day.get("pressure"),
        "clouds_pct": day.get("clouds"),
        "wind_speed_ms": day.get("speed"),
        "wind_deg": day.get("deg"),
        "rain_mm": round(float(day.get("rain", 0) or 0), 1),
        "weather_desc": desc,
    }
    return out


def _build_bedrock_prompt(f: dict) -> dict:
    """Create Claude prompt for a short, farmer-friendly irrigation suggestion."""
    system = (
        "You are an agriculture advisor. Write a very short irrigation recommendation for a farmer "
        "based on tomorrow's forecast. Keep it <=110 words. Include: "
        "1) Irrigate/Skip, 2) Why, 3) When, 4) Approx mm of water, 5) Any precautions. Avoid jargon."
    )
    context = (
        f"Farm: {FARM_NAME}\n"
        f"Location/City: {f['city']}\n"
        f"Date: {f['date']}\n"
        f"Tomorrow: day {f['temp_day_c']}°C (min {f['temp_min_c']}°C / max {f['temp_max_c']}°C), "
        f"humidity {f['humidity_pct']}%, rain {f['rain_mm']} mm, clouds {f['clouds_pct']}%, "
        f"wind {f['wind_speed_ms']} m/s @ {f['wind_deg']}°, weather: {f['weather_desc']}.\n"
        "General rule of thumb: >=5mm expected rain usually means skipping irrigation for most field crops."
    )
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 450,
        "temperature": 0.2,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": system + "\n\n" + context}]}
        ],
    }


def _call_bedrock_claude(body: dict) -> str:
    """Invoke Amazon Bedrock (Claude 3 Haiku)."""
    resp = bedrock.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    out = json.loads(resp["body"].read())
    return out["content"][0]["text"]


def _send_email(subject: str, text_body: str):
    """Send plain-text email via SES."""
    ses.send_email(
        Source=SENDER_EMAIL,
        Destination={"ToAddresses": RECIPIENTS},
        Message={
            "Subject": {"Data": subject},
            "Body": {"Text": {"Data": text_body}},
        },
    )


def lambda_handler(event, context):
    # 1) Weather
    data = _fetch_climate(LAT, LON, CNT, UNITS)
    forecast = _extract_forecast(data)

    # 2) Bedrock summary
    prompt = _build_bedrock_prompt(forecast)
    summary = _call_bedrock_claude(prompt)

    # 3) Email
    subject = f"[{FARM_NAME}] Irrigation advice for {forecast['date']} – {forecast['city']}"
    _send_email(subject, summary)

    # 4) Return for logs / observability
    return {
        "ok": True,
        "city": forecast["city"],
        "date": forecast["date"],
        "sent_to": RECIPIENTS,
    }
'''
#code which take the data from table and model and genrate output
import os
import json
import boto3
import urllib.request
import urllib.parse
from datetime import datetime

# AWS Clients
REGION = os.environ.get("AWS_REGION", "us-east-1")
dynamodb = boto3.resource("dynamodb", region_name=REGION)
secrets = boto3.client("secretsmanager", region_name=REGION)
ses = boto3.client("ses", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)

# Environment variables
TABLE_NAME = os.environ["TABLE_NAME"]
LAT = os.environ["LAT"]
LON = os.environ["LON"]
SENDER_EMAIL = os.environ["SENDER_EMAIL"]
RECIPIENTS = [r.strip() for r in os.environ["RECIPIENTS"].split(",")]
WEATHER_SECRET_NAME = os.environ["WEATHER_SECRET_NAME"]
MODEL_ID = os.environ["MODEL_ID"]
FARM_NAME = os.environ.get("FARM_NAME", "Hansal Kasuweri – Pilot Farm")

# --------------------------
# Helper functions
# --------------------------
def _get_openweather_key():
    resp = secrets.get_secret_value(SecretId=WEATHER_SECRET_NAME)
    s = resp.get("SecretString")
    try:
        return json.loads(s)["api_key"]
    except Exception:
        return s.strip()

def _fetch_weather(lat, lon):
    """Fetch next-day weather forecast"""
    api_key = _get_openweather_key()
    base = "https://pro.openweathermap.org/data/2.5/forecast/climate"
    qs = urllib.parse.urlencode({
        "lat": str(lat),
        "lon": str(lon),
        "cnt": "1",
        "appid": api_key,
        "units": "metric"
    })
    url = f"{base}?{qs}"
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))

def _get_latest_entry():
    """Get the latest record from DynamoDB by date"""
    table = dynamodb.Table(TABLE_NAME)
    response = table.scan()
    items = response.get("Items", [])
    if not items:
        raise RuntimeError("No entries found in table.")
    items.sort(key=lambda x: x.get("date", "1970-01-01"), reverse=True)
    return items[0]

def _combine_data(entry, weather):
    """Merge the table data and weather forecast"""
    forecast = weather["list"][0]
    entry["weather_update"] = {
        "temp_day": forecast["temp"]["day"],
        "humidity": forecast["humidity"],
        "rain": forecast.get("rain", 0),
        "desc": forecast["weather"][0]["description"]
    }
    return entry

def _generate_summary(data):
    """Generate farmer-friendly summary using Bedrock"""
    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 350,
        "temperature": 0.3,
        "messages": [
            {"role": "user", "content": [{
                "type": "text",
                "text": (
                    "You are an agriculture assistant. Based on this data, "
                    "generate a short irrigation summary (<120 words):\n\n"
                    + json.dumps(data, indent=2)
                )
            }]}
        ]
    }

    resp = bedrock.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(prompt),
    )
    result = json.loads(resp["body"].read())
    return result["content"][0]["text"]

def _send_email(subject, text):
    ses.send_email(
        Source=SENDER_EMAIL,
        Destination={"ToAddresses": RECIPIENTS},
        Message={
            "Subject": {"Data": subject},
            "Body": {"Text": {"Data": text}},
        },
    )

# --------------------------
# Lambda entry point
# --------------------------
def lambda_handler(event, context):
    print("Fetching latest record from DynamoDB...")

    # 1️⃣ Get latest data
    latest = _get_latest_entry()

    # 2️⃣ Fetch real-time weather
    weather = _fetch_weather(LAT, LON)

    # 3️⃣ Merge
    combined = _combine_data(latest, weather)

    # 4️⃣ Generate Bedrock summary
    summary = _generate_summary(combined)

    # 5️⃣ Send email
    subject = f"[{FARM_NAME}] Daily Irrigation Summary – {datetime.now().date()}"
    _send_email(subject, summary)

    print("✅ Email sent successfully.")
    return {"ok": True, "subject": subject}

