import os
import time
from functools import wraps
from typing import Optional, Callable, Any, Dict

import jwt
from flask import request, jsonify

SECRET = os.getenv("SECRET_KEY", "dev-secret")
ALG = "HS256"


def make_token(user_id: int, role: str, hours: int = 12) -> str:
    payload = {
        "sub": int(user_id),
        "role": role,
        "exp": int(time.time()) + hours * 3600,
    }
    return jwt.encode(payload, SECRET, algorithm=ALG)


def parse_token() -> Optional[Dict[str, Any]]:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth.split(" ", 1)[1].strip()
    try:
        return jwt.decode(token, SECRET, algorithms=[ALG])
    except jwt.PyJWTError:
        return None


def require_auth(role: Optional[str] = None) -> Callable:
    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any):
            data = parse_token()
            if not data:
                return jsonify({"error": "Unauthorized"}), 401
            if role and data.get("role") != role:
                return jsonify({"error": "Forbidden"}), 403
            return fn(*args, **kwargs)

        return wrapper

    return deco
