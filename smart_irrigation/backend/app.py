import os
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from hansal.config import Config
from hansal.routes.auth import bp as auth_bp
from hansal.routes.fields import bp as fields_bp
from hansal.routes.predictions import bp as predictions_bp
from hansal.routes.admin import bp as admin_bp 

# Built SPA path
FRONTEND_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))

app = Flask(__name__, static_folder=FRONTEND_DIST)  # let static_url_path default to /static
app.config.from_object(Config)
CORS(app, supports_credentials=True)

# API routes
app.register_blueprint(auth_bp)
app.register_blueprint(fields_bp)
app.register_blueprint(predictions_bp)
app.register_blueprint(admin_bp)

@app.get("/api/health")
def health():
    return jsonify({"ok": True})

# Serve SPA + assets with fallback
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_spa(path):
    candidate = os.path.join(FRONTEND_DIST, path)
    if path != "" and os.path.isfile(candidate):
        return send_from_directory(FRONTEND_DIST, path)
    index_path = os.path.join(FRONTEND_DIST, "index.html")
    if os.path.exists(index_path):
        return send_from_directory(FRONTEND_DIST, "index.html")
    return jsonify({
        "message": "Frontend not built yet. Run `npm run build` in frontend/ and try again, "
                   "or run the Vite dev server for local development."
    }), 500


@app.errorhandler(404)
def spa_fallback(e):
    if request.path.startswith("/api"):
        return e
    index_path = os.path.join(FRONTEND_DIST, "index.html")
    if os.path.exists(index_path):
        return send_from_directory(FRONTEND_DIST, "index.html")
    return e
