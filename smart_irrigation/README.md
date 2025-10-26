# Smart Irrigation — Full-Stack Web App (React + Flask on EC2)

A production-ready Smart Irrigation application with a **React (Vite + TypeScript + Tailwind)** frontend and a **Flask** API, deployed on **Ubuntu EC2** behind **Nginx** and **Gunicorn**, using **AWS DynamoDB** for authentication/data and **IAM roles** (no access keys on disk).

In production, daily prediction JSON files written to **S3 bucket `weather-aus-data`** (keys like `irrigation_prediction_*.json`) trigger **Lambda `s3-prediction-to-fields-ddb`**, which updates the **`fields`** table with latest predictions and sensor snapshots.

---

## Features

- **Modern SPA**: React (Vite), Tailwind, React Router, Axios client with JWT interceptor  
- **API**: Flask with Blueprints, CORS, SPA fallback, Gunicorn `wsgi:app` entry  
- **Auth**: Email/password with Werkzeug hashing + JWT (HS256)  
- **Data**: DynamoDB tables
  - `users` — **PK** `user_id` (Number) + **GSI `email-index`** on `email` (String, lowercase)
  - `fields` — **PK** `field_id` (String) — ownership, metadata, daily sensor & prediction entries
- **Ops**: systemd service, Nginx reverse proxy, IAM role on EC2, AWS CLI v2 helper commands
- **Pipeline hook**: S3 → Lambda → DynamoDB (**`fields`**) for daily updates

---

## Architecture (high level)

```
Browser ──► Nginx :80 ─► Gunicorn (127.0.0.1:8000) ─► Flask API + serves /frontend/dist
                                 │
                                 ├─► DynamoDB (users, fields)
                                 └─► S3 weather-aus-data  ─► Lambda s3-prediction-to-fields-ddb ─► DynamoDB fields
```

---

## Project Structure

```
smart_irrigation/
├─ backend/                    # Flask API; can serve built SPA
│  ├─ app.py                   # registers blueprints; SPA fallback
│  ├─ wsgi.py                  # gunicorn entry: wsgi:app
│  ├─ requirements.txt
│  ├─ hansal/
│  │  ├─ config.py             # reads env vars
│  │  ├─ auth_utils.py         # JWT helpers
│  │  └─ routes/               # auth, fields, admin, predictions
│  └─ scripts/
│     ├─ seed_users.py         # inserts demo users (hashed)
│     └─ seed_fields.py
└─ frontend/                   # React + Vite + TS + Tailwind
   ├─ package.json
   └─ src/                     # pages, components, routing
```

---

## Quick Start (Local)

**Backend**
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# env (local):
export SECRET_KEY=dev
export AWS_REGION=us-east-1
export DDB_USERS_TABLE=users
export DDB_FIELDS_TABLE=fields
flask --app app run --debug   # http://127.0.0.1:5000
```

**Frontend**
```bash
cd frontend
npm install
npm run dev                   # http://127.0.0.1:5173
```

---

## DynamoDB Setup (Required)

```bash
# users with email-index
aws dynamodb create-table --region us-east-1 \
  --table-name users \
  --attribute-definitions AttributeName=user_id,AttributeType=N AttributeName=email,AttributeType=S \
  --key-schema AttributeName=user_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --global-secondary-indexes '[{"IndexName":"email-index","KeySchema":[{"AttributeName":"email","KeyType":"HASH"}],"Projection":{"ProjectionType":"ALL"}}]'

# fields
aws dynamodb create-table --region us-east-1 \
  --table-name fields \
  --attribute-definitions AttributeName=field_id,AttributeType=S \
  --key-schema AttributeName=field_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
```

**Seed demo users** (hashed passwords):
```bash
cd backend && source .venv/bin/activate
export AWS_REGION=us-east-1 DDB_USERS_TABLE=users
python scripts/seed_users.py
deactivate
```

---

## Production Build (Single Server)

```bash
# Build SPA
cd frontend && npm ci && npm run build

# Gunicorn (local smoke)
cd ../backend
source .venv/bin/activate
gunicorn -w 3 -b 127.0.0.1:8000 wsgi:app
```

Flask serves `/frontend/dist` and falls back to `index.html` for non-API routes.

---

## EC2 Deployment (Nginx + Gunicorn + systemd)

1) **EC2 prep (Ubuntu)**
```bash
sudo apt update && sudo apt install -y python3-venv python3-pip nginx unzip
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

2) **Upload & build**
```bash
# /srv/smart_irrigation/{frontend,backend}
cd /srv/smart_irrigation/frontend && npm ci && npm run build
cd /srv/smart_irrigation/backend
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && deactivate
```

3) **Runtime env** — `/etc/smart_irrigation.env`
```ini
SECRET_KEY=change-me
AWS_REGION=us-east-1
DDB_USERS_TABLE=users
DDB_FIELDS_TABLE=fields
```

4) **systemd** — `/etc/systemd/system/smart_irrigation.service`
```ini
[Unit] Description=Smart Irrigation (Flask via Gunicorn) After=network.target
[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/srv/smart_irrigation/backend
EnvironmentFile=/etc/smart_irrigation.env
ExecStart=/srv/smart_irrigation/backend/.venv/bin/gunicorn -w 3 -b 127.0.0.1:8000 wsgi:app
Restart=always
[Install] WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now smart_irrigation
```

5) **Nginx** — `/etc/nginx/sites-available/smart_irrigation`
```nginx
server {
  listen 80 default_server;
  server_name _;
  location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
  }
}
```
```bash
sudo ln -sf /etc/nginx/sites-available/smart_irrigation /etc/nginx/sites-enabled/smart_irrigation
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx
```

**Security Group**: open **HTTP :80**. **IAM**: attach a role with DynamoDB permissions to the instance (no keys).

---

## API (Quick Reference)

- **Auth**: `POST /api/auth/login`, `POST /api/auth/signup`, `GET /api/auth/_diag`  
- **Fields (user)**: `GET /api/fields/`, `GET /api/fields/<int:field_id>`  
- **Admin**: `GET /api/admin/users`, `GET /api/admin/users/<int:user_id>/fields`,  
  `POST /api/admin/users/<int:user_id>/fields/assign`,  
  `GET|POST /api/admin/fields`, `PUT|DELETE /api/admin/fields/<int:field_id>`  
- **Predictions**: `GET /api/predictions/`

---

## Troubleshooting

- **Login 500** → ensure `users` has **GSI `email-index`** and at least one user with **`password_hash`**; `email` must be lowercase  
- **Nginx won’t start** → free port 80 / remove `/run/nginx.pid`; `sudo nginx -t`  
- **AWS creds** → attach IAM role; `aws sts get-caller-identity` should succeed (no reboot needed)

---

## License


