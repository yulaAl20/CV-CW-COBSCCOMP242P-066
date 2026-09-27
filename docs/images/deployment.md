# Deployment

## The constraint that shapes everything

The exported backbone is 81 MB. That single fact rules out several popular free
tiers and decides the ordering below.

| Limit | Value | Consequence |
|---|---|---|
| GitHub file size | 50 MB warning, 100 MB hard | weights cannot live in the repo |
| Memory at runtime | ~450 MB with one session loaded | 512 MB tiers work, one worker only |
| Container image | ~600 MB | fits everywhere; would be 4 GB with torch |
| Cold start | 25–40 s | first request after a sleep is slow |

**Recommended: Streamlit Community Cloud.** It is free, it deploys straight from
a GitHub repo with no Dockerfile, it handles Git LFS, and it redeploys on every
push. The other options below are for the FastAPI service, which is still in the
repository and still works.

### Which front end am I deploying?

The repository has two, over one engine:

| | Streamlit | FastAPI |
|---|---|---|
| Entry point | `streamlit_app.py` | `backend/main.py` |
| Hosting | Streamlit Cloud, no Docker | HF Spaces, Render, Fly, any VPS |
| Gives you | a working app | an app **and** a JSON API |
| Effort | push to GitHub | build and deploy a container |

Both import the same `backend/` package, so they cannot disagree about what the
model said. Pick Streamlit if you want it hosted today; pick FastAPI if you need
the API surface. Deploying both is fine — they are separate processes.

---

## Step 0 — publish the weights

Do this first. Every option needs it.

```bash
# from the machine that has your trained checkpoint
python scripts/export_deployment.py --checkpoint checkpoints/best.pt --out models
```

Then attach `backbone_fp32.onnx`, `heads.npz` and (optionally)
`backbone_cam.onnx` to a GitHub Release:

1. On your repo, go to **Releases → Draft a new release**
2. Tag `v1.0.0`, title "Trained model weights"
3. Drag the files from `models/` into the attachments box
4. Publish

Release assets have no practical size limit and do not count against repository
size. Copy the download URL — it looks like:

```
https://github.com/YOUR-USERNAME/dr-triage-app/releases/download/v1.0.0
```

---

## Option A — Streamlit Community Cloud (recommended)

Free, 1 GB RAM, no Dockerfile, redeploys on every push.

### 1. Push the repo to GitHub

Follow `docs/commit-plan.md`. Nothing extra is needed — `streamlit_app.py` is
already at the root, which is where Streamlit Cloud looks.

### 2. Deploy

Go to [share.streamlit.io](https://share.streamlit.io) → **New app**:

- **Repository**: `YOUR-USERNAME/dr-triage-app`
- **Branch**: `main`
- **Main file path**: `streamlit_app.py`

Click **Deploy**. The first build takes 3–6 minutes.

### 3. Point it at the weights

In the app's **Settings → Secrets**, paste:

```toml
DR_WEIGHTS_URL = "https://github.com/YOUR-USERNAME/dr-triage-app/releases/download/v1.0.0"
DR_ONNX_THREADS = "1"
```

The app downloads the weights on first run and caches them for the life of the
container. Until you set this it runs in demo mode, with a banner saying so.

### 4. Watch the memory

1 GB is the hard limit, and the app sits at roughly:

| | Resident |
|---|---|
| Python + Streamlit + numpy + OpenCV | ~350 MB |
| ONNX Runtime + the fp32 backbone session | ~280 MB |
| A second session for heatmaps (`backbone_cam.onnx`) | **~280 MB more** |

Two sessions plus the base is close enough to 1 GB to get the container killed
under load. If the app restarts when you open the Evidence tab, that is what
happened. Turn the heatmap off in Secrets:

```toml
DR_ENABLE_CAM = "0"
```

Grading, uncertainty and triage all still work; only the heatmap panel goes
away. Alternatively upload only `backbone_fp32.onnx` and `heads.npz` to the
release, and the app will never look for the CAM model.

### 5. Keep it awake

Free apps sleep after about a week of no visits and take ~40 s to wake. Open it
once before a demonstration.

---

## Running Streamlit locally

```bash
git clone https://github.com/YOUR-USERNAME/dr-triage-app.git
cd dr-triage-app

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python scripts/download_models.py  # or export your own

streamlit run streamlit_app.py
```

Opens on <http://localhost:8501>. Without weights it starts in demo mode.

---

## Option B — Hugging Face Spaces (FastAPI)

Free, 2 vCPU, 16 GB RAM, Git LFS included. Public Spaces do not sleep.

### 1. Create the Space

Go to [huggingface.co/new-space](https://huggingface.co/new-space):

- **Space name**: `dr-triage`
- **License**: MIT
- **SDK**: **Docker** → Blank
- **Hardware**: CPU basic (free)
- **Visibility**: Public

### 2. Push the code

```bash
git clone https://huggingface.co/spaces/YOUR-HF-USERNAME/dr-triage hf-space
cd hf-space

# copy everything except git metadata
rsync -av --exclude '.git' --exclude 'models/*.onnx' --exclude 'models/*.npz' \
      /path/to/dr-triage-app/ .
```

### 3. Add the Space header to `README.md`

Spaces reads configuration from a YAML block at the very top of `README.md`.
Without it the Space will not start.

```yaml
---
title: RetinaTriage
emoji: 👁️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: mit
---
```

### 4. Push the weights through LFS

```bash
git lfs install
git lfs track "models/*.onnx"
git lfs track "models/*.npz"
git add .gitattributes

# the repo .gitignore excludes these; force them in for the Space only
git add -f models/backbone_fp32.onnx models/heads.npz
git add .
git commit -m "feature: deploy RetinaTriage to Spaces"
git push
```

The build takes 5–10 minutes. Watch it under the **Logs** tab.

### 5. Confirm

```bash
curl https://YOUR-HF-USERNAME-dr-triage.hf.space/api/health
```

`"status": "ok"` means the weights loaded. `"status": "demo"` means they did
not — check that LFS actually uploaded them (`git lfs ls-files`) rather than
committing 130-byte pointer files.

---

## Option C — Render (FastAPI)

Free tier, 512 MB RAM. Sleeps after 15 minutes of inactivity and takes 30–50
seconds to wake, which is worth knowing before you demo it live.

1. [render.com](https://render.com) → **New → Web Service** → connect the repo
2. Settings:
   - **Environment**: Docker
   - **Instance type**: Free
   - **Health check path**: `/api/health`
3. Environment variables:
   - `DR_WEIGHTS_URL` = your release URL
   - `DR_ONNX_THREADS` = `1` (the free tier gives you a fraction of a core)
   - `DR_ALLOW_DEMO` = `0`
4. Deploy

Render does not run `download_models.py` automatically. Add it to the start
command:

```yaml
# render.yaml
services:
  - type: web
    name: retinatriage
    env: docker
    plan: free
    healthCheckPath: /api/health
    dockerCommand: >
      sh -c "python scripts/download_models.py || true &&
             uvicorn backend.main:app --host 0.0.0.0 --port $PORT --workers 1"
    envVars:
      - key: DR_WEIGHTS_URL
        sync: false
      - key: DR_ONNX_THREADS
        value: "1"
```

The `|| true` matters: if the download fails, you want the app up in demo mode
rather than a container that crash-loops.

---

## Option D — Fly.io (FastAPI)

Not free, but about $2/month for a machine that actually stays awake, and the
weights can live in a volume so they survive redeploys.

```bash
fly launch --no-deploy --name retinatriage
fly volume create models --size 1 --region lhr
```

```toml
# fly.toml
app = "retinatriage"
primary_region = "lhr"

[build]
  dockerfile = "Dockerfile"

[env]
  PORT = "8000"
  DR_MODEL_DIR = "/data/models"
  DR_ONNX_THREADS = "1"
  DR_ALLOW_DEMO = "0"

[[mounts]]
  source = "models"
  destination = "/data"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = false      # keep it warm; cold starts are 30 s
  min_machines_running = 1

  [http_service.http_options.response]
    # an 81 MB model on a shared CPU needs more than the 30 s default
    [http_service.http_options.response.headers]

[[vm]]
  memory = "1gb"
  cpu_kind = "shared"
  cpus = 1
```

```bash
fly deploy
fly ssh console -C "python scripts/download_models.py --out /data/models"
fly apps restart retinatriage
```

---

## Option E — Docker anywhere (FastAPI)

Any VPS, a lab machine, or your own laptop.

```bash
docker build -t retinatriage .
docker run -d -p 8000:8000 \
  -v "$(pwd)/models:/app/models" \
  -e DR_ALLOW_DEMO=0 \
  --name retinatriage \
  retinatriage
```

Mounting `models/` rather than baking it into the image keeps rebuilds fast and
lets you swap weights without rebuilding.

Behind nginx:

```nginx
server {
    listen 80;
    server_name retinatriage.example.com;

    client_max_body_size 15M;     # must exceed DR_MAX_UPLOAD_MB

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;   # cold start plus a slow first inference
    }
}
```

---

## Running the FastAPI service locally

```bash
git clone https://github.com/YOUR-USERNAME/dr-triage-app.git
cd dr-triage-app

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python scripts/download_models.py  # or export your own

python app.py
```

Open <http://localhost:8000>.

Without weights the app still starts, in demo mode, with a banner saying so.
That is enough to develop the interface against.

---

## Recording the demonstration video

The coursework asks for a hosted video of the prototype in operation. A
suggested three-minute run:

| Time | Show |
|---|---|
| 0:00–0:20 | the deployed URL, model-ready badge, the landing view |
| 0:20–0:50 | upload a healthy fundus — stage 0, routine rescreen |
| 0:50–1:30 | upload a referable case — the ladder, the confidence bars, the triage card |
| 1:30–2:00 | the pipeline tab: each preprocessing step in order |
| 2:00–2:20 | the evidence heatmap over lesions |
| 2:20–2:50 | scroll to the performance panel: confusion matrix, per-stage table, curves |
| 2:50–3:00 | `/docs` — the generated OpenAPI page |

Record with OBS Studio (free) or Loom, upload unlisted to YouTube, and put the
link in the report.

Say out loud that it is a prototype and not a medical device. It costs five
seconds and it is the kind of thing markers notice.

---

## Troubleshooting

**Status shows "demo" after deploying with weights**

The files are not where the app expects. Check:

```bash
curl https://your-app/api/health | python -m json.tool
```

The `model.error` field names the exact paths it looked in. On Spaces, the
usual cause is Git LFS pointer files being committed instead of the real
binaries — `git lfs ls-files` should list both weights.

**Container killed, exit code 137**

Out of memory. Almost always more than one uvicorn worker: each loads its own
81 MB session. Keep `--workers 1` and scale with containers.

**`ImportError: libGL.so.1`**

OpenCV's system dependencies are missing. The Dockerfile installs `libgl1` and
`libglib2.0-0`; if you wrote your own, add them.

**Upload returns 413**

The file exceeds `DR_MAX_UPLOAD_MB`, or a reverse proxy is capping it earlier.
Nginx's `client_max_body_size` defaults to 1 MB and must be raised above the
app's own limit.

**First request takes 40 seconds, the rest are fast**

Cold start on a sleeping free tier. Either accept it, ping `/api/health` every
10 minutes with a cron service, or move to a platform that stays warm.

**Streamlit app restarts when you open a heavy page**

Out of memory on the 1 GB tier, almost always two ONNX sessions. Set
`DR_ENABLE_CAM = "0"` in Secrets.

**Streamlit shows "Oh no. Error running app"**

Open **Manage app → logs** in the bottom right. The usual causes are a missing
`DR_WEIGHTS_URL` (harmless — it falls back to demo mode) and a dependency that
failed to install, which the log names.

**Heatmap tab is greyed out**

`backbone_cam.onnx` is not present. Grading works without it; only the
explanation panel needs it.
