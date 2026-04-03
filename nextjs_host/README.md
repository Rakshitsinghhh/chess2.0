# Next.js Hosting App (Play vs NN)

This folder contains a web app to play against your chess neural network from anywhere.

**Vercel:** in the project settings, set **Root Directory** to **`nextjs_host`** (this folder). The repository root may be `chess2.0`; Vercel must build from `nextjs_host` so `package.json` and `next.config.js` are used correctly.

Architecture:

- Next.js frontend (`app/page.tsx`) with drag-and-drop chessboard
- Inference is either **ONNX inside Node** (good for Vercel-style deploys) or a **FastAPI** Python server—see below.

## Option A — ONNX in Next.js (no Python server)

Export `latest.pt` → `public/model.onnx`:

```bash
cd /home/rakshit/projects1/chess2.0
. .venv/bin/activate
pip install onnx
python scripts/export_model_onnx.py
# writes nextjs_host/public/model.onnx
```

Env (`nextjs_host/.env.local`):

```bash
USE_ONNX=1
```

Optional: if the file is not on disk at runtime (some hosts), host `model.onnx` on any HTTPS URL and set:

```bash
MODEL_ONNX_URL=https://your-site.example/model.onnx
```

Then:

```bash
cd nextjs_host
npm install
npm run dev
```

**Vercel:** set `USE_ONNX=1`, commit or upload `public/model.onnx` (or use `MODEL_ONNX_URL`). The function bundles `onnxruntime-node` (native binaries); very large ONNX files or cold starts may hit platform limits—watch deployment size and timeout settings.

---

## Option B — Python model server (original flow)

From repo root:

```bash
cd /home/rakshit/projects1/chess2.0
. .venv/bin/activate
python3 -m pip install -r nextjs_host/model_server/requirements.txt
uvicorn nextjs_host.model_server.server:app --host 0.0.0.0 --port 8001
```

Health check:

```bash
curl http://127.0.0.1:8001/health
```

Keep `USE_ONNX=0` (or unset) and point Next.js at the server:

```bash
MODEL_SERVER_URL=http://127.0.0.1:8001
```

---

## Start Next.js app

```bash
cd /home/rakshit/projects1/chess2.0/nextjs_host
cp .env.example .env.local
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### Public tunnel (dev)

```bash
ngrok http 3000
```

---

## Notes

- Default PyTorch checkpoint path for export: `outputs/models/latest.pt` or `MODEL_PATH`.
- Python server: `FAST_MODE=1` by default; override with `FAST_MODE=0` for full `nn_predict` in `model_server/server.py`.
- ONNX path must match the same architecture and move encoding as training (`4672`-move policy, `20×8×8` board tensor).
