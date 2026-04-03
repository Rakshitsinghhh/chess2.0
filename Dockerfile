FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app

# ✅ FIXED numpy version
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    "numpy<2" \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.30.6" \
    python-multipart==0.0.12 \
    python-chess==1.999 \
    tqdm==4.66.4 \
    "huggingface_hub==0.23.4" \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY model/          ./model/
COPY utils/          ./utils/
COPY inference/      ./inference/
COPY bot/            ./bot/
COPY nextjs_host/model_server/server.py ./model_server/server.py
COPY nextjs_host/model_server/start.sh  ./model_server/start.sh

RUN mkdir -p ./outputs/models

ENV MODEL_PATH=/chess2.0/outputs/models/latest.pt
ENV MODEL_URL=""
ENV FAST_MODE=1
ENV PORT=8001
ENV WORKERS=1

RUN chmod +x ./model_server/start.sh

EXPOSE 8001

WORKDIR /app/model_server
CMD ["sh", "start.sh"]