#!/bin/sh

echo "🚀 Starting Chess2.0 server..."

echo "📍 MODEL_PATH: $MODEL_PATH"

if [ ! -f "$MODEL_PATH" ]; then
  echo "❌ Model not found"

  if [ -n "$MODEL_URL" ]; then
    echo "📥 Downloading model..."
    mkdir -p $(dirname "$MODEL_PATH")
    wget -O "$MODEL_PATH" "$MODEL_URL"
  else
    echo "❌ MODEL_URL not set"
    exit 1
  fi
fi

echo "✅ Model ready"

uvicorn server:app --host 0.0.0.0 --port $PORT