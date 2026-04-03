import { existsSync, readFileSync } from "fs";
import { join } from "path";
import type { InferenceSession } from "onnxruntime-node";

let sessionPromise: Promise<InferenceSession> | null = null;

async function loadModelBuffer(): Promise<ArrayBuffer> {
  const localPath = join(process.cwd(), "public", "model.onnx");
  if (existsSync(localPath)) {
    const buf = readFileSync(localPath);
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  }

  const url = process.env.MODEL_ONNX_URL;
  if (url) {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`MODEL_ONNX_URL fetch failed: ${res.status}`);
    }
    return res.arrayBuffer();
  }

  throw new Error(
    "Missing model.onnx: add nextjs_host/public/model.onnx or set MODEL_ONNX_URL to a reachable HTTPS URL."
  );
}

export async function getOnnxSession(): Promise<InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const ort = await import("onnxruntime-node");
      const buffer = await loadModelBuffer();
      return ort.InferenceSession.create(buffer, { executionProviders: ["cpuExecutionProvider"] });
    })();
  }
  return sessionPromise;
}

export function resetOnnxSessionForTests(): void {
  sessionPromise = null;
}
