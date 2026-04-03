import { existsSync, readFileSync } from "fs";
import { join } from "path";
import type { InferenceSession } from "onnxruntime-web";

type OrtNS = typeof import("onnxruntime-web");

let ortModulePromise: Promise<OrtNS> | null = null;
let sessionPromise: Promise<InferenceSession> | null = null;

async function loadOrt(): Promise<OrtNS> {
  const ort = await import("onnxruntime-web");
  // Serverless-friendly: single thread; WASM from package dist (Node resolves ort.node entry).
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  const dist = join(process.cwd(), "node_modules", "onnxruntime-web", "dist");
  if (existsSync(dist)) {
    ort.env.wasm.wasmPaths = dist.endsWith("/") ? dist : `${dist}/`;
  }
  return ort;
}

export async function getOrtModule(): Promise<OrtNS> {
  if (!ortModulePromise) {
    ortModulePromise = loadOrt();
  }
  return ortModulePromise;
}

async function loadModelBuffer(): Promise<ArrayBuffer> {
  const localPath = join(process.cwd(), "public", "model.onnx");
  if (existsSync(localPath)) {
    const buf = readFileSync(localPath);
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  }

  const explicitUrl = process.env.MODEL_ONNX_URL;
  if (explicitUrl) {
    const res = await fetch(explicitUrl);
    if (!res.ok) {
      throw new Error(`MODEL_ONNX_URL fetch failed: ${res.status}`);
    }
    return res.arrayBuffer();
  }

  // Vercel: model is served from static CDN at /model.onnx but not shipped inside the function bundle.
  const vercel = process.env.VERCEL_URL;
  if (vercel) {
    const res = await fetch(`https://${vercel}/model.onnx`);
    if (res.ok) {
      return res.arrayBuffer();
    }
  }

  throw new Error(
    "Missing model.onnx: add nextjs_host/public/model.onnx (local), set MODEL_ONNX_URL, or deploy model.onnx to public and use Vercel (fetches https://$VERCEL_URL/model.onnx)."
  );
}

export async function getOnnxSession(): Promise<InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const ort = await getOrtModule();
      const buffer = await loadModelBuffer();
      return ort.InferenceSession.create(buffer, {
        executionProviders: ["wasm"]
      });
    })();
  }
  return sessionPromise;
}

export function resetOnnxSessionForTests(): void {
  ortModulePromise = null;
  sessionPromise = null;
}
