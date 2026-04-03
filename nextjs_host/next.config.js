/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    // Keep serverless bundle under Vercel’s 250 MB limit: onnxruntime-web ships extra
    // WebGL/WebGPU/JSEP artifacts we never use in API routes (WASM + ort.node only).
    outputFileTracingExcludes: {
      "*": [
        "node_modules/onnxruntime-web/dist/**/*.map",
        "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.*",
        "node_modules/onnxruntime-web/dist/ort.all*",
        "node_modules/onnxruntime-web/dist/ort.webgl*",
        "node_modules/onnxruntime-web/dist/ort.webgpu*",
        // Keep model off the Lambda filesystem trace; load via /model.onnx CDN or MODEL_ONNX_URL.
        "public/model.onnx"
      ]
    }
  }
};

module.exports = nextConfig;
